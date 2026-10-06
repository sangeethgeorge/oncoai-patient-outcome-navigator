import os
import json
import shutil
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, classification_report
import mlflow
import mlflow.pyfunc
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType
from mlflow.types.utils import _infer_schema
from mlflow.tracking import MlflowClient


# --- Custom utility imports ---
from oncoai_prototype.utils.io_utils import load_dataset
from oncoai_prototype.utils.preprocessing import NON_FEATURE_COLS, grouped_stratified_split
from oncoai_prototype.utils.leakage import check_for_leakage, assert_unique_ids, assert_no_group_overlap

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")
OOF_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "oof_predictions.parquet")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TARGET = "mortality_30d"
GROUP_COL = "subject_id"
BASELINE_FEATURES = ["age"]  # known at ICU admission, no lab/vital data
SEED = 42
TOP_N = 10

# --- MLflow PyFunc Model Wrappers ---
class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        preds = self.model.predict(model_input)
        probs = self.model.predict_proba(model_input)[:, 1]
        return pd.DataFrame({
            "predicted_mortality_30d": preds,
            "predicted_probability": probs
        })

class ScalerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler):
        self.scaler = scaler

    def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler.transform(model_input)
        return pd.DataFrame(scaled, columns=model_input.columns)

# --- Building blocks (each fit only ever sees training rows) ---
def impute_with_train_medians(X_train: pd.DataFrame, X_other: pd.DataFrame):
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians), X_other.fillna(medians)

def select_features_shap(X_train: pd.DataFrame, y_train: pd.Series, top_n=TOP_N, seed=SEED) -> list:
    """Rank features by mean |SHAP| of an XGBoost model fit on the training rows only."""
    booster = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                eval_metric='logloss', random_state=seed)
    booster.fit(X_train, y_train)
    shap_values = shap.TreeExplainer(booster)(X_train)
    mean_abs = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_train.columns)
    return mean_abs.sort_values(ascending=False).head(top_n).index.tolist()

def fit_logreg(X_train: pd.DataFrame, y_train: pd.Series, seed=SEED):
    scaler = StandardScaler().fit(X_train)
    X_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(X_scaled, y_train)
    return scaler, model

def predict_proba(scaler, model, X: pd.DataFrame) -> np.ndarray:
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return model.predict_proba(X_scaled)[:, 1]

# --- Evaluation ---
def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))

def calibration_slope(y_true, y_prob):
    """Slope of a logistic recalibration of y on logit(p); 1.0 is perfect, <1 means overfit."""
    p = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(penalty=None).fit(logit, y_true).coef_[0][0])

def evaluate(y_true, y_prob) -> dict:
    lo, hi = bootstrap_auc_ci(y_true, y_prob)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "roc_auc_ci_low": lo,
        "roc_auc_ci_high": hi,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration_slope": calibration_slope(y_true, y_prob),
    }

def grouped_cv_auc(X_raw: pd.DataFrame, y: pd.Series, groups: pd.Series, top_n=TOP_N, seed=SEED, n_splits=5):
    """Cross-validated AUC on the training set, re-running imputation and feature selection per fold."""
    aucs = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in splitter.split(X_raw, y, groups):
        X_tr, X_va = impute_with_train_medians(X_raw.iloc[tr], X_raw.iloc[va])
        features = select_features_shap(X_tr, y.iloc[tr], top_n, seed)
        scaler, model = fit_logreg(X_tr[features], y.iloc[tr], seed)
        aucs.append(roc_auc_score(y.iloc[va], predict_proba(scaler, model, X_va[features])))
    return float(np.mean(aucs)), float(np.std(aucs))

def cross_fitted_predictions(X_raw: pd.DataFrame, y: pd.Series, groups: pd.Series, ids: pd.Series,
                             top_n=TOP_N, seed=SEED, n_splits=5) -> pd.DataFrame:
    """Out-of-fold risk for every stay: each stay is scored by a model (imputation, feature selection
    and fit) that never saw that patient. Used for risk tiers and observed/expected ratios, not for
    the published test metrics."""
    parts = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(splitter.split(X_raw, y, groups)):
        assert_no_group_overlap(groups.iloc[tr], groups.iloc[va], GROUP_COL)
        X_tr, X_va = impute_with_train_medians(X_raw.iloc[tr], X_raw.iloc[va])
        features = select_features_shap(X_tr, y.iloc[tr], top_n, seed)
        scaler, model = fit_logreg(X_tr[features], y.iloc[tr], seed)
        parts.append(pd.DataFrame({"icustay_id": ids.iloc[va].to_numpy(), "fold": fold,
                                   "pred_prob": predict_proba(scaler, model, X_va[features])}))
    return pd.concat(parts, ignore_index=True)

# --- Pipeline Orchestration ---
def run_training_pipeline(df: pd.DataFrame, top_n=TOP_N, seed=SEED) -> dict:
    assert_unique_ids(df, "icustay_id")
    df = df.dropna(subset=[TARGET])
    X_all = df.drop(columns=[TARGET] + [c for c in NON_FEATURE_COLS if c in df.columns])
    X_all = check_for_leakage(X_all.select_dtypes(include=[np.number]), target_col=TARGET)
    y = df[TARGET].astype(int)
    groups = df[GROUP_COL]

    cohort = {"n_stays": len(df), "n_events": int(y.sum()), "prevalence": float(y.mean())}
    print(f"Cohort: {cohort['n_stays']} ICU stays, {cohort['n_events']} deaths ({cohort['prevalence']:.1%})")

    print("Splitting by patient (stratified, grouped)...")
    train_idx, test_idx = grouped_stratified_split(y, groups, test_size=0.2, random_state=seed)
    assert_no_group_overlap(groups.loc[train_idx], groups.loc[test_idx], GROUP_COL)
    X_train, X_test = impute_with_train_medians(X_all.loc[train_idx], X_all.loc[test_idx])
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    print("Cross-validating on the training split...")
    cv_mean, cv_std = grouped_cv_auc(X_all.loc[train_idx], y_train, groups.loc[train_idx], top_n, seed)

    print("Selecting features on the training split...")
    features = select_features_shap(X_train, y_train, top_n, seed)
    print(f"Top {top_n} SHAP features: {features}")

    print("Training logistic regression model...")
    scaler, model = fit_logreg(X_train[features], y_train, seed)
    y_prob = predict_proba(scaler, model, X_test[features])
    print(classification_report(y_test, (y_prob >= 0.5).astype(int), zero_division=0))

    baseline_features = [f for f in BASELINE_FEATURES if f in X_train.columns]
    b_scaler, b_model = fit_logreg(X_train[baseline_features], y_train, seed)
    baseline_prob = predict_proba(b_scaler, b_model, X_test[baseline_features])

    print("Cross-fitting out-of-fold predictions for every stay (analytics only)...")
    oof = cross_fitted_predictions(X_all, y, groups, df["icustay_id"], top_n, seed)
    y_oof = y.set_axis(df["icustay_id"]).loc[oof["icustay_id"]]
    oof_eval = evaluate(y_oof, oof["pred_prob"])

    metrics = {
        **cohort,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "cv_roc_auc_mean": cv_mean,
        "cv_roc_auc_std": cv_std,
        **{f"test_{k}": v for k, v in evaluate(y_test, y_prob).items()},
        **{f"baseline_{k}": v for k, v in evaluate(y_test, baseline_prob).items()},
        "oof_roc_auc": oof_eval["roc_auc"],
        "oof_brier": oof_eval["brier"],
        "oof_calibration_slope": oof_eval["calibration_slope"],
    }
    print(f"Test ROC-AUC {metrics['test_roc_auc']:.3f} "
          f"(95% CI {metrics['test_roc_auc_ci_low']:.3f}-{metrics['test_roc_auc_ci_high']:.3f}); "
          f"baseline {baseline_features}: {metrics['baseline_roc_auc']:.3f}; "
          f"5-fold CV {cv_mean:.3f} ± {cv_std:.3f}")

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "baseline_features": baseline_features,
        "metrics": metrics,
        "X_train": X_train[features],
        "oof_predictions": oof,
    }

def feature_ranges(X_train: pd.DataFrame) -> dict:
    """Aggregate input ranges for the dashboard widgets (no row-level values)."""
    q = X_train.quantile([0.05, 0.5, 0.95])
    return {c: {"low": round(float(q.loc[0.05, c]), 2), "default": round(float(q.loc[0.5, c]), 2),
                "high": round(float(q.loc[0.95, c]), 2)} for c in X_train.columns}

def export_model_artifacts_locally(model_path, scaler_path, features_path, output_dir=MODELS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    # Strip file:// prefix if present
    def to_local_path(uri):
        return uri.replace("file://", "") if uri.startswith("file://") else uri

    shutil.copy(to_local_path(model_path), os.path.join(output_dir, "model.pkl"))
    shutil.copy(to_local_path(scaler_path), os.path.join(output_dir, "scaler.pkl"))
    shutil.copy(to_local_path(features_path), os.path.join(output_dir, "feature_names.txt"))
    print(f"🗂 Artifacts copied to local folder: {output_dir}/")


# --- Main Execution ---
if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    if df.empty:
        raise SystemExit("Dataset not found or empty. Run feature_engineering.py first.")

    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name=f"logreg_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_training_pipeline(df, top_n=TOP_N, seed=SEED)
        model, scaler, features, metrics = result["model"], result["scaler"], result["features"], result["metrics"]
        X_train = result["X_train"]

        mlflow.log_params({"model_type": "LogisticRegression", "scaler": "StandardScaler",
                           "feature_selection": "XGBoost mean |SHAP| on train split",
                           "top_n": TOP_N, "seed": SEED, "split": "StratifiedGroupKFold by subject_id",
                           "baseline_features": ",".join(result["baseline_features"])})
        mlflow.log_metrics(metrics)

        model_artifact_dir = os.path.join(PROJECT_ROOT, "tmp_onco_model_artifacts")
        os.makedirs(model_artifact_dir, exist_ok=True)
        features_txt_path = os.path.join(model_artifact_dir, "feature_names.txt")
        with open(features_txt_path, "w") as f:
            for col in features:
                f.write(f"{col}\n")
        mlflow.log_artifact(features_txt_path, artifact_path="features")

        X_train_scaled_df = pd.DataFrame(scaler.transform(X_train), columns=features)
        model_signature = ModelSignature(
            inputs=Schema([ColSpec(DataType.double, col) for col in features]),
            outputs=Schema([ColSpec(DataType.long, "predicted_mortality_30d"),
                            ColSpec(DataType.double, "predicted_probability")]),
        )
        # No input_example: it would store patient-level rows in the run artifacts.
        mlflow.pyfunc.log_model(
            artifact_path="onco_model",
            python_model=SklearnWrapper(model),
            signature=model_signature,
            artifacts={"feature_names": features_txt_path},
            registered_model_name="OncoAICancerMortalityPredictor"
        )

        scaler_signature = ModelSignature(inputs=_infer_schema(X_train.iloc[:0]),
                                          outputs=_infer_schema(X_train_scaled_df.iloc[:0]))
        mlflow.pyfunc.log_model(
            artifact_path="onco_scaler_model",
            python_model=ScalerWrapper(scaler),
            signature=scaler_signature,
            registered_model_name="onco_scaler"
        )

        run_id = mlflow.active_run().info.run_id
        client = MlflowClient()
        export_model_artifacts_locally(
            model_path=client.download_artifacts(run_id, "onco_model/python_model.pkl"),
            scaler_path=client.download_artifacts(run_id, "onco_scaler_model/python_model.pkl"),
            features_path=client.download_artifacts(run_id, "features/feature_names.txt"),
        )
        with open(os.path.join(MODELS_DIR, "feature_ranges.json"), "w") as f:
            json.dump(feature_ranges(X_train), f, indent=2)
        # Row-level and derived from MIMIC: stays in the git-ignored data/processed/ folder.
        result["oof_predictions"].to_parquet(OOF_PATH, index=False)
        with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
            json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, f, indent=2)

        shutil.rmtree(model_artifact_dir, ignore_errors=True)
        print("✅ Training run and model registration complete.")
