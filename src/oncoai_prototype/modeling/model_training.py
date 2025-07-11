import os
import time
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import shutil
import mlflow
import mlflow.pyfunc
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType
from mlflow.types.utils import _infer_schema

# --- Custom utility imports ---
from oncoai_prototype.utils.io_utils import load_dataset
from oncoai_prototype.utils.preprocessing import train_test_impute_split, scale_features
from oncoai_prototype.utils.leakage import check_for_leakage

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")

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

# --- Model Training ---
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    return model, y_pred, y_prob

# --- Pipeline Orchestration ---
def run_training_pipeline():
    print("Loading ML-ready cohort...")
    df = load_dataset(DATA_PATH)
    if df.empty:
        print("Dataset not found or empty.")
        return None

    columns_to_drop = [
        'icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob',
        'dod', 'intime', 'outtime', 'icd9_code'
    ]
    df_features = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    target_column = 'mortality_30d'

    print("Splitting + imputing data...")
    X_train, X_test, y_train, y_test = train_test_impute_split(df_features, target_col=target_column)

    print("Checking for data leakage...")
    X_train_processed = check_for_leakage(X_train, target_col=target_column)
    X_test_processed = X_test.reindex(columns=X_train_processed.columns)

    print("Scaling features...")
    X_train_array, X_test_array, scaler = scale_features(X_train_processed, X_test_processed)
    X_train_df = pd.DataFrame(X_train_array, columns=X_train_processed.columns)
    X_test_df = pd.DataFrame(X_test_array, columns=X_test_processed.columns)

    print("Training logistic regression model...")
    model, y_pred, y_prob = train_logistic_regression(X_train_df, y_train, X_test_df, y_test)

    print("Model training complete.")
    return model, y_test, y_pred, y_prob, X_train_processed, scaler, X_train_df

# --- Main Execution ---
if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name=f"logreg_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_training_pipeline()
        if result is None:
            print("Training aborted.")
            exit()

        model, y_test, y_pred, y_prob, X_train_orig, scaler, X_train_scaled_df = result

        auc = roc_auc_score(y_test, y_prob)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_metric("roc_auc", auc)

        # Save feature names
        feature_names = X_train_orig.columns.tolist()
        model_artifact_dir = os.path.join(PROJECT_ROOT, "tmp_onco_model_artifacts")
        os.makedirs(model_artifact_dir, exist_ok=True)

        features_txt_path = os.path.join(model_artifact_dir, "feature_names.txt")
        with open(features_txt_path, "w") as f:
            for col in feature_names:
                f.write(f"{col}\n")

        # Log feature_names.txt as general run artifact (optional)
        mlflow.log_artifact(features_txt_path, artifact_path="features")
        print(f"📁 Feature names logged to: {mlflow.get_artifact_uri('features/feature_names.txt')}")

        # Log classifier model as PyFunc with embedded artifact
        input_schema = Schema([ColSpec(DataType.double, col) for col in X_train_scaled_df.columns])
        output_schema = Schema([
            ColSpec(DataType.long, "predicted_mortality_30d"),
            ColSpec(DataType.double, "predicted_probability")
        ])
        model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        wrapped_model = SklearnWrapper(model)
        mlflow.pyfunc.log_model(
            artifact_path="onco_model",
            python_model=wrapped_model,
            signature=model_signature,
            input_example=X_train_scaled_df.iloc[:5],
            artifacts={"feature_names": features_txt_path},  # ✅ Embed inside model
            registered_model_name="OncoAICancerMortalityPredictor"
        )

        # Log scaler as PyFunc
        scaler_input_example = X_train_orig.iloc[:5]
        scaler_output_df = pd.DataFrame(scaler.transform(scaler_input_example), columns=X_train_orig.columns)
        scaler_signature = ModelSignature(
            inputs=_infer_schema(scaler_input_example),
            outputs=_infer_schema(scaler_output_df)
        )

        wrapped_scaler = ScalerWrapper(scaler)
        mlflow.pyfunc.log_model(
            artifact_path="onco_scaler_model",
            python_model=wrapped_scaler,
            signature=scaler_signature,
            input_example=scaler_input_example,
            registered_model_name="onco_scaler"
        )

        # Optional: Clean up local artifacts directory
        shutil.rmtree(model_artifact_dir, ignore_errors=True)

        print("✅ Training run and model registration complete.")
