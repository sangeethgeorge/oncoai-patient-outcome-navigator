# src/oncoai_prototype/modeling/predict.py

import os
import mlflow
import pandas as pd
import glob
from datetime import datetime
from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model

# Custom utility imports
from oncoai_prototype.utils.io_utils import load_dataset
from oncoai_prototype.utils.shap_utils import run_shap_explainer

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")
LOCAL_SHAP_BASE_DIR = os.path.join(PROJECT_ROOT, "reports", "shap_plots")
LOCAL_FINAL_SUMMARY_DIR = os.path.join(LOCAL_SHAP_BASE_DIR, "final")
LOCAL_EXPLORATORY_DIR = os.path.join(LOCAL_SHAP_BASE_DIR, "inference")

os.makedirs(LOCAL_FINAL_SUMMARY_DIR, exist_ok=True)
os.makedirs(LOCAL_EXPLORATORY_DIR, exist_ok=True)

def get_latest_model_run_id(model_name="OncoAICancerMortalityPredictor"):
    client = MlflowClient()

    # Search all model versions registered under the given name
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            print(f"No versions found for model '{model_name}'")
            return None

        # Sort by creation time, descending (latest first)
        latest_version = sorted(all_versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
        return latest_version.run_id

    except Exception as e:
        print(f"Error retrieving model version for '{model_name}': {e}")
        return None

def run_inference():
    print("Loading cleaned dataset...")
    df = load_dataset(DATA_PATH)
    if df.empty:
        print("No data available. Exiting inference.")
        return

    run_id = get_latest_model_run_id()
    if not run_id:
        print("Could not determine run ID. Exiting.")
        return

    model = load_model("models:/OncoAICancerMortalityPredictor/Latest")
    scaler = mlflow.pyfunc.load_model("models:/onco_scaler/Latest")

    # Load feature names from MLflow artifact
    client = MlflowClient()
    features_dir = client.download_artifacts(run_id, "features")

    # Find the .txt file inside the "features" directory
    feature_files = glob.glob(os.path.join(features_dir, "*.txt"))
    if not feature_files:
        print("No feature names file found in 'features' artifact.")
        return

    with open(feature_files[0], "r") as f:
        feature_names = [line.strip() for line in f]

    df_filtered = df.drop(columns=[col for col in [ 'icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob',
        'dod', 'intime', 'outtime', 'icd9_code'] if col in df.columns], errors='ignore')
    X_raw = df_filtered.reindex(columns=feature_names)

    if X_raw.isnull().any().any():
        print("Warning: NaNs detected in input data.")

    X_scaled_array = scaler.predict(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=feature_names, index=X_raw.index)

    print("Running model inference...")
    y_pred = model.predict(X_scaled_df)

    # Optionally also get predicted probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled_df)[:, 1]
        results_df = pd.DataFrame({
            "predicted_mortality_30d": y_pred,
            "predicted_probability": y_prob
        }, index=X_scaled_df.index)
    else:
        results_df = pd.DataFrame({
            "predicted_mortality_30d": y_pred
        }, index=X_scaled_df.index)

    df = df.join(results_df)

    print("Generating SHAP explanations...")
    run_shap_explainer(
        model=model,
        X_scaled=X_scaled_df.values,
        X_df=X_scaled_df,
        output_dir=LOCAL_EXPLORATORY_DIR,
        X_raw=X_raw,
        log_to_mlflow=True,
        mlflow_path="inference_shap_plots_exploratory"
    )

if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        run_inference()
        mlflow.log_artifacts(LOCAL_EXPLORATORY_DIR, artifact_path="inference_shap_plots_exploratory")
    print("Inference run logged to MLflow.")