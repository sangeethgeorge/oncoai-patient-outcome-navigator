# src/oncoai_prototype/modeling/predict.py

import os
import mlflow
import pandas as pd
import joblib # Needed if you manually save/load scaler as joblib artifact
from sklearn.preprocessing import StandardScaler # Needed if you load scaler object directly
from datetime import datetime

# Custom utility imports
from oncoai_prototype.utils.io_utils import load_dataset # Only for loading the cleaned dataset
from oncoai_prototype.utils.preprocessing import preprocess_for_inference # Might not be directly used if df is already ML-ready
from oncoai_prototype.utils.shap_utils import run_shap_explainer # Ensure this import is correct

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")

# SHAP_PLOTS_DIR will be used for exploratory plots and the base for final summary plots if not specified otherwise
# Your shap_util.py defines paths like "reports/shap_plots/final" relative to the current working directory,
# so we need to pass a more absolute path or manage current directory.
# Let's adjust SHAP_PLOTS_DIR to be the base for both exploratory and final summary plots
# within the local file system.
LOCAL_SHAP_BASE_DIR = os.path.join(PROJECT_ROOT, "reports", "shap_plots")
LOCAL_FINAL_SUMMARY_DIR = os.path.join(LOCAL_SHAP_BASE_DIR, "final")
LOCAL_EXPLORATORY_DIR = os.path.join(LOCAL_SHAP_BASE_DIR, "inference")

os.makedirs(LOCAL_FINAL_SUMMARY_DIR, exist_ok=True)
os.makedirs(LOCAL_EXPLORATORY_DIR, exist_ok=True)


# --- MLflow Configuration ---
# Set the tracking URI if it's not default (e.g., if you have a remote server)
# mlflow.set_tracking_uri("http://localhost:5000") # Uncomment if applicable

def get_latest_model_run_id(experiment_name="OncoAI-Mortality-Prediction"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return None
    try:
        # Fetch the latest version of the registered model
        # The 'stages=["None"]' is deprecated but included for compatibility if your MLflow version warns without it.
        # It's better to omit it if possible for newer MLflow versions.
        latest_model_version = client.get_latest_versions("OncoAICancerMortalityPredictor", stages=["None"])[0]
        run_id = latest_model_version.run_id
        print(f"Loading model from run associated with latest registered version: {run_id}")
        return run_id
    except Exception as e:
        print(f"Could not retrieve latest model version from registry: {e}")
        # Fallback to searching runs by start time if registry access fails or model not found in registry
        print("Falling back to searching runs by start_time for the latest run.")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs:
            print(f"Loading model from latest run: {runs[0].info.run_id}")
            return runs[0].info.run_id
        else:
            print(f"No runs found for experiment '{experiment_name}'.")
            return None


def run_inference():
    print("Loading cleaned dataset...")
    df = load_dataset(DATA_PATH)
    if df.empty:
        print("No data available. Exiting inference.")
        return

    # Get the run ID associated with the latest registered model
    run_id_to_load = get_latest_model_run_id()
    if run_id_to_load is None:
        print("Could not determine run ID to load artifacts from. Exiting inference.")
        return

    # --- Load model from Model Registry ---
    model_name = "OncoAICancerMortalityPredictor"
    scaler_name = "OncoAIScaler"

    print(f"Loading registered model: models:/{model_name}/Latest")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Latest")

    print(f"Loading registered scaler: models:/{scaler_name}/Latest")
    scaler = mlflow.sklearn.load_model(f"models:/{scaler_name}/Latest")
    
    # --- Load feature names from artifact ---
    feature_names_artifact_path = "extra_files/feature_names.txt" 
    client = mlflow.tracking.MlflowClient()
    
    try:
        local_feature_names_path = client.download_artifacts(run_id=run_id_to_load, path=feature_names_artifact_path)
    except Exception as e:
        print(f"Error downloading feature names artifact from run {run_id_to_load}, path {feature_names_artifact_path}: {e}")
        print("Ensure 'feature_names.txt' was logged to the 'extra_files' subdirectory within the run's artifacts.")
        return # Exit if we can't get feature names

    feature_names = []
    with open(local_feature_names_path, "r") as f:
        for line in f:
            feature_names.append(line.strip())
    print("Model, scaler, and feature names loaded from MLflow.")

    # --- Prepare input features ---
    # Ensure any non-feature identifier columns are removed from 'df' if they exist.
    # This should mirror the dropping done during training.
    columns_to_exclude_from_features = ['subject_id', 'hadm_id', 'icustay_id']
    df_filtered = df.drop(columns=[col for col in columns_to_exclude_from_features if col in df.columns], errors='ignore')

    # Create the X_for_inference DataFrame using only the relevant features
    X_for_inference = df_filtered[[col for col in feature_names if col in df_filtered.columns]]
    X_for_inference = X_for_inference.reindex(columns=feature_names)

    # Check for NaNs and warn if any (assuming `onco_features_cleaned.parquet` should be clean)
    if X_for_inference.isnull().any().any():
        print("⚠️ Warning: NaNs detected in X_for_inference. Ensure data is properly imputed before scaling.")

    print("Scaling features...")
    X_scaled_array = scaler.transform(X_for_inference)
    
    # Convert the scaled array back to a DataFrame with feature names for prediction
    # This addresses the sklearn UserWarning about missing feature names.
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=feature_names, index=X_for_inference.index)


    print("Running model inference...")
    y_pred = model.predict(X_scaled_df) # Use X_scaled_df
    y_prob = model.predict_proba(X_scaled_df)[:, 1] # Use X_scaled_df

    results_df = df.copy() # Keep original df for context/IDs if needed in results
    results_df['predicted_mortality_30d'] = y_pred
    results_df['predicted_probability'] = y_prob
    print(results_df[['predicted_mortality_30d', 'predicted_probability']].head())

    print("Generating SHAP explanations...")
    # Call run_shap_explainer with the correct arguments based on shap_util.py
    run_shap_explainer(
        model=model,
        X_scaled=X_scaled_df, # Pass the scaled DataFrame with feature names to SHAP
        X_raw=X_for_inference, # Pass the original (unscaled) DataFrame for plotting context
        final_summary_path=LOCAL_FINAL_SUMMARY_DIR,
        exploratory_path=LOCAL_EXPLORATORY_DIR,
        log_to_mlflow=True # Set to True to log the main SHAP summary plot to MLflow
    )


# --- Main ---
if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        run_inference()
        # The run_shap_explainer function itself logs `shap_summary_overall.png` if log_to_mlflow is True.
        # You might still want to log the entire directory for review, or skip this if all desired plots are individually logged.
        # As your shap_util.py specifies "exploratory_path" plots are NOT logged to MLflow by default,
        # you can log that entire directory here if you want them in MLflow too.
        # Or you can remove the line below if you only want the 'final' summary plot in MLflow.
        mlflow.log_artifacts(LOCAL_EXPLORATORY_DIR, artifact_path="inference_shap_plots_exploratory") # Log exploratory plots
    print("Inference run logged to MLflow.")