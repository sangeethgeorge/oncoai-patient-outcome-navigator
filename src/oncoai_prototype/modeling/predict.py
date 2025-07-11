import os
import mlflow
import pandas as pd
import glob
from datetime import datetime
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model, PythonModel
from mlflow.models import ModelSignature
from mlflow.types.utils import _infer_schema
import tempfile

# --- Custom utility imports ---
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

# --- Wrapper for inference model ---
class InferenceWrapper(PythonModel):
    def __init__(self, pyfunc_model):
        self.model = pyfunc_model

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(model_input)

def get_latest_model_run_id(model_name="OncoAICancerMortalityPredictor"):
    client = MlflowClient()
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            print(f"No versions found for model '{model_name}'")
            return None
        latest_version = sorted(all_versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
        return latest_version.run_id
    except Exception as e:
        print(f"Error retrieving model version for '{model_name}': {e}")
        return None

def run_inference():
    print("🔄 Loading cleaned dataset...")
    df = load_dataset(DATA_PATH)
    if df.empty:
        print("❌ No data available. Exiting inference.")
        return None, None, None

    run_id = get_latest_model_run_id()
    if not run_id:
        print("❌ Could not determine run ID. Exiting.")
        return None, None, None

    # Load PyFunc models
    model = load_model("models:/OncoAICancerMortalityPredictor/Latest")
    scaler = load_model("models:/onco_scaler/Latest")

    # Load feature names from artifact
    client = MlflowClient()
    features_dir = client.download_artifacts(run_id, "features")
    feature_files = glob.glob(os.path.join(features_dir, "*.txt"))
    if not feature_files:
        print("❌ No feature names file found in 'features' artifact.")
        return None, None, None

    with open(feature_files[0], "r") as f:
        feature_names = [line.strip() for line in f]

    # Prepare input
    drop_cols = ['icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob',
                 'dod', 'intime', 'outtime', 'icd9_code']
    df_filtered = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    X_raw = df_filtered.reindex(columns=feature_names)
    if X_raw.isnull().any().any():
        print("⚠️ Warning: NaNs detected in input data.")

    X_scaled_df = scaler.predict(X_raw)

    print("🧠 Running model inference...")
    preds_df = model.predict(X_scaled_df)
    df_results = df.join(preds_df)

    return df_results, model, X_scaled_df

# --- Execution ---
if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        df_results, original_model, X_scaled_df = run_inference()

        if df_results is not None:
            print("🧩 Generating SHAP explanations...")
            run_shap_explainer(
                model=original_model,
                X_scaled=X_scaled_df.values,
                X_df=X_scaled_df,
                output_dir=LOCAL_EXPLORATORY_DIR,
                X_raw=X_scaled_df,  # optional: pass X_raw instead
                log_to_mlflow=True,
                mlflow_path="inference_shap_plots_exploratory"
            )

            # Save example input/output schema
            input_example = X_scaled_df.iloc[:5]
            output_example = original_model.predict(input_example)

            signature = ModelSignature(
                inputs=_infer_schema(input_example),
                outputs=_infer_schema(output_example)
            )

            print("📦 Registering inference model...")
            wrapped_inference_model = InferenceWrapper(original_model)
            mlflow.pyfunc.log_model(
                artifact_path="onco_inference_model",
                python_model=wrapped_inference_model,
                input_example=input_example,
                signature=signature,
                registered_model_name="OncoAICancerMortalityPredictorInference"
            )

            mlflow.log_artifacts(LOCAL_EXPLORATORY_DIR, artifact_path="inference_shap_plots_exploratory")
            print("✅ Inference model logged and registered.")
