import os
import joblib
import requests
from io import BytesIO
import mlflow
import pandas as pd
from datetime import datetime
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model, PythonModel
from mlflow.models import ModelSignature
from mlflow.types.utils import _infer_schema

# --- Custom utility imports ---
from oncoai_prototype.utils.io_utils import load_dataset
from oncoai_prototype.utils.preprocessing import preprocess_for_inference
from oncoai_prototype.utils.shap_utils import run_shap_explainer

# --- Configuration ---
USE_GITHUB_MODE = os.environ.get("ONCOAI_MODE", "mlflow") == "github"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")
LOCAL_SHAP_BASE_DIR = os.path.join(PROJECT_ROOT, "reports", "shap_plots/inference")
os.makedirs(LOCAL_SHAP_BASE_DIR, exist_ok=True)

MODEL_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/model.pkl"
SCALER_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/scaler.pkl"
FEATURES_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/feature_names.txt"

# --- MLflow Wrapper for Inference Model ---
class InferenceWrapper(PythonModel):
    def __init__(self, pyfunc_model):
        self.model = pyfunc_model
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(model_input)

def get_latest_model_run_id(model_name="OncoAICancerMortalityPredictor"):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    latest = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
    return latest.run_id

def load_artifacts():
    if USE_GITHUB_MODE:
        print("🌐 Loading artifacts from GitHub...")
        model = joblib.load(BytesIO(requests.get(MODEL_GITHUB_URL).content))
        scaler = joblib.load(BytesIO(requests.get(SCALER_GITHUB_URL).content))
        feature_names = requests.get(FEATURES_GITHUB_URL).text.strip().splitlines()
    else:
        print("📦 Loading artifacts from MLflow registry...")
        model = load_model("models:/OncoAICancerMortalityPredictor/Latest")
        scaler = load_model("models:/onco_scaler/Latest")

        run_id = get_latest_model_run_id()
        if run_id is None:
            raise RuntimeError("Could not find latest model run.")

        client = MlflowClient()
        feature_path = client.download_artifacts(run_id, "features/feature_names.txt")
        with open(feature_path, "r") as f:
            feature_names = [line.strip() for line in f]

    return model, scaler, feature_names

def run_inference():
    print("🔄 Loading dataset...")
    df = load_dataset(DATA_PATH)
    if df.empty:
        print("❌ Dataset is empty.")
        return None, None, None

    model, scaler, feature_names = load_artifacts()

    # Prepare input
    X_raw = preprocess_for_inference(df.drop(columns=["mortality_30d"], errors="ignore")).reindex(columns=feature_names)
    if X_raw.isnull().any().any():
        raise ValueError("Input is missing model features: "
                         f"{X_raw.columns[X_raw.isnull().all()].tolist()}")

    X_scaled = scaler.predict(X_raw)

    print("🧠 Running model inference...")
    predictions = model.predict(X_scaled)
    results_df = df.join(predictions)

    return results_df, model, X_scaled

# --- Execution ---
if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        df_results, model, X_scaled = run_inference()

        if df_results is not None:
            print("🔍 Running SHAP explainability...")
            run_shap_explainer(
                model=model,
                X_scaled=X_scaled.values,
                X_df=X_scaled,
                output_dir=LOCAL_SHAP_BASE_DIR,
                X_raw=X_scaled,
                log_to_mlflow=True,
                mlflow_path="inference_shap_plots_exploratory"
            )

            input_example = X_scaled.iloc[:5]
            output_example = model.predict(input_example)

            signature = ModelSignature(
                inputs=_infer_schema(input_example),
                outputs=_infer_schema(output_example)
            )

            print("📦 Logging inference model...")
            wrapped_model = InferenceWrapper(model)
            mlflow.pyfunc.log_model(
                artifact_path="onco_inference_model",
                python_model=wrapped_model,
                input_example=input_example,
                signature=signature,
                registered_model_name="OncoAICancerMortalityPredictorInference"
            )

            mlflow.log_artifacts(LOCAL_SHAP_BASE_DIR, artifact_path="inference_shap_plots_exploratory")
            print("✅ Inference complete and logged.")
