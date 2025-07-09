import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib # Needed if you manually save/load scaler as joblib artifact
from sklearn.preprocessing import StandardScaler # Needed if you load scaler object directly

# Ensure these utility functions are correctly implemented or imported within your project structure
# Assuming utils/model_utils.py, utils/input_utils.py, utils/shap_utils.py exist
# For this example, I'll put the helper functions directly in the file or assume they are available.

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True) # Ensure this directory exists for local artifact storage

# --- MLflow Configuration ---
# Set the tracking URI if it's not default (e.g., if you have a remote server)
# mlflow.set_tracking_uri("http://localhost:5000") # Uncomment if applicable

# --- Utility Functions (Adapted from predict.py and assumed input_utils.py) ---

def get_latest_model_run_id(experiment_name="OncoAI-Mortality-Prediction"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return None
    try:
        latest_model_version = client.get_latest_versions("OncoAICancerMortalityPredictor", stages=["None"])[0]
        run_id = latest_model_version.run_id
        print(f"Loading model from run associated with latest registered version: {run_id}")
        return run_id
    except Exception as e:
        print(f"Could not retrieve latest model version from registry: {e}")
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

def load_artifacts():
    model_name = "OncoAICancerMortalityPredictor"
    scaler_name = "OncoAIScaler"

    print(f"Loading registered model: models:/{model_name}/Latest")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Latest")

    print(f"Loading registered scaler: models:/{scaler_name}/Latest")
    scaler = mlflow.sklearn.load_model(f"models:/{scaler_name}/Latest")

    run_id_to_load = get_latest_model_run_id()
    if run_id_to_load is None:
        st.error("Could not determine run ID to load artifacts from. Exiting.")
        st.stop() # Stop the Streamlit app
        return None, None, None

    feature_names_artifact_path = "extra_files/feature_names.txt"
    client = mlflow.tracking.MlflowClient()

    try:
        local_feature_names_path = client.download_artifacts(run_id=run_id_to_load, path=feature_names_artifact_path, dst_path=MODEL_DIR)
    except Exception as e:
        st.error(f"Error downloading feature names artifact from run {run_id_to_load}, path {feature_names_artifact_path}: {e}")
        st.error("Ensure 'feature_names.txt' was logged to the 'extra_files' subdirectory within the run's artifacts.")
        st.stop() # Stop the Streamlit app
        return None, None, None

    feature_names = []
    with open(local_feature_names_path, "r") as f:
        for line in f:
            feature_names.append(line.strip())
    print("Model, scaler, and feature names loaded from MLflow.")
    return model, scaler, feature_names


def get_feature_template(feature_names):
    """
    Creates an empty DataFrame with the correct feature columns, initialized to zeros.
    This ensures consistency with the model's expected input.
    """
    # Create a dictionary with feature names as keys and 0.0 as values
    template_data = {feature: 0.0 for feature in feature_names}
    return pd.DataFrame([template_data])

def get_feature_info():
    """
    Provides display names, min/max, and default values for key features.
    This should be aligned with the actual features used by the model.
    """
    # This is a placeholder. You should populate this with actual features and their ranges.
    # The keys here must match the actual feature names from feature_names.txt
    return {
        'mean_hr': ("Mean Heart Rate (bpm)", 40.0, 180.0, 80.0),
        'min_temp': ("Min Temperature (C)", 30.0, 42.0, 37.0),
        'max_resp_rate': ("Max Respiratory Rate (breaths/min)", 10.0, 40.0, 20.0),
        'mean_spo2': ("Mean SpO2 (%)", 70.0, 100.0, 95.0),
        'mean_glucose': ("Mean Glucose (mg/dL)", 50.0, 300.0, 100.0),
        'min_wbc': ("Min WBC (K/uL)", 1.0, 30.0, 7.0),
        'max_creatinine': ("Max Creatinine (mg/dL)", 0.3, 10.0, 1.0),
        'mean_bun': ("Mean BUN (mg/dL)", 5.0, 100.0, 15.0),
        'min_platelets': ("Min Platelets (K/uL)", 50.0, 500.0, 150.0),
        'max_bilirubin': ("Max Bilirubin (mg/dL)", 0.1, 20.0, 1.0),
        # Add other top features identified by SHAP from feature_engineering.py here
        # Example: 'mean_sbp', 'mean_dbp', 'max_lactate', etc.
    }

def align_user_input(input_data, feature_template):
    """
    Aligns user input with the model's expected feature set.
    Fills in missing features with zeros and ensures correct column order.
    """
    user_df = pd.DataFrame([input_data])
    # Reindex to ensure all model features are present, filling missing with 0
    aligned_df = user_df.reindex(columns=feature_template.columns, fill_value=0.0)
    return aligned_df

def generate_shap_explanation(model, scaler, user_input_df, background_data):
    """
    Generates SHAP explanation for a single prediction.
    Args:
        model: Trained model.
        scaler: Fitted scaler.
        user_input_df (pd.DataFrame): The single row of user input (unscaled).
        background_data (pd.DataFrame): Background data for SHAP Explainer (scaled).
    Returns:
        shap.Explanation: SHAP explanation object.
    """
    explainer = shap.Explainer(model, background_data)
    user_input_scaled = scaler.transform(user_input_df)
    shap_values = explainer(user_input_scaled)
    # For single instance explanation, explainer returns an Explanation object directly.
    # No need to index with [0] for shap_values.values if it's already for a single instance.
    return shap_values

def plot_waterfall(shap_explanation):
    """
    Generates a SHAP waterfall plot and saves it to a temporary file.
    Returns the path to the saved plot.
    """
    shap.waterfall_plot(shap_explanation)
    plot_path = "shap_waterfall_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def create_shap_table(user_input_df, shap_explanation):
    """
    Creates a DataFrame summarizing SHAP contributions.
    """
    feature_names = user_input_df.columns
    # Ensure shap_explanation.values is 1D for a single prediction
    shap_values_array = shap_explanation.values.flatten()
    feature_values_array = user_input_df.iloc[0].values

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_values_array,
        'SHAP Value': shap_values_array
    })
    shap_df['Contribution Direction'] = shap_df['SHAP Value'].apply(lambda x: 'Increase Risk' if x > 0 else 'Decrease Risk')
    shap_df['Absolute SHAP Value'] = shap_df['SHAP Value'].abs()
    return shap_df.sort_values(by='Absolute SHAP Value', ascending=False)


# --- App Configuration ---
st.set_page_config(page_title="OncoAI Risk Dashboard", layout="wide")
st.title("\U0001F9EC OncoAI 30-Day Mortality Predictor")

st.markdown("""
Welcome to **OncoAI**, a research prototype built on the MIMIC-III dataset.

This tool predicts the **30-day mortality risk** for ICU patients with cancer based on early vitals and lab values. It also provides an explanation of how each feature contributes to the prediction using SHAP.

\u26A0\ufe0f *Note: This app is for demonstration and research purposes only, not for clinical use.*
""")

# Load artifacts including feature names
model, scaler, feature_names = load_artifacts()

# Ensure feature_names is not None before proceeding
if feature_names is None:
    st.error("Failed to load necessary artifacts for the dashboard. Please check MLflow logs.")
    st.stop() # Stop the app if essential components are missing

feature_template = get_feature_template(feature_names)
feature_info = get_feature_info() # This should contain info for features in feature_names

input_data = {}

# --- User Input Panel ---
st.subheader("\U0001F4DD Enter Patient Features")

# Filter feature_names to only include those that have info in feature_info for display
display_features = [f for f in feature_names if f in feature_info]

# Create two columns for input fields
col1, col2 = st.columns(2)
num_features_per_column = (len(display_features) + 1) // 2 # Roughly half in each column

for i, col in enumerate(display_features):
    target_column = col1 if i < num_features_per_column else col2
    with target_column:
        label, min_val, max_val, default_val = feature_info[col]
        input_data[col] = st.number_input(f"{label}", min_value=min_val, max_value=max_val, value=default_val, key=col)

user_input_df = align_user_input(input_data, feature_template)


# --- Predict and Explain ---
if st.button("\U0001F50D Predict 30-Day Mortality"):
    # Convert input_data dictionary to a DataFrame for scaling and prediction
    # Ensure the order of columns matches the training data by reindexing
    # Also, ensure only the actual features are passed for scaling/prediction
    X_user_input_for_prediction = user_input_df[feature_names]

    input_scaled = scaler.transform(X_user_input_for_prediction)
    prob = model.predict_proba(input_scaled)[0][1]
    st.success(f"\U0001FA78 **Predicted 30-Day Mortality Probability: {prob:.2%}**")

    with st.spinner("Explaining prediction..."):
        # For SHAP background, we can sample from a representative dataset or use feature_template
        # A more robust solution might involve loading a small subset of the training data
        # For demonstration, we'll create a background that aligns with expected features
        background_sample_size = 100
        # Create a background DataFrame with same columns as training data, filled with typical values or zeros
        background_data_df = pd.DataFrame(0.0, index=np.arange(background_sample_size), columns=feature_names)
        # You might want to populate background_data_df with more realistic sample data for better SHAP accuracy.
        # For now, a simple zero-filled background or sampling the feature_template is used.
        # To make it more robust, we should sample from actual training data if available.
        # For this example, let's assume we can generate a basic background from feature_template for simplicity.
        # This is a simplification; in a real app, load a small, representative sample of X_train_scaled.
        background_scaled = scaler.transform(feature_template.sample(background_sample_size, replace=True, random_state=42).reindex(columns=feature_names, fill_value=0.0))


        shap_expl = generate_shap_explanation(model, scaler, X_user_input_for_prediction, background_scaled)
        shap_fig_path = plot_waterfall(shap_expl)
        st.image(shap_fig_path, caption="SHAP Explanation", use_container_width=True)

        with st.expander("\U0001F4D8 How to interpret the SHAP Waterfall Plot"):
            st.markdown("""
            - The **SHAP waterfall plot** explains how each feature contributed to the prediction for this individual.
            - The **baseline value** (gray line) is the average model output (e.g., typical risk across all patients).
            - **Red bars** indicate features that **increase risk**.
            - **Blue bars** indicate features that **decrease risk**.
            - The **final prediction** is shown at the top, after applying all contributions.

            ---
            **Example:**
            If *Mean Glucose* is high and shown in red, it means this patient's glucose level increases their predicted mortality risk relative to average patients.
            """)

        # --- SHAP Table ---
        st.markdown("### \U0001F50D Feature-Level Breakdown")
        shap_contributions_df = create_shap_table(X_user_input_for_prediction, shap_expl)
        st.dataframe(shap_contributions_df, use_container_width=True, height=500)

st.markdown("---")
st.markdown("\U0001F9D1‍\U0001F4BB Developed by **Sangeeth George** — OncoAI (MIMIC-III)")