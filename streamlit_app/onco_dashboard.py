import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from datetime import datetime
import shap
from mlflow.tracking import MlflowClient
import glob

# --- App Setup ---
st.set_page_config(page_title="OncoAI Risk Dashboard", layout="wide")
st.title(":microscope: OncoAI 30-Day Mortality Predictor")

st.markdown("""
Welcome to **OncoAI**, a research prototype built on the MIMIC-III dataset.

This tool predicts the **30-day mortality risk** for ICU patients with cancer based on early vitals and lab values. It also provides an explanation of how each feature contributes to the prediction using SHAP.

:warning: *Note: This app is for demonstration and research purposes only, not for clinical use.*
""")

# --- MLflow Configuration ---
# Ensure MLflow experiment is set
mlflow.set_experiment("OncoAI-Mortality-Prediction")


# --- Helper Functions to Load Artifacts ---
@st.cache_resource
def load_model_and_scaler():
    """Loads the latest registered model and scaler from MLflow."""
    try:
        model = mlflow.sklearn.load_model("models:/OncoAICancerMortalityPredictor/Latest")
        scaler = mlflow.pyfunc.load_model("models:/onco_scaler/Latest")
        st.success("Model and Scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler from MLflow: {e}")
        return None, None

@st.cache_data
def load_feature_names():
    """Loads feature names from the latest MLflow run that logged them."""
    client = MlflowClient()
    try:
        st.info("Attempting to load feature names...")
        all_versions = client.search_model_versions(f"name='OncoAICancerMortalityPredictor'")
        if not all_versions:
            st.error("No versions found for model 'OncoAICancerMortalityPredictor'.")
            return None

        latest_version = sorted(all_versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
        run_id = latest_version.run_id
        st.info(f"Found latest model version: {latest_version.version}, Run ID: {run_id}")

        st.info("Downloading 'features' artifacts...")
        # This downloads the *contents* of the 'features' artifact path to a local temp dir
        features_dir = client.download_artifacts(run_id, "features")
        st.info(f"Downloaded 'features' artifacts to local directory: {features_dir}")

        # List contents of the downloaded directory to confirm the file is there
        downloaded_contents = os.listdir(features_dir)
        st.info(f"Contents of downloaded 'features' directory: {downloaded_contents}")

        feature_files = glob.glob(os.path.join(features_dir, "*.txt"))
        st.info(f"Files found by glob in {features_dir}: {feature_files}")

        if not feature_files:
            st.error(f"No .txt file found in 'features' artifact. Glob result was empty: {feature_files}")
            return None

        # Use the first .txt file found
        selected_file = feature_files[0]
        st.info(f"Using feature file: {os.path.basename(selected_file)}")

        with open(selected_file, "r") as f:
            feature_names = [line.strip() for line in f if line.strip()]
        st.success(f"Successfully loaded {len(feature_names)} feature names.")
        return feature_names

    except Exception as e:
        st.error(f"Error retrieving feature names from MLflow artifacts: {e}")
        return None


# Placeholder for run_shap_explainer as it's not directly provided in the context of the dashboard app's imports
# We need a simplified version for the dashboard that can compute SHAP values and potentially plot them.
# The original shap_utils.py file has a run_shap_explainer, but it saves plots to disk.
# For Streamlit, we need to return the shap_values object and plot it directly in the app.

def compute_shap_values(model, X_scaled_background, X_scaled_instance):
    """Computes SHAP values for a given instance."""
    explainer = shap.Explainer(model, X_scaled_background)
    shap_values = explainer(X_scaled_instance)
    return shap_values

# --- Load Artifacts ---
model, scaler = load_model_and_scaler()
feature_names = load_feature_names()

if model is None or scaler is None or feature_names is None:
    st.info("Attempting to load MLflow artifacts. Please ensure MLflow Tracking Server is running and models are registered.")
    st.stop()


feature_template = pd.DataFrame([{f: 0.0 for f in feature_names}])

# --- Define default feature metadata for input panel ---
def get_feature_info():
    # These are illustrative. In a real app, you might load this from a config or schema.
    return {
        'mean_hr': ("Mean Heart Rate (bpm)", 40.0, 180.0, 80.0),
        'min_temp': ("Min Temperature (C)", 30.0, 42.0, 37.0),
        'max_resp_rate': ("Max Respiratory Rate", 10.0, 40.0, 20.0),
        'mean_spo2': ("Mean SpO2 (%)", 70.0, 100.0, 95.0),
        'mean_glucose': ("Mean Glucose (mg/dL)", 50.0, 300.0, 100.0),
        'min_wbc': ("Min WBC (K/uL)", 1.0, 30.0, 7.0),
        'max_creatinine': ("Max Creatinine (mg/dL)", 0.3, 10.0, 1.0),
        'mean_bun': ("Mean BUN (mg/dL)", 5.0, 100.0, 15.0),
        'min_platelets': ("Min Platelets (K/uL)", 50.0, 500.0, 150.0),
        'max_bilirubin': ("Max Bilirubin (mg/dL)", 0.1, 20.0, 1.0),
    }

feature_info = get_feature_info()
input_data = {}

st.subheader(":memo: Enter Patient Features")
display_features = [f for f in feature_names if f in feature_info]
col1, col2 = st.columns(2)

for i, feature in enumerate(display_features):
    col = col1 if i < len(display_features) // 2 else col2
    with col:
        label, min_val, max_val, default_val = feature_info[feature]
        input_data[feature] = st.number_input(label, min_value=min_val, max_value=max_val, value=default_val)

user_input_df = pd.DataFrame([input_data]).reindex(columns=feature_names, fill_value=0.0)

if st.button(":mag: Predict 30-Day Mortality"):
    input_scaled = scaler.predict(user_input_df) # Use scaler.predict for PyFunc model
    # Convert input_scaled numpy array back to DataFrame for model.predict
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
    prob = model.predict_proba(input_scaled_df)[0][1] # Use model.predict_proba for PyFunc model
    st.success(f":dna: **Predicted 30-Day Mortality Probability: {prob:.2%}**")

    with st.spinner("Explaining prediction with SHAP..."):
        # For SHAP, it's often better to use a small sample of the training data as background
        # Here we'll generate a dummy background for demonstration if actual training data isn't available
        # In a real scenario, you'd load a sample of your training X_train_scaled_df
        if len(feature_names) > 0:
            background_data = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
            background_scaled = scaler.predict(background_data)
            background_scaled_df = pd.DataFrame(background_scaled, columns=feature_names) # Convert to DataFrame
        else:
            background_scaled_df = pd.DataFrame() # Empty DataFrame if no features

        if not background_scaled_df.empty:
            shap_values_obj = compute_shap_values(model, background_scaled_df, input_scaled_df)

            st.markdown("### :bar_chart: SHAP Plot Type")
            plot_type = st.radio("Choose SHAP explanation plot:", ["Waterfall", "Force", "Beeswarm"])

            if plot_type == "Waterfall":
                # shap_values_obj.values[0] is for the first (and only) instance
                # shap_values_obj.base_values[0] is for the base value of the first instance
                fig = shap.plots.waterfall(shap_values_obj[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
            elif plot_type == "Force":
                st.set_option("deprecation.showPyplotGlobalUse", False)
                # Ensure the feature names are passed to force_plot
                shap.force_plot(shap_values_obj.base_values[0], shap_values_obj.values[0], feature_names=feature_names, matplotlib=True, show=False)
                st.pyplot(bbox_inches='tight')
            elif plot_type == "Beeswarm":
                # For beeswarm, we usually show multiple samples.
                # If only one prediction, beeswarm might not be as illustrative,
                # but we can still show it for the single instance.
                # shap.plots.beeswarm expects a shap.Explanation object or array of SHAP values.
                fig, ax = plt.subplots()
                shap.plots.beeswarm(shap_values_obj, ax=ax, show=False)
                plt.tight_layout()
                st.pyplot(fig)

            with st.expander(":book: How to interpret SHAP Plots"):
                st.markdown("""
                - **Waterfall Plot** shows how each feature pushes the model output up or down for one prediction.
                - **Force Plot** provides a compact visualization of feature contributions.
                - **Beeswarm Plot** aggregates SHAP values across features if multiple samples are shown.
                """)

            st.markdown("### :mag: Feature-Level Breakdown")
            contrib_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': user_input_df.values[0],
                'SHAP Value': shap_values_obj.values[0], # Access the values for the first instance
            })
            contrib_df['Direction'] = contrib_df['SHAP Value'].apply(lambda x: 'Increase Risk' if x > 0 else 'Decrease Risk')
            contrib_df['|SHAP|'] = contrib_df['SHAP Value'].abs()
            st.dataframe(contrib_df.sort_values('|SHAP|', ascending=False), use_container_width=True, height=500)
        else:
            st.warning("Could not generate SHAP explanations due to missing background data or features.")


st.markdown("---")
st.markdown(":technologist: Developed by **Sangeeth George** — OncoAI (MIMIC-III)")