import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import shap
import joblib
import requests
from io import BytesIO
from datetime import datetime

# --- App Setup ---
st.set_page_config(page_title="OncoAI Risk Dashboard", layout="wide")
st.title(":microscope: OncoAI 30-Day Mortality Predictor")

st.markdown("""
Welcome to **OncoAI**, a research prototype built on the MIMIC-III dataset.

This tool predicts the **30-day mortality risk** for ICU patients with cancer based on early vitals and lab values. It also provides an explanation of how each feature contributes to the prediction using SHAP.

⚠️ warning: *Note: This app is for demonstration and research purposes only, not for clinical use.*
""")

st.info("""
### How to Use This Tool
1. Enter patient vitals and lab values using the form below.
2. Click **"Predict 30-Day Mortality"** to generate a risk estimate.
3. Review the **SHAP explanation plot** to understand which features increased or decreased the predicted risk.
4. Explore the **feature-level table** to see individual contributions and input values.
""")

# --- MLflow Configuration ---

# --- Artifact Loading ---
USE_GITHUB_MODE = True

if not USE_GITHUB_MODE:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient


MODEL_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/model.pkl"
SCALER_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/scaler.pkl"
FEATURES_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/feature_names.txt"

if not USE_GITHUB_MODE:
    @st.cache_data
    def get_latest_model_run_id(model_name="OncoAICancerMortalityPredictor"):
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        latest = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
        return latest.run_id



@st.cache_resource
def load_artifacts():
    try:
        if USE_GITHUB_MODE:
            st.info("🔄 Loading model from GitHub (Streamlit Cloud mode)")
            model = joblib.load(BytesIO(requests.get(MODEL_GITHUB_URL).content))
            scaler = joblib.load(BytesIO(requests.get(SCALER_GITHUB_URL).content))
            feature_names = requests.get(FEATURES_GITHUB_URL).text.strip().splitlines()
        else:
            st.info("🧪 Loading model from MLflow (local dev mode)")
            model = mlflow.pyfunc.load_model("models:/OncoAICancerMortalityPredictor/Latest")
            scaler = mlflow.pyfunc.load_model("models:/onco_scaler/Latest")

            client = MlflowClient()
            run_id = client.search_model_versions("name='OncoAICancerMortalityPredictor'")[0].run_id
            feature_path = client.download_artifacts(run_id, "features/feature_names.txt")
            with open(feature_path, "r") as f:
                feature_names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        st.error(f"🚨 Failed to load model artifacts: {e}")
        st.stop()

    return model, scaler, feature_names


# --- Utility Functions ---
def get_feature_template(feature_names):
    return pd.DataFrame([{feature: 0.0 for feature in feature_names}])

def get_feature_info():
    return {
        'mean_glucose': ("Mean Glucose (mg/dL)", 50.0, 300.0, 100.0),
        'min_heart_rate': ("Min Heart Rate (bpm)", 40.0, 180.0, 60.0),
        'mean_urea_nitrogen': ("Mean BUN (mg/dL)", 5.0, 100.0, 18.0),
        'slope_chloride': ("Slope of Chloride", -5.0, 5.0, 0.0),
        'max_bicarbonate': ("Max Bicarbonate (mEq/L)", 10.0, 40.0, 24.0),
        'mean_rdw': ("Mean RDW (%)", 10.0, 20.0, 13.5),
        'min_white_blood_cells': ("Min WBC (K/uL)", 1.0, 30.0, 6.0),
        'mean_mchc': ("Mean MCHC (g/dL)", 30.0, 38.0, 34.0),
        'mean_chloride': ("Mean Chloride (mEq/L)", 95.0, 115.0, 105.0),
        'slope_bicarbonate': ("Slope of Bicarbonate", -5.0, 5.0, 0.0)
    }


def align_user_input(input_data, feature_template):
    return pd.DataFrame([input_data]).reindex(columns=feature_template.columns, fill_value=0.0)

def generate_shap_explanation(model, user_input_scaled, background_scaled):
    def predict_fn(x):
        df = pd.DataFrame(x, columns=user_input_scaled.columns)
        return model.predict(df)["predicted_probability"].values

    explainer = shap.Explainer(predict_fn, background_scaled)
    return explainer(user_input_scaled)  

def create_shap_table(user_input_df, shap_explanation):
    feature_names = user_input_df.columns
    shap_values_array = shap_explanation.values.flatten()
    feature_values_array = user_input_df.iloc[0].values

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_values_array,
        'SHAP Value': shap_values_array
    })
    shap_df['Contribution Direction'] = shap_df['SHAP Value'].apply(lambda x: 'Increase Risk' if x > 0 else 'Decrease Risk')
    shap_df['|SHAP|'] = shap_df['SHAP Value'].abs()
    return shap_df.sort_values(by='|SHAP|', ascending=False)

# --- Load and Display ---
model, scaler, feature_names = load_artifacts()
feature_template = get_feature_template(feature_names)
feature_info = get_feature_info()

st.subheader(":memo: Enter Patient Features")
input_data = {}
col1, col2 = st.columns(2)
display_features = [f for f in feature_names if f in feature_info]
missing = [f for f in feature_names if f not in feature_info]
if missing:
    st.warning(f"The following features are missing from feature_info: {missing}")

for i, feature in enumerate(display_features):
    col = col1 if i < len(display_features) // 2 else col2
    label, min_val, max_val, default_val = feature_info[feature]
    input_data[feature] = col.number_input(label, min_value=min_val, max_value=max_val, value=default_val)

user_input_df = align_user_input(input_data, feature_template)

if st.button(":mag: Predict 30-Day Mortality"):
    # Scale input and background
    input_scaled_df = scaler.predict(user_input_df)
    background_df = feature_template.sample(n=100, replace=True, random_state=42)
    background_scaled_df = scaler.predict(background_df)

    # Predict
    pred_result = model.predict(input_scaled_df)
    prob = pred_result["predicted_probability"].iloc[0]

    st.success(f"Predicted 30-Day Mortality Probability: {prob:.2%}")

    with st.spinner("Generating SHAP Explanation..."):
        shap_expl = generate_shap_explanation(model, input_scaled_df, background_scaled_df)

        st.markdown("### :bar_chart: SHAP Feature Contributions")
        shap_ax = shap.plots.waterfall(shap_expl[0], show=False)
        fig = shap_ax.figure
        st.pyplot(fig)


        with st.expander("📘book: How to interpret this plot"):
            st.markdown("""
            - **Red bars** increase risk.
            - **Blue bars** decrease risk.
            - The prediction builds from the baseline (average).
            """)

        contrib_df = create_shap_table(user_input_df, shap_expl)
        st.markdown("### :mag: Feature-Level Breakdown")
        st.dataframe(contrib_df, use_container_width=True, height=500)

st.markdown("---")
st.markdown("Developed by **Sangeeth George** — https://www.linkedin.com/in/sangeeth-george/ OncoAI (MIMIC-III)")