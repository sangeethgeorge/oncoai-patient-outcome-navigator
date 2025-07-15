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
import mlflow

# --- App Setup ---
st.set_page_config(page_title="OncoAI Risk Dashboard", layout="wide")
# --- Custom Styles for UI Enhancements ---
st.markdown("""
<style>
/* General expander header styling */
details summary {
    background-color: #f1f8ff !important;
    border-left: 6px solid #1e88e5;
    font-weight: 600;
    padding: 10px 14px;
    font-size: 1.15rem;
    border-radius: 4px;
    cursor: pointer;
}

details[open] summary {
    background-color: #e0f7fa !important;
    border-left: 6px solid #00acc1;
}

/* SHAP-specific override */
details[open] summary:has-text('How to Interpret') {
    background-color: #bbdefb !important;
    border-left: 6px solid #1976d2;
}

/* Expander content font size */
.streamlit-expanderContent {
    font-size: 1.05rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 OncoAI 30-Day Mortality Predictor")

st.markdown("""
Welcome to **OncoAI**, an AI-powered research tool built on MIMIC-III data.

Designed for clinical researchers and data scientists, OncoAI helps explore how early ICU labs and vitals can predict **30-day mortality** in cancer patients.

It provides:
- A risk estimate based on early ICU data
- Feature-level explanations using SHAP

⚠️ *For research and educational use only. Not intended for clinical decision-making.*
""")

with st.expander("🧭 How to Use"):
    st.markdown("""
    1. Enter patient vitals and lab values in the form below.
    2. Click **Predict 30-Day Mortality** to generate a risk estimate.
    3. View the SHAP explanation to understand how each feature influenced the prediction.
    4. Review the feature table to compare values and their contribution.
    """)

# --- Configuration ---
USE_GITHUB_MODE = os.environ.get("ONCOAI_MODE", "github").lower() == "github"

MODEL_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/model.pkl"
SCALER_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/scaler.pkl"
FEATURES_GITHUB_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models/feature_names.txt"

# --- Helper to use MLflow PyFunc correctly (only if MLflow is used) ---
def pyfunc_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    try:
        return model.predict(None, df)
    except AttributeError:
        # If it's a joblib model, assume direct predict call
        return pd.DataFrame({"predicted_probability": model.predict_proba(df)[:, 1]})


# --- Load model artifacts ---

# --- Helper to use MLflow PyFunc correctly (only if MLflow is used) ---
def pyfunc_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Assumes MLflow pyfunc model structure where predict method takes (context, dataframe)
        # and returns a dataframe (or series that can be converted).
        # The first argument (context) is None when called directly without MLflow tracking context.
        return model.predict(None, df)
    except AttributeError:
        # If it's a joblib-loaded scikit-learn model, assume direct predict_proba call
        # and return a DataFrame with a 'predicted_probability' column.
        return pd.DataFrame({"predicted_probability": model.predict_proba(df)[:, 1]})


# --- Load model artifacts ---
@st.cache_resource
def load_artifacts():
    # No debug print for USE_GITHUB_MODE here, as it's for internal logic
    try:
        if USE_GITHUB_MODE:
            st.info("🔄 Loading model from GitHub (Streamlit Cloud mode)")
            model_response = requests.get(MODEL_GITHUB_URL)
            scaler_response = requests.get(SCALER_GITHUB_URL)
            features_response = requests.get(FEATURES_GITHUB_URL)

            # Check for successful responses
            model_response.raise_for_status()
            scaler_response.raise_for_status()
            features_response.raise_for_status()

            model = joblib.load(BytesIO(model_response.content))
            scaler = joblib.load(BytesIO(scaler_response.content))
            feature_names = features_response.text.strip().splitlines()

        else:
            # This block will ONLY be executed if USE_GITHUB_MODE is False
            # Therefore, mlflow imports are only attempted if this path is taken.
            import mlflow.sklearn
            from mlflow.tracking import MlflowClient

            def get_latest_model_run_id(model_name="OncoAICancerMortalityPredictor"):
                client = MlflowClient()
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    return None
                latest = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)[0]
                return latest.run_id

            st.info("🧪 Loading model from MLflow (local dev mode)")
            model = mlflow.pyfunc.load_model("models:/OncoAICancerMortalityPredictor/Latest")
            scaler = mlflow.pyfunc.load_model("models:/onco_scaler/Latest")
            run_id = get_latest_model_run_id()
            feature_path = MlflowClient().download_artifacts(run_id, "features/feature_names.txt")
            with open(feature_path, "r") as f:
                feature_names = [line.strip() for line in f if line.strip()]

    except requests.exceptions.RequestException as req_e:
        st.error(f"🚨 Network or file not found error during GitHub artifact loading: {req_e}")
        st.stop()
    except Exception as e:
        # This will catch joblib errors, or any other unexpected errors
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
        return pyfunc_predict(model, df)["predicted_probability"].values

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

# --- Load Artifacts ---
model, scaler, feature_names = load_artifacts()
feature_template = get_feature_template(feature_names)
feature_info = get_feature_info()
background_df = get_feature_template(feature_names) # dummy background for SHAP

# --- User Input UI ---
st.subheader("📋 Enter Patient Features")
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


# --- Prediction ---
if st.button("🔍 Predict 30-Day Mortality"):
    input_scaled_df = pyfunc_predict(scaler, user_input_df)
    background_scaled_df = pyfunc_predict(scaler, background_df)

    pred_result = pyfunc_predict(model, input_scaled_df)
    prob = pred_result["predicted_probability"].iloc[0]

    # Styled Risk Box
    color = "#d9534f" if prob > 0.5 else "#f0ad4e" if prob > 0.2 else "#5cb85c"
    st.markdown(f"""
    <div style='background-color:{color}; padding: 20px; border-radius: 8px; text-align: center; color: white; font-size: 20px; font-weight: bold;'>
        🧮 Predicted 30-Day Mortality Risk: {prob:.2%}
    </div>
    """, unsafe_allow_html=True)

    st.caption("This means the model estimates a {:.0f}% chance of mortality within 30 days based on the inputs.".format(prob * 100))

    with st.spinner("🧠 Generating SHAP Explanation..."):
        shap_expl = generate_shap_explanation(model, input_scaled_df, background_scaled_df)

    # --- SHAP Explanation ---
    st.markdown("## 📈 SHAP Feature Contributions")
    shap_ax = shap.plots.waterfall(shap_expl[0], show=False)
    st.pyplot(shap_ax.figure)
    
    st.markdown("🔍 Want help interpreting the SHAP plot?")
    with st.expander("ℹ️ How to Interpret This Plot"):
        st.info("""
        SHAP (SHapley Additive exPlanations) shows how each input moved the risk from the average:

        - 🔴 **Red** → Increased predicted risk  
        - 🔵 **Blue** → Decreased predicted risk  
        - 📏 **Length** → Impact size  
        - ⚪ **Base value** is the average risk across patients

        **Example:**  
        - `+0.12` → 12% increase in risk  
        - `-0.05` → 5% decrease in risk  
        """)

    # --- SHAP Table ---
    st.markdown("## 📌 Feature-Level Breakdown")
    contrib_df = create_shap_table(user_input_df, shap_expl)
    contrib_df["Direction"] = contrib_df["Contribution Direction"].map({
        "Increase Risk": "🔺 Increase Risk",
        "Decrease Risk": "🔻 Decrease Risk"
    })

    styled_df = contrib_df[["Feature", "Value", "SHAP Value", "Direction"]].style\
        .format({"SHAP Value": "{:+.4f}", "Value": "{:.2f}"})\
        .bar(subset=["SHAP Value"], align="zero", color=['#d65f5f', '#5fba7d'])\
        .set_properties(**{'text-align': 'left'})\
        .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

    st.dataframe(styled_df, use_container_width=True, height=550)

# --- Footer ---
st.markdown("---")
st.markdown("Developed by **Sangeeth George** — [LinkedIn](https://www.linkedin.com/in/sangeeth-george/) | OncoAI (MIMIC-III)")