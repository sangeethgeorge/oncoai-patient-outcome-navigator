import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import hashlib
import matplotlib.pyplot as plt
import shap
import joblib
import requests
from io import BytesIO

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

ARTIFACT_BASE_URL = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/models"
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# --- Helper to use MLflow PyFunc correctly (only if MLflow is used) ---
def pyfunc_predict(model, df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        # Plain scikit-learn estimator
        return pd.DataFrame({"predicted_probability": model.predict_proba(df)[:, 1]})
    if hasattr(model, "metadata"):
        # mlflow.pyfunc.PyFuncModel loaded from the registry: predict(data)
        return pd.DataFrame(model.predict(df))
    # Raw PythonModel wrapper unpickled from GitHub artifacts: predict(context, data)
    return model.predict(None, df)


# --- Load model artifacts ---
def fetch_artifact(name: str) -> bytes:
    """One file from models/: local copy in mlflow (dev) mode, this repo's main branch in github mode."""
    if not USE_GITHUB_MODE:
        with open(os.path.join(LOCAL_MODELS_DIR, name), "rb") as f:
            return f.read()
    response = requests.get(f"{ARTIFACT_BASE_URL}/{name}", timeout=15)
    response.raise_for_status()
    return response.content

@st.cache_data(ttl=300, show_spinner=False)
def artifact_release() -> str:
    """Fingerprint of the published model release, re-checked every 5 minutes. Used as the
    cache key for load_artifacts, so a newly pushed model replaces the cached one."""
    try:
        if USE_GITHUB_MODE:
            return hashlib.sha256(fetch_artifact("feature_names.txt") + fetch_artifact("metrics.json")).hexdigest()
        from mlflow.tracking import MlflowClient
        versions = MlflowClient().search_model_versions("name='OncoAICancerMortalityPredictor'")
        return str(max(int(v.version) for v in versions))
    except Exception as e:
        st.error(f"🚨 Could not reach the model artifacts: {e}")
        st.stop()

@st.cache_resource(max_entries=1, show_spinner="Loading model...")
def load_artifacts(release: str):
    """Model, scaler, feature list, input ranges and metrics, loaded together from one release."""
    try:
        if USE_GITHUB_MODE:
            model = joblib.load(BytesIO(fetch_artifact("model.pkl")))
            scaler = joblib.load(BytesIO(fetch_artifact("scaler.pkl")))
            feature_names = fetch_artifact("feature_names.txt").decode().split()
        else:
            import mlflow.pyfunc
            from mlflow.tracking import MlflowClient
            model = mlflow.pyfunc.load_model(f"models:/OncoAICancerMortalityPredictor/{release}")
            scaler = mlflow.pyfunc.load_model("models:/onco_scaler/Latest")
            run_id = MlflowClient().get_model_version("OncoAICancerMortalityPredictor", release).run_id
            with open(MlflowClient().download_artifacts(run_id, "features/feature_names.txt")) as f:
                feature_names = f.read().split()
        ranges = json.loads(fetch_artifact("feature_ranges.json"))
        metrics = json.loads(fetch_artifact("metrics.json"))
    except Exception as e:
        st.error(f"🚨 Failed to load model artifacts: {e}")
        st.stop()
    return model, scaler, feature_names, ranges, metrics


# --- Utility Functions ---
def get_feature_template(feature_names):
    return pd.DataFrame([{feature: 0.0 for feature in feature_names}])

def feature_label(feature):
    stat, _, measure = feature.partition("_")
    return f"{stat.capitalize()} {measure.replace('_', ' ')}"

def get_feature_info(feature_names, ranges):
    info = {}
    for feature in feature_names:
        r = ranges.get(feature)
        if r is None:
            continue
        span = max(r["high"] - r["low"], 1.0)
        info[feature] = (feature_label(feature), r["low"] - span, r["high"] + span, r["default"])
    return info

def align_user_input(input_data, feature_template):
    aligned = pd.DataFrame([input_data]).reindex(columns=feature_template.columns)
    if aligned.isnull().values.any():
        raise ValueError(f"No input for model features: {aligned.columns[aligned.isnull().any()].tolist()}")
    return aligned

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
model, scaler, feature_names, feature_ranges, metrics = load_artifacts(artifact_release())
feature_template = get_feature_template(feature_names)
feature_info = get_feature_info(feature_names, feature_ranges)
# SHAP background: the training-set mean, which is all zeros after standard scaling.
background_scaled_df = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

if metrics:
    with st.expander("📊 Model performance (held-out test set)"):
        st.markdown(f"""
        - **Cohort:** {metrics['n_stays']:,} first ICU stays ≥ 48 h in adult cancer patients (MIMIC-III),
          {metrics['n_events']:,} deaths within 30 days ({metrics['prevalence']:.1%})
        - **ROC-AUC:** {metrics['test_roc_auc']:.3f} (95% CI {metrics['test_roc_auc_ci_low']:.3f}–{metrics['test_roc_auc_ci_high']:.3f}),
          vs. {metrics['baseline_roc_auc']:.3f} for an age-only baseline
        - **PR-AUC:** {metrics['test_pr_auc']:.3f} · **Brier:** {metrics['test_brier']:.3f} · **Calibration slope:** {metrics['test_calibration_slope']:.2f}
        - Test set: {metrics['n_test']:,} stays, split by patient. Features chosen on the training split only.
        """)

# --- User Input UI ---
st.subheader("📋 Enter Patient Features")
input_data = {}
col1, col2 = st.columns(2)
display_features = [f for f in feature_names if f in feature_info]
missing = [f for f in feature_names if f not in feature_info]
if missing:
    # Every model input needs a widget; predicting with placeholder values would be meaningless.
    st.error(f"🚨 Model artifacts are out of sync: no input range for {missing}. "
             "Reload the page in a few minutes, or reboot the app.")
    st.stop()

for i, feature in enumerate(display_features):
    col = col1 if i < len(display_features) // 2 else col2
    label, min_val, max_val, default_val = feature_info[feature]
    input_data[feature] = col.number_input(label, min_value=float(min_val), max_value=float(max_val), value=float(default_val))

user_input_df = align_user_input(input_data, feature_template)


# --- Prediction ---
if st.button("🔍 Predict 30-Day Mortality"):
    input_scaled_df = pyfunc_predict(scaler, user_input_df)

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
    fig = shap_ax.figure
    st.pyplot(fig, use_container_width=True)   

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