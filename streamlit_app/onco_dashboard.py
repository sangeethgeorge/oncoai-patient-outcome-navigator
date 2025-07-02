import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# --- App Configuration ---
st.set_page_config(page_title="OncoAI Risk Dashboard", layout="wide")
st.title("🧬 OncoAI 30-Day Mortality Predictor")

st.markdown("""
Welcome to **OncoAI**, a research prototype built on the MIMIC-III dataset.

This tool predicts the **30-day mortality risk** for ICU patients with cancer based on early vitals and lab values. It also provides an explanation of how each feature contributes to the prediction using SHAP.

⚠️ *Note: This app is for demonstration and research purposes only, not for clinical use.*
""")
# --- Paths ---
PROJECT_ROOT = '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator'
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/logreg_model_run_ae10757d5eb94eb681115211fb918898.joblib")  
EXAMPLE_FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "onco_features_cleaned.parquet")

# --- Load model and scaler ---
@st.cache_resource
def load_artifacts():
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact['model'], artifact['scaler']
    except FileNotFoundError:
        st.error(f"❌ Model file not found at {MODEL_PATH}")
        st.stop()

model, scaler = load_artifacts()

# --- Load feature template ---
@st.cache_data
def get_feature_template():
    df = pd.read_parquet(EXAMPLE_FEATURES_PATH)
    df = df.drop(columns=['icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob', 'dod', 'intime', 'outtime', 'icd9_code', 'mortality_30d'], errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    return df.head(1).copy()

feature_template = get_feature_template()
input_data = {}

# --- Feature Metadata with Alias and Range ---
feature_info = {
    'age': (
        "Age (years) [Range: 16–100]", 
        16.0, 100.0, 65.0
    ),
    'min_heart_rate': (
        "Min Heart Rate (bpm) [Normal: 60–100]", 
        30.0, 120.0, 70.0
    ),
    'min_urea_nitrogen': (
        "Min BUN (mg/dL) [Normal: 7–20]", 
        3.0, 30.0, 8.0
    ),
    'mean_urea_nitrogen': (
        "Mean BUN (mg/dL) [Normal: 7–20]", 
        5.0, 80.0, 25.0
    ),
    'min_white_blood_cells': (
        "Min WBC (K/µL) [Normal: 4.5–11.0]", 
        0.5, 20.0, 5.0
    ),
    'mean_chloride': (
        "Mean Chloride (mEq/L) [Normal: 96–106]", 
        85.0, 115.0, 100.0
    ),
    'mean_glucose': (
        "Mean Glucose (mg/dL) [Normal: 70–99 (fasting)]", 
        50.0, 300.0, 120.0
    ),
    'max_bicarbonate': (
        "Max Bicarbonate (mEq/L) [Normal: 22–28]", 
        10.0, 45.0, 25.0
    ),
    'mean_mchc': (
        "Mean MCHC (g/dL) [Normal: 32–36]", 
        28.0, 38.0, 34.0
    ),
    'max_mchc': (
        "Max MCHC (g/dL) [Normal: 32–36]", 
        28.0, 40.0, 36.0
    )
}

# --- User Input Panel ---
st.subheader("📝 Enter Patient Features")

for col in feature_template.columns:
    if col in feature_info:
        label, min_val, max_val, default_val = feature_info[col]
        input_data[col] = st.number_input(f"{label}", min_value=min_val, max_value=max_val, value=default_val)
    else:
        input_data[col] = st.number_input(f"{col}", value=0.0)

user_input_df = pd.DataFrame([input_data])

# Align with training columns
missing_cols = set(feature_template.columns) - set(user_input_df.columns)
for col in missing_cols:
    user_input_df[col] = 0
user_input_df = user_input_df[feature_template.columns]

# --- Predict and Explain ---
if st.button("🔍 Predict 30-Day Mortality"):
    input_scaled = scaler.transform(user_input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    st.success(f"🩸 **Predicted 30-Day Mortality Probability: {prob:.2%}**")

    with st.spinner("Explaining prediction..."):
        background = scaler.transform(feature_template.sample(100, replace=True))
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(input_scaled)

        single_expl = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=user_input_df.iloc[0].values,
            feature_names=user_input_df.columns.tolist()
        )

        shap_fig_path = os.path.join(PROJECT_ROOT, "shap_temp_waterfall.png")
        shap.plots.waterfall(single_expl, show=False)
        plt.title("SHAP Waterfall: Patient Risk Factors")
        plt.savefig(shap_fig_path, bbox_inches='tight')
        plt.close()

        st.image(shap_fig_path, caption="SHAP Explanation", use_container_width=True)

        # --- SHAP Interpretation Help ---
        with st.expander("📘 How to interpret the SHAP Waterfall Plot"):
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

        # --- Interactive SHAP Value Table ---
        st.markdown("### 🔍 Feature-Level Breakdown")

        shap_contributions_df = pd.DataFrame({
            "Feature": user_input_df.columns,
            "Input Value": user_input_df.iloc[0].values,
            "SHAP Value": single_expl.values,
            "Impact": ["⬆️ Increases Risk" if val > 0 else "⬇️ Decreases Risk" for val in single_expl.values]
        })

        # Sort by absolute SHAP value (biggest contributors first)
        shap_contributions_df["|SHAP|"] = np.abs(shap_contributions_df["SHAP Value"])
        shap_contributions_df = shap_contributions_df.sort_values(by="|SHAP|", ascending=False).drop(columns="|SHAP|")

        st.dataframe(shap_contributions_df, use_container_width=True, height=500)


st.markdown("---")
st.markdown("🧑‍💻 Developed by **Sangeeth George** — OncoAI (MIMIC-III)")
