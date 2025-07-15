import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
from io import BytesIO
import requests

# --------------------------
# GitHub Artifact Paths
# --------------------------
REPO_BASE = "https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main"

MODEL_URL = f"{REPO_BASE}/models/model.pkl"
SCALER_URL = f"{REPO_BASE}/models/scaler.pkl"
FEATURES_URL = f"{REPO_BASE}/models/feature_names.txt"

COHORT_URL = f"{REPO_BASE}/data/processed/onco_features_cleaned.parquet"
VITALS_URL = f"{REPO_BASE}/data/processed/all_vitals_48h.parquet"
LABS_URL = f"{REPO_BASE}/data/processed/all_labs_48h.parquet"

# --------------------------
# Data + Model Loaders
# --------------------------
@st.cache_data
def fetch_parquet(url):
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_parquet(BytesIO(r.content))

@st.cache_data
def fetch_txt(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.text.strip().splitlines()

@st.cache_resource
def fetch_model(url):
    r = requests.get(url)
    r.raise_for_status()
    return joblib.load(BytesIO(r.content))

# --------------------------
# Load All Artifacts
# --------------------------
features = fetch_parquet(COHORT_URL)
vitals_df = fetch_parquet(VITALS_URL)
labs_df = fetch_parquet(LABS_URL)
model = fetch_model(MODEL_URL)
scaler = fetch_model(SCALER_URL)
feature_names = fetch_txt(FEATURES_URL)

feature_info = {
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

def predict_and_explain(X_scaled, X_df):
    pred = model.predict_proba(X_scaled)[0][1]
    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_values = explainer(X_scaled)
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': X_df.iloc[0].values,
        'SHAP Value': shap_values.values[0]
    })
    shap_df['Contribution'] = shap_df['SHAP Value'].apply(lambda x: '↑ Risk' if x > 0 else '↓ Risk')
    shap_df['Abs SHAP'] = np.abs(shap_df['SHAP Value'])
    return pred, shap_values, shap_df.sort_values(by='Abs SHAP', ascending=False)

# --------------------------
# App UI
# --------------------------
st.set_page_config(page_title="OncoAI Dashboard", layout="wide")
st.title("🧠 OncoAI Patient Outcome Navigator")

tabs = st.tabs([
    "📘 Overview",
    "🧬 Cohort Explorer",
    "🔎 Patient Risk Viewer",
    "✍️ Manual Risk Calculator",
    "📊 SHAP Explainability"
])

# --------------------------
# 1. Overview
# --------------------------
with tabs[0]:
    st.markdown("""
    ### AI-powered clinical dashboard for 30-day mortality prediction in oncology ICU patients

    OncoAI helps explore how early ICU data (labs + vitals) can inform prognosis in critically ill cancer patients.

    **Pipeline:**
    - Define cohort (ICD neoplasm codes)
    - Extract 48h vitals/labs
    - Train ML model for 30-day mortality
    - Interpret predictions using SHAP
    - Use GPT-4/NLP to summarize notes

    ⚠️ For academic/demo use only — MIMIC-III demo data only.
    """)

    st.image("https://raw.githubusercontent.com/sangeethgeorge/oncoai-patient-outcome-navigator/main/docs/oncopath.png", caption="OncoAI Workflow", use_column_width=True)

# --------------------------
# 2. Cohort Explorer
# --------------------------
with tabs[1]:
    st.subheader("🧬 Oncology ICU Cohort (n = {})".format(len(features)))
    col1, col2 = st.columns(2)

    gender = col1.selectbox("Filter by Gender", ["All", "M", "F"])
    min_age, max_age = col2.slider("Filter by Age", 18, 89, (18, 89))

    filtered = features.copy()
    if gender != "All":
        filtered = filtered[filtered["gender"] == gender]
    filtered = filtered[(filtered["age"] >= min_age) & (filtered["age"] <= max_age)]

    st.dataframe(filtered[["subject_id", "age", "gender", "icustay_id", "mortality_30d"]], use_container_width=True)


# --------------------------
# 3. Patient Risk Viewer
# --------------------------
with tabs[2]:
    st.subheader("🔎 Patient Risk Viewer")
    sid = st.selectbox("Select Patient", features["subject_id"].unique())
    pdata = features[features["subject_id"] == sid]
    X = pdata[feature_names]
    X_scaled = scaler.transform(X)
    prob, shap_vals, shap_df = predict_and_explain(X_scaled, X)

    st.metric("30-Day Mortality Risk", f"{prob:.2%}")

    # Trend plots
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Labs (CRP, Hgb)**")
        crp = labs_df[(labs_df.subject_id == sid) & (labs_df.labs_label.str.upper() == "C-REACTIVE PROTEIN")]
        hgb = labs_df[(labs_df.subject_id == sid) & (labs_df.labs_label.str.upper() == "HEMOGLOBIN")]
        if not crp.empty:
            st.line_chart(crp.set_index("charttime")["labs_valuenum"])
        if not hgb.empty:
            st.line_chart(hgb.set_index("charttime")["labs_valuenum"])
    with col2:
        st.markdown("**Vitals (MAP, HR)**")
        map_ = vitals_df[(vitals_df.subject_id == sid) & (vitals_df.vitals_label.str.contains("Mean BP", case=False))]
        hr = vitals_df[(vitals_df.subject_id == sid) & (vitals_df.vitals_label.str.contains("Heart Rate", case=False))]
        if not map_.empty:
            st.line_chart(map_.set_index("charttime")["vitals_valuenum"])
        if not hr.empty:
            st.line_chart(hr.set_index("charttime")["vitals_valuenum"])

    st.markdown("### SHAP Waterfall Plot")
    shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(bbox_inches="tight")

    st.markdown("### SHAP Beeswarm Plot")
    shap.plots.beeswarm(shap_vals, show=False)
    st.pyplot(bbox_inches="tight")

# --------------------------
# 4. Manual Risk Calculator
# --------------------------
with tabs[3]:
    st.subheader("✍️ Manual Risk Calculator")
    input_data = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_info):
        label, minv, maxv, default = feature_info[feature]
        col = col1 if i < 5 else col2
        input_data[feature] = col.number_input(label, min_value=minv, max_value=maxv, value=default)
    user_df = pd.DataFrame([input_data])[feature_names]
    user_scaled = scaler.transform(user_df)
    prob, shap_vals, shap_df = predict_and_explain(user_scaled, user_df)
    st.metric("Predicted Mortality Risk", f"{prob:.2%}")
    st.markdown("### SHAP Waterfall")
    shap.plots.waterfall(shap_vals[0], show=False)
    st.pyplot(bbox_inches="tight")

    st.markdown("### Feature Contribution Table")
    st.dataframe(shap_df[["Feature", "Value", "SHAP Value", "Contribution"]], use_container_width=True)

# --------------------------
# 5. SHAP Explainability
# --------------------------
with tabs[4]:
    st.subheader("📊 Global SHAP Summary (Entire Cohort)")
    cohort_X = features[feature_names]
    cohort_scaled = scaler.transform(cohort_X)
    explainer = shap.Explainer(model, feature_names=feature_names)
    shap_vals = explainer(cohort_scaled)
    shap_df_all = pd.DataFrame(shap_vals.values, columns=feature_names)
    shap_sum = shap_df_all.abs().mean().sort_values(ascending=False).reset_index()
    shap_sum.columns = ["Feature", "Mean |SHAP|"]

    top_n = st.slider("Top N Features", 5, 20, 10)
    st.bar_chart(shap_sum.set_index("Feature").head(top_n))

    if st.checkbox("Show Raw SHAP Values Table"):
        st.dataframe(shap_df_all.head(100))
