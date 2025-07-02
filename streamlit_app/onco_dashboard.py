import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
st.set_page_config(page_title="🧬 OncoAI Risk Dashboard", layout="wide")
st.title("🧬 OncoAI 30-Day Mortality Predictor")

st.markdown("""
This dashboard allows you to enter a patient's features and predict the probability of 30-day mortality.
It also visualizes feature contributions via SHAP.
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
    try:
        df = pd.read_parquet(EXAMPLE_FEATURES_PATH)
        df = df.drop(columns=['icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob', 'dod', 'intime', 'outtime', 'icd9_code', 'mortality_30d'], errors='ignore')
        df = pd.get_dummies(df, drop_first=True)
        return df.head(1).copy()
    except FileNotFoundError:
        st.error(f"❌ Feature template file not found at {EXAMPLE_FEATURES_PATH}")
        st.stop()

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
        input_data[col] = st.number_input(f"{label}", min_value=min_val, max_value=max_val, value=default_val, key=f"input_{col}")
    else:
        # Handle new columns gracefully, perhaps with a generic number input or a warning
        # For now, let's assume all columns in feature_template are either in feature_info or are one-hot encoded
        # For one-hot encoded columns, we'll make them checkboxes
        if feature_template[col].dtype == 'uint8' or feature_template[col].dtype == 'bool': # Assuming one-hot encoded are uint8 or bool
            input_data[col] = st.checkbox(f"Feature: {col.replace('_', ' ').title()}", value=False, key=f"input_{col}")
        else:
            input_data[col] = st.number_input(f"{col}", value=0.0, key=f"input_{col}")


user_input_df = pd.DataFrame([input_data])

# Align with training columns
# Reindex user_input_df to match the order and presence of columns in feature_template
user_input_df = user_input_df.reindex(columns=feature_template.columns, fill_value=0)


# --- Predict and Explain ---
if st.button("🔍 Predict 30-Day Mortality"):
    input_scaled = scaler.transform(user_input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    st.success(f"🩸 **Predicted 30-Day Mortality Probability: {prob:.2%}**")

    with st.spinner("Explaining prediction..."):
        # Use a smaller, representative background dataset for SHAP to avoid memory issues
        # and ensure it's scaled correctly. Using `feature_template` directly ensures column alignment.
        if feature_template.shape[0] < 100:
            # If the template is too small, sample with replacement or just use the template itself
            background_df = feature_template.sample(min(100, feature_template.shape[0] * 5), replace=True, random_state=42)
        else:
            background_df = feature_template.sample(100, random_state=42)

        background_scaled = scaler.transform(background_df)
        explainer = shap.LinearExplainer(model, background_scaled)
        shap_values = explainer.shap_values(input_scaled)

        # Ensure shap_values is a 1D array if the model has a single output
        if isinstance(shap_values, list):
            shap_values = shap_values[0] # For binary classification, shap_values is a list of two arrays

        single_expl = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=user_input_df.iloc[0].values,
            feature_names=user_input_df.columns.tolist()
        )

        # Create the SHAP waterfall plot and display it using Streamlit's pyplot functionality
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(single_expl, show=False) # Don't show immediately, save to figure
        plt.title("SHAP Waterfall: Patient Risk Factors")
        plt.tight_layout()
        st.pyplot(fig) # Display the matplotlib figure in Streamlit
        plt.close(fig) # Close the figure to free up memory

st.markdown("---")
st.markdown("Developed by Sangeeth George")
