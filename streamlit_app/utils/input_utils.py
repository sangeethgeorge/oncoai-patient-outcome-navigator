# utils/model_utils.py

import streamlit as st
import joblib
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "logreg_model.joblib")

@st.cache_resource
def load_artifacts():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["scaler"]
