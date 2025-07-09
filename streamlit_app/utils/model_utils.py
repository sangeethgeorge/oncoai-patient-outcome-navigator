# utils/input_utils.py

import streamlit as st
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "onco_features_cleaned.parquet")

@st.cache_data
def get_feature_template():
    df = pd.read_parquet(FEATURES_PATH)
    drop_cols = ['icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob', 'dod', 'intime', 'outtime', 'icd9_code', 'mortality_30d']
    df = df.drop(columns=drop_cols, errors="ignore")
    return pd.get_dummies(df, drop_first=True).head(1).copy()

@st.cache_data
def get_feature_info():
    return {
        'age': ("Age (years)", 16.0, 100.0, 65.0),
        'min_heart_rate': ("Min Heart Rate", 30.0, 120.0, 70.0),
        # Add more features as needed
    }

def align_user_input(input_data, template_df):
    df = pd.DataFrame([input_data])
    for col in set(template_df.columns) - set(df.columns):
        df[col] = 0
    return df[template_df.columns]

