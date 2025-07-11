import os
import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from dotenv import load_dotenv

from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_and_impute,
)
from oncoai_prototype.utils.leakage import check_for_leakage

# --- Config ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FILE = os.path.join(DATA_DIR, "onco_features_cleaned.parquet")
os.makedirs(DATA_DIR, exist_ok=True)

# Get PostgreSQL connection string from environment variable
load_dotenv()
POSTGRES_CONN_STR = os.getenv("ONCOAI_POSTGRES_CONN_STR")

if POSTGRES_CONN_STR is None:
    raise ValueError("ONCOAI_POSTGRES_CONN_STR environment variable not set.")


# --- Load data from Postgres via postgres_scan ---
def load_data():
    try:
        duckdb.sql("INSTALL postgres_scanner;")
    except duckdb.CatalogException:
        pass  # Already installed

    duckdb.sql("LOAD postgres_scanner;")

    queries = {
        'labs': f"SELECT * FROM postgres_scan('{POSTGRES_CONN_STR}', 'public', 'all_labs_48h')",
        'vitals': f"SELECT * FROM postgres_scan('{POSTGRES_CONN_STR}', 'public', 'all_vitals_48h')",
        'cohort': f"SELECT * FROM postgres_scan('{POSTGRES_CONN_STR}', 'public', 'oncology_icu_base')"
    }

    print("Fetching labs, vitals, and cohort data from PostgreSQL...")
    return {name: duckdb.sql(query).df() for name, query in queries.items()}

# --- Full pipeline with SHAP feature selection ---
def build_onco_shap_features(top_n=10):
    print("Loading source data...")
    data = load_data()

    print("Filtering high-coverage vitals and labs...")
    vitals = filter_high_coverage(data['vitals'], label_col='vitals_label', min_coverage=0.95)
    labs = filter_high_coverage(data['labs'], label_col='labs_label', min_coverage=0.70)  
   
    print("Creating time-series features...")
    vitals_features = compute_time_series_features( df=vitals,
    time_col='charttime',
    value_col='vitals_valuenum',
    label_col='vitals_label',
    icu_id_col='icustay_id')
    labs_features = compute_time_series_features(df=labs,
    time_col='charttime',
    value_col='labs_valuenum',
    label_col='labs_label',
    icu_id_col='icustay_id')

    print("Merging cohort with vitals and labs...")
    full_df = merge_features(data['cohort'], vitals_features, labs_features)

    print("Filtering + imputing incomplete rows/columns...")
    df_clean = filter_and_impute(full_df)

    print("Preparing data for SHAP feature selection...")
    id_cols = ['subject_id', 'hadm_id', 'icustay_id']
    label_col = 'mortality_30d'
    X = df_clean.drop(columns=[label_col] + id_cols, errors='ignore').select_dtypes(include=[np.number])
    y = df_clean[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Checking for data leakage...")
    X_train = check_for_leakage(X_train, target_col="mortality_30d")  
    X_test = X_test[X_train.columns]

    print("Fitting XGBoost for SHAP feature importance...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
	eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    print("Computing SHAP values...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    mean_shap = pd.DataFrame(shap_values.values, columns=X_test.columns).abs().mean().sort_values(ascending=False)
    top_features = mean_shap.head(top_n).index.tolist()

    print(f"Top {top_n} SHAP features: {top_features}")

    all_cols = id_cols + top_features + [label_col]
    final_df = df_clean[all_cols].copy()

    return final_df

# --- Run Pipeline ---
if __name__ == "__main__":
    print("Starting OncoAI full feature + SHAP pipeline...")
    final_df = build_onco_shap_features(top_n=10)
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved final feature subset to {OUTPUT_FILE} — shape: {final_df.shape}")
