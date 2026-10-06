import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from oncoai_prototype.utils.db_utils import connect_to_postgres
from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_low_coverage_columns,
)
from oncoai_prototype.utils.leakage import assert_unique_ids

# --- Config ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

FEATURE_FILE = os.path.join(DATA_DIR, "onco_features_cleaned.parquet")
VITALS_FILE = os.path.join(DATA_DIR, "all_vitals_48h.parquet")
LABS_FILE = os.path.join(DATA_DIR, "all_labs_48h.parquet")

ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
LABEL_COL = 'mortality_30d'
# Covariates known at ICU admission, kept alongside the 48h lab/vital features. ICD-derived counts
# (e.g. n_cancer_codes) are excluded: MIMIC assigns ICD codes at discharge, after the prediction time.
COHORT_FEATURES = ['age']


def get_conn_str() -> str:
    load_dotenv()
    conn_str = os.getenv("ONCOAI_POSTGRES_CONN_STR")
    if conn_str is None:
        raise ValueError("ONCOAI_POSTGRES_CONN_STR environment variable not set.")
    return conn_str


# --- Load data from PostgreSQL via DuckDB ---
def load_data():
    conn_str = get_conn_str()
    con = connect_to_postgres(conn_str)
    tables = {'labs': 'all_labs_48h', 'vitals': 'all_vitals_48h', 'cohort': 'oncology_icu_base'}

    print("Fetching labs, vitals, and cohort data from PostgreSQL...")
    try:
        return {
            name: con.sql(f"SELECT * FROM postgres_scan('{conn_str}', 'public', '{table}')").df()
            for name, table in tables.items()
        }
    finally:
        con.close()


def build_features(data: dict) -> pd.DataFrame:
    """All candidate features, one row per ICU stay. No split, no target-aware selection:
    feature selection happens inside model training on the training split only."""
    cohort = data['cohort']
    assert_unique_ids(cohort, 'icustay_id')

    print("Filtering high-coverage vitals and labs...")
    vitals = filter_high_coverage(data['vitals'], label_col='vitals_label', min_coverage=0.95)
    labs = filter_high_coverage(data['labs'], label_col='labs_label', min_coverage=0.70)

    print("Creating time-series features...")
    vitals_features = compute_time_series_features(
        df=vitals, time_col='charttime', value_col='vitals_valuenum',
        label_col='vitals_label', icu_id_col='icustay_id'
    )
    labs_features = compute_time_series_features(
        df=labs, time_col='charttime', value_col='labs_valuenum',
        label_col='labs_label', icu_id_col='icustay_id'
    )
    # A label present in both sources (e.g. vitals' 'BUN' vs labs' 'Urea Nitrogen') keeps both;
    # an identical column name would collide, so prefix the chart-sourced copy.
    shared = set(vitals_features.columns) & set(labs_features.columns) - {'icustay_id'}
    vitals_features = vitals_features.rename(columns={c: f"chart_{c}" for c in shared})

    print("Merging cohort with vitals and labs...")
    base = cohort[ID_COLS + COHORT_FEATURES + [LABEL_COL]]
    full_df = merge_features(base, vitals_features, labs_features)

    feature_cols = [c for c in full_df.columns if c not in ID_COLS + [LABEL_COL]]
    full_df = filter_low_coverage_columns(full_df, feature_cols, min_col_coverage=0.8)
    assert_unique_ids(full_df, 'icustay_id')
    return full_df


# --- Run Pipeline ---
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Starting OncoAI feature pipeline...")
    data = load_data()
    data["vitals"].to_parquet(VITALS_FILE, index=False)
    data["labs"].to_parquet(LABS_FILE, index=False)

    final_df = build_features(data)
    final_df.to_parquet(FEATURE_FILE, index=False)
    n_events = int(final_df[LABEL_COL].sum())
    print(f"✅ Saved features to {FEATURE_FILE} — shape: {final_df.shape}")
    print(f"   Cohort: {len(final_df)} ICU stays, {n_events} deaths ({n_events / len(final_df):.1%})")
