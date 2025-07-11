#tests/test_leakage.py


import os
import pytest
import pandas as pd
import numpy as np
import duckdb
from dotenv import load_dotenv

from oncoai_prototype.utils.leakage import check_for_leakage
from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_and_impute
)
from oncoai_prototype.data_processing.feature_engineering import load_data as load_actual_data


# --- Real data fixture ---
@pytest.fixture(scope="module")
def leakage_test_df():
    load_dotenv()

    try:
        duckdb.sql("INSTALL postgres_scanner;")
    except duckdb.CatalogException:
        pass
    duckdb.sql("LOAD postgres_scanner;")

    raw_data = load_actual_data()

    vitals = filter_high_coverage(raw_data['vitals'], label_col='vitals_label', min_coverage=0.95)
    labs = filter_high_coverage(raw_data['labs'], label_col='labs_label', min_coverage=0.70)

    vitals_feat = compute_time_series_features(
        vitals, time_col='charttime', value_col='vitals_valuenum',
        label_col='vitals_label', icu_id_col='icustay_id'
    )
    labs_feat = compute_time_series_features(
        labs, time_col='charttime', value_col='labs_valuenum',
        label_col='labs_label', icu_id_col='icustay_id'
    )

    merged = merge_features(raw_data['cohort'], vitals_feat, labs_feat)
    df_cleaned = filter_and_impute(merged, min_col_coverage=0.8)

    if "mortality_30d" not in df_cleaned.columns:
        pytest.skip("⚠️ Skipping: 'mortality_30d' not found in merged data.")

    return df_cleaned.sample(n=min(50, len(df_cleaned)), random_state=42)


# --- Synthetic unit tests ---

def test_detects_leakage_with_target_column_in_name(capfd):
    df = pd.DataFrame({
        "feature_1": [0.1, 0.2],
        "mortality_30d_prob": [0.9, 0.8],  # suspicious
        "mortality_30d": [0, 1]
    })
    df_checked = check_for_leakage(df, target_col="mortality_30d")
    out, _ = capfd.readouterr()

    assert "Potential data leakage" in out
    assert "mortality_30d_prob" not in df_checked.columns
    assert "mortality_30d" in df_checked.columns


def test_detects_no_leakage(capfd):
    df = pd.DataFrame({
        "feature_1": [1, 2],
        "feature_2": [3, 4],
        "mortality_30d": [0, 1]
    })
    df_checked = check_for_leakage(df, target_col="mortality_30d")
    out, _ = capfd.readouterr()

    assert "No significant leakage detected" in out
    assert df_checked.equals(df)


# --- Realistic test using actual PostgreSQL-derived data ---

def test_check_for_leakage_on_real_data(capfd, leakage_test_df):
    X = leakage_test_df.drop(columns=["mortality_30d"], errors="ignore").select_dtypes(include=[np.number])
    X_checked = check_for_leakage(X, target_col="mortality_30d")
    out, _ = capfd.readouterr()

    assert "No significant leakage detected" in out
    assert isinstance(X_checked, pd.DataFrame)
    assert set(X_checked.columns).issubset(set(X.columns))  # no unexpected columns added
