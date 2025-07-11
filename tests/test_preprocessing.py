#tests/test_preprocessing.py

import os
import pytest
import pandas as pd
import numpy as np
import duckdb
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.preprocessing import (
    train_test_impute_split,
    scale_features,
    preprocess_for_inference
)
from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_and_impute,
)
from oncoai_prototype.data_processing.feature_engineering import load_data as load_actual_data


# --- Fixture: Load and build clean ML-ready data from source tables ---
@pytest.fixture(scope="module")
def ml_ready_df():
    """
    Load data from Postgres, compute features, and return a clean ML-ready dataframe.
    """
    load_dotenv()

    try:
        duckdb.sql("INSTALL postgres_scanner;")
    except duckdb.CatalogException:
        pass
    duckdb.sql("LOAD postgres_scanner;")

    print("\nLoading raw data from PostgreSQL...")
    raw_data = load_actual_data()

    print("Running preprocessing: filtering + feature engineering...")
    vitals = filter_high_coverage(raw_data['vitals'], label_col='vitals_label', min_coverage=0.95)
    labs = filter_high_coverage(raw_data['labs'], label_col='labs_label', min_coverage=0.70)

    vitals_feat = compute_time_series_features(vitals, time_col='charttime', value_col='vitals_valuenum',
                                                label_col='vitals_label', icu_id_col='icustay_id')
    labs_feat = compute_time_series_features(labs, time_col='charttime', value_col='labs_valuenum',
                                              label_col='labs_label', icu_id_col='icustay_id')

    merged = merge_features(raw_data['cohort'], vitals_feat, labs_feat)
    df_cleaned = filter_and_impute(merged, min_col_coverage=0.8)

    if 'mortality_30d' not in df_cleaned.columns:
        pytest.skip("⚠️ Skipping: 'mortality_30d' not found in cleaned data.")

    return df_cleaned.sample(n=min(100, len(df_cleaned)), random_state=42)


# --- Tests using live PostgreSQL-backed ML data ---

def test_train_test_impute_split(ml_ready_df):
    X_train, X_test, y_train, y_test = train_test_impute_split(ml_ready_df, target_col="mortality_30d")

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
    assert set(y_train.unique()).issubset({0, 1})


def test_scale_features(ml_ready_df):
    X_train, X_test, y_train, y_test = train_test_impute_split(ml_ready_df, target_col="mortality_30d")

    # Filter numeric-only columns for scaling
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test[X_train.columns]  # ensure alignment

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert isinstance(scaler, StandardScaler)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-1)


def test_preprocess_for_inference(ml_ready_df):
    df = ml_ready_df.drop(columns=["mortality_30d"], errors="ignore")
    df_processed = preprocess_for_inference(df)

    # Drop-list columns should not remain
    excluded_cols = ['icustay_id', 'subject_id', 'hadm_id', 'admittime',
                     'dob', 'dod', 'intime', 'outtime', 'icd9_code']
    for col in excluded_cols:
        assert col not in df_processed.columns

    assert not df_processed.isnull().values.any()


