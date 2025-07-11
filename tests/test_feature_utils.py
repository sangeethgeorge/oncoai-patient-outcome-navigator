#tests/test_feature_utils.py

import pandas as pd
import numpy as np
import pytest
import os
import duckdb
from dotenv import load_dotenv

# Import the actual utility functions
from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_and_impute
)

# Import the data loading function from feature_engineering
from oncoai_prototype.data_processing.feature_engineering import load_data as load_actual_data

# --- Fixture for loading actual data ---
@pytest.fixture(scope="module")
def actual_raw_data():
    """
    Fixture to load actual raw data from the PostgreSQL database
    once per module (test file).
    """
    # Ensure environment variables are loaded
    load_dotenv()
    # Ensure postgres_scanner is loaded if not already
    try:
        duckdb.sql("INSTALL postgres_scanner;")
    except duckdb.CatalogException:
        pass # Already installed
    duckdb.sql("LOAD postgres_scanner;")

    # Load data using the function from feature_engineering.py
    print("\nLoading actual data from PostgreSQL for tests...")
    data = load_actual_data()
    print("Actual data loaded.")
    return data

@pytest.fixture(scope="module")
def processed_time_series_features(actual_raw_data):
    """
    Fixture to compute time series features from actual data.
    """
    vitals = filter_high_coverage(actual_raw_data['vitals'], label_col='vitals_label', min_coverage=0.95)
    labs = filter_high_coverage(actual_raw_data['labs'], label_col='labs_label', min_coverage=0.70)

    vitals_features = compute_time_series_features(
        df=vitals,
        time_col='charttime',
        value_col='vitals_valuenum',
        label_col='vitals_label',
        icu_id_col='icustay_id'
    )
    labs_features = compute_time_series_features(
        df=labs,
        time_col='charttime',
        value_col='labs_valuenum',
        label_col='labs_label',
        icu_id_col='icustay_id'
    )
    return vitals_features, labs_features

@pytest.fixture(scope="module")
def actual_merged_df(actual_raw_data, processed_time_series_features):
    """
    Fixture to merge cohort with processed vitals and labs features.
    """
    vitals_features, labs_features = processed_time_series_features
    full_df = merge_features(actual_raw_data['cohort'], vitals_features, labs_features)
    return full_df

# --- Tests using actual data fixtures ---

def test_filter_high_coverage_actual(actual_raw_data):
    """
    Test filter_high_coverage using actual 'vitals' data.
    This test now checks if the filtering works as expected on real data patterns.
    """
    vitals_df = actual_raw_data['vitals'].copy()

    # Example: Filter for a high coverage threshold (e.g., 95%)
    # This will depend on your actual data's coverage.
    # You might need to adjust min_coverage based on your data distribution.
    min_coverage_threshold = 0.95 
    filtered_vitals = filter_high_coverage(
        df=vitals_df,
        label_col='vitals_label',
        group_col='icustay_id',
        min_coverage=min_coverage_threshold
    )

    # Assertions:
    # 1. Check that icustay_id and vitals_label are still present
    assert 'icustay_id' in filtered_vitals.columns
    assert 'vitals_label' in filtered_vitals.columns
    # 2. Check if the resulting dataframe is not empty (assuming there's high coverage data)
    assert not filtered_vitals.empty
    # 3. (More advanced) You could manually calculate expected coverage for a known label
    # and assert if it passes or fails the threshold.
    # For instance:
    # total_stays = vitals_df['icustay_id'].nunique()
    # hr_coverage = vitals_df[vitals_df['vitals_label'] == 'Heart Rate']['icustay_id'].nunique() / total_stays
    # if hr_coverage < min_coverage_threshold:
    #     assert 'Heart Rate' not in filtered_vitals['vitals_label'].unique()
    # else:
    #     assert 'Heart Rate' in filtered_vitals['vitals_label'].unique()


def test_compute_time_series_features_actual(processed_time_series_features):
    """
    Test compute_time_series_features using actual processed vitals/labs features.
    """
    vitals_features, labs_features = processed_time_series_features

    # Assertions for vitals features
    assert not vitals_features.empty
    assert 'icustay_id' in vitals_features.columns
    # Check if expected feature columns (mean, min, max, slope) are present for some common vital
    # This assumes 'HeartRate' or similar is a common label in your actual data
    expected_vitals_stats_cols = [col for col in vitals_features.columns if any(stat in col for stat in ['mean_', 'min_', 'max_', 'slope_'])]
    assert len(expected_vitals_stats_cols) > 0, "No time series feature columns found in vitals features."
    assert vitals_features.isnull().sum().sum() == 0 or vitals_features.isnull().sum().sum() < (vitals_features.shape[0] * vitals_features.shape[1] * 0.5), \
           "Too many NaNs in vitals features, indicating potential issue."

    # Assertions for labs features
    assert not labs_features.empty
    assert 'icustay_id' in labs_features.columns
    expected_labs_stats_cols = [col for col in labs_features.columns if any(stat in col for stat in ['mean_', 'min_', 'max_', 'slope_'])]
    assert len(expected_labs_stats_cols) > 0, "No time series feature columns found in labs features."
    assert labs_features.isnull().sum().sum() == 0 or labs_features.isnull().sum().sum() < (labs_features.shape[0] * labs_features.shape[1] * 0.5), \
           "Too many NaNs in labs features, indicating potential issue."

def test_merge_features_actual(actual_raw_data, processed_time_series_features):
    """
    Test merge_features using actual data.
    """
    vitals_features, labs_features = processed_time_series_features
    cohort_df = actual_raw_data['cohort'].copy()

    merged_df = merge_features(cohort_df, vitals_features, labs_features)

    # Assertions
    assert not merged_df.empty
    assert 'icustay_id' in merged_df.columns
    assert 'subject_id' in merged_df.columns # From cohort
    # Check if some features from vitals_features and labs_features are present
    assert any(col in merged_df.columns for col in vitals_features.columns if col != 'icustay_id')
    assert any(col in merged_df.columns for col in labs_features.columns if col != 'icustay_id')
    # Check the number of rows should be consistent with the cohort (left merge)
    assert merged_df.shape[0] == cohort_df.shape[0]


def test_filter_and_impute_actual(actual_merged_df):
    """
    Test filter_and_impute using an actual merged dataframe.
    """
    df_to_clean = actual_merged_df.copy()

    # It's important to understand what min_col_coverage means for your actual data.
    # A very high value might drop almost everything if your raw data is sparse.
    cleaned_df = filter_and_impute(df_to_clean, min_col_coverage=0.8)

    # Assertions
    assert not cleaned_df.empty, "Cleaned dataframe is empty, min_col_coverage might be too high for actual data."
    assert cleaned_df.isnull().sum().sum() == 0, "NaNs still present after filtering and imputation."
    # Check that the number of columns is less than or equal to the original
    assert cleaned_df.shape[1] <= df_to_clean.shape[1]
    # Check that the number of rows is less than or equal to the original
    assert cleaned_df.shape[0] <= df_to_clean.shape[0]
