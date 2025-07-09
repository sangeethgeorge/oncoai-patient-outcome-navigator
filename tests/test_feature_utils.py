#tests/test_feature_utils.py

import pandas as pd
import numpy as np
from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_and_impute
)

def test_filter_high_coverage():
    df = pd.DataFrame({
        "col_1": [1, 2, 3, 4, 5],
        "col_2": [1, None, None, None, None],  # 20% coverage
        "col_3": [1, 2, 3, 4, None]            # 80% coverage
    })
    filtered = filter_high_coverage(df, min_coverage=0.8)
    assert "col_1" in filtered.columns
    assert "col_3" in filtered.columns
    assert "col_2" not in filtered.columns

def test_compute_time_series_features():
    df = pd.DataFrame({
        "icustay_id": [101, 101, 102, 102, 102],
        "hr": [80, 85, 90, 88, 91],
        "bp": [120, 122, 130, 128, 125]
    })
    features = compute_time_series_features(df, groupby_col="icustay_id")
    expected_cols = {
        "hr_mean", "hr_min", "hr_max", "hr_std",
        "bp_mean", "bp_min", "bp_max", "bp_std"
    }
    assert set(expected_cols).issubset(set(features.columns))
    assert "icustay_id" in features.columns
    assert features.shape[0] == 2  # two groups

def test_merge_features():
    cohort = pd.DataFrame({
        "icustay_id": [101, 102],
        "subject_id": [1, 2]
    })
    vitals = pd.DataFrame({
        "icustay_id": [101],
        "hr_mean": [82]
    })
    labs = pd.DataFrame({
        "icustay_id": [102],
        "wbc_mean": [6.5]
    })
    merged = merge_features(cohort, vitals, labs)

    assert "hr_mean" in merged.columns
    assert "wbc_mean" in merged.columns
    assert merged.shape[0] == 2
    assert pd.isna(merged.loc[merged.icustay_id == 102, "hr_mean"]).all()
    assert pd.isna(merged.loc[merged.icustay_id == 101, "wbc_mean"]).all()

def test_filter_and_impute():
    df = pd.DataFrame({
        "a": [1, 2, 3, None, 5],
        "b": [None, None, None, None, None],  # 0% coverage
        "c": [1, 2, 3, 4, 5],  # 100% coverage
    })
    cleaned = filter_and_impute(df, min_col_coverage=0.6)

    assert "b" not in cleaned.columns  # dropped
    assert cleaned.shape[0] == 4       # row with None in 'a' dropped
    assert cleaned.isnull().sum().sum() == 0
