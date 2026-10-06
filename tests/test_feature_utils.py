#tests/test_feature_utils.py

import pandas as pd
import numpy as np
import pytest

from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_low_coverage_columns,
)


@pytest.fixture
def long_measurements():
    t0 = pd.Timestamp("2150-01-01 08:00")
    rows = []
    # Stay 1: heart rate rises 2 bpm/hour; glucose single reading
    for h in range(0, 6):
        rows.append((1, "Heart Rate", t0 + pd.Timedelta(hours=h), 80 + 2 * h))
    rows.append((1, "Glucose", t0, 120.0))
    rows.append((1, "INR(PT)", t0, 1.1))
    # Stay 2: flat heart rate, only heart rate measured
    for h in range(0, 3):
        rows.append((2, "Heart Rate", t0 + pd.Timedelta(hours=h), 70.0))
    return pd.DataFrame(rows, columns=["icustay_id", "label", "charttime", "value"])


def test_time_series_features_slope_is_per_hour(long_measurements):
    wide = compute_time_series_features(long_measurements, time_col="charttime", value_col="value",
                                        label_col="label", icu_id_col="icustay_id").set_index("icustay_id")

    assert wide.loc[1, "slope_heart_rate"] == pytest.approx(2.0)
    assert wide.loc[2, "slope_heart_rate"] == pytest.approx(0.0)
    assert wide.loc[1, "mean_heart_rate"] == pytest.approx(85.0)
    assert wide.loc[1, "min_heart_rate"] == 80 and wide.loc[1, "max_heart_rate"] == 90
    # One reading: no slope; label absent for a stay: NaN, not an error
    assert np.isnan(wide.loc[1, "slope_glucose"])
    assert np.isnan(wide.loc[2, "mean_glucose"])


def test_time_series_features_one_row_per_stay(long_measurements):
    wide = compute_time_series_features(long_measurements, "charttime", "value", "label", "icustay_id")
    assert wide["icustay_id"].is_unique
    assert len(wide) == 2
    assert "mean_inr_pt" in wide.columns  # punctuation normalized to snake_case


def test_filter_high_coverage_keeps_common_labels(long_measurements):
    filtered = filter_high_coverage(long_measurements, label_col="label", min_coverage=0.95)
    assert set(filtered["label"]) == {"Heart Rate"}  # glucose only in 1 of 2 stays


def test_merge_features_rejects_duplicate_stays():
    cohort = pd.DataFrame({"icustay_id": [1, 1, 2], "mortality_30d": [0, 0, 1]})
    feats = pd.DataFrame({"icustay_id": [1, 2], "mean_x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate icustay_id"):
        merge_features(cohort, feats, feats.rename(columns={"mean_x": "mean_y"}))


def test_merge_features_left_joins_one_row_per_stay():
    cohort = pd.DataFrame({"icustay_id": [1, 2, 3], "mortality_30d": [0, 1, 0]})
    vitals = pd.DataFrame({"icustay_id": [1, 2], "mean_hr": [80.0, 90.0]})
    labs = pd.DataFrame({"icustay_id": [2, 3], "mean_na": [140.0, 135.0]})
    merged = merge_features(cohort, vitals, labs)
    assert len(merged) == 3
    assert merged["icustay_id"].is_unique


def test_filter_low_coverage_columns_keeps_rows():
    df = pd.DataFrame({
        "icustay_id": range(10),
        "good": [1.0] * 9 + [np.nan],
        "sparse": [1.0] * 5 + [np.nan] * 5,
    })
    out = filter_low_coverage_columns(df, ["good", "sparse"], min_col_coverage=0.8)
    assert "sparse" not in out.columns
    assert "good" in out.columns
    assert len(out) == 10  # rows with missing values are kept for imputation


# --- Live database checks ---

@pytest.mark.db
def test_db_features_one_row_per_stay(db_features_df):
    assert db_features_df["icustay_id"].is_unique
    assert set(db_features_df["mortality_30d"].unique()) <= {0, 1}
