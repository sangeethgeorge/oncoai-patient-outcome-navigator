# tests/test_leakage.py

import pandas as pd
import numpy as np
import pytest

from oncoai_prototype.utils.leakage import check_for_leakage, assert_unique_ids, assert_no_group_overlap


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


def test_assert_unique_ids():
    assert_unique_ids(pd.DataFrame({"icustay_id": [1, 2, 3]}))
    with pytest.raises(ValueError, match="1 duplicate rows"):
        assert_unique_ids(pd.DataFrame({"icustay_id": [1, 2, 2]}))


def test_assert_no_group_overlap():
    assert_no_group_overlap([1, 2], [3, 4])
    with pytest.raises(ValueError, match="both train and test"):
        assert_no_group_overlap([1, 2], [2, 3])


# --- Live database check ---

@pytest.mark.db
def test_check_for_leakage_on_real_data(capfd, db_features_df):
    X = db_features_df.drop(columns=["mortality_30d"]).select_dtypes(include=[np.number])
    X_checked = check_for_leakage(X, target_col="mortality_30d")
    out, _ = capfd.readouterr()

    assert "No significant leakage detected" in out
    assert set(X_checked.columns) == set(X.columns)
