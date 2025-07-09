#tests/test_db_leakage.py

import pandas as pd
from oncoai_prototype.utils.leakage import check_for_leakage

def test_detects_leakage_with_target_column_in_name(capfd):
    df = pd.DataFrame({
        "feature_1": [0.1, 0.2],
        "mortality_30d_prob": [0.9, 0.8],  # suspicious
        "mortality_30d": [0, 1]
    })
    has_leakage = check_for_leakage(df, target_col="mortality_30d")
    out, _ = capfd.readouterr()

    assert has_leakage is True
    assert "Potential data leakage" in out

def test_detects_no_leakage(capfd):
    df = pd.DataFrame({
        "feature_1": [1, 2],
        "feature_2": [3, 4],
        "mortality_30d": [0, 1]
    })
    has_leakage = check_for_leakage(df, target_col="mortality_30d")
    out, _ = capfd.readouterr()

    assert has_leakage is False
    assert "No significant leakage detected" in out
