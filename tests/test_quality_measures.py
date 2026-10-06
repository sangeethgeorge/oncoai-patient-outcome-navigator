# tests/test_quality_measures.py

import numpy as np
import pandas as pd
import pytest

from oncoai_prototype.analytics import quality_measures as qm


def test_risk_tiers_follow_cutpoints():
    tiers = qm.assign_risk_tiers([0.05, 0.2, 0.29999, 0.3, 0.6], cuts=[0.2, 0.3])
    assert list(tiers) == ["low", "medium", "medium", "high", "high"]


def test_tertile_cutpoints_split_evenly():
    p = np.random.default_rng(0).random(3000)
    tiers = qm.assign_risk_tiers(p, qm.tier_cutpoints(p))
    assert tiers.value_counts().between(950, 1050).all()


def test_byar_ci_brackets_ratio():
    lo, hi = qm.byar_ci([20, 0], [20.0, 5.0])
    assert lo[0] < 1 < hi[0]
    assert lo[1] == 0 and hi[1] > 0
    # Known value: O=20, E=20 gives roughly 0.61-1.54
    assert lo[0] == pytest.approx(0.611, abs=0.01) and hi[0] == pytest.approx(1.544, abs=0.01)


def test_calibration_by_group_flags_underprediction():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"unit": np.repeat(["A", "B"], 500), "pred_prob": 0.2})
    df["mortality_30d"] = np.r_[rng.random(500) < 0.2, rng.random(500) < 0.4].astype(int)
    cal = qm.calibration_by_group(df, "unit").set_index("unit")
    assert cal.loc["A", "calibration"] == "consistent"
    assert cal.loc["B", "calibration"] == "model under-predicts"
    assert cal.loc["B", "predicted"] == pytest.approx(100)


def test_calibration_by_decile_uses_equal_bins():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    df = pd.DataFrame({"pred_prob": p, "mortality_30d": (rng.random(1000) < p).astype(int)})
    dec = qm.calibration_by_decile(df)
    assert (dec["n"] == 100).all()
    assert dec["mean_predicted"].is_monotonic_increasing
    assert np.allclose(dec["observed_rate"], dec["mean_predicted"], atol=0.15)


def _stays():
    return pd.DataFrame({
        "group": ["a"] * 6,
        "mortality_30d": [1, 1, 0, 0, 0, 1],
        "hospital_expire_flag": [1, 1, 0, 0, 0, 0],
        "icu_death": [True, False, False, False, False, False],
        "icu_los_days": [3.0, 5.0, 2.0, 4.0, 6.0, 8.0],
        "hosp_los_days": [3.0, 9.0, 7.0, 8.0, 10.0, 12.0],
        "icu_readmit_48h": [np.nan, 1.0, 0.0, 0.0, 1.0, 0.0],
    })


def test_outcome_summary_splits_los_and_uses_icu_survivors():
    out = qm.outcome_summary(_stays(), "group").iloc[0]
    assert out["n"] == 6 and out["deaths_30d"] == 3 and out["icu_deaths"] == 1
    assert out["icu_los_decedents"] == 4.0          # median of the two hospital deaths (3, 5)
    assert out["icu_los_survivors"] == 5.0          # median of 2, 4, 6, 8
    assert out["icu_survivors"] == 5 and out["readmits_48h"] == 2
    assert out["readmit_48h_rate"] == pytest.approx(0.4)


def test_small_cells_suppressed():
    t = pd.DataFrame({"group": ["a", "b", "c", "d"], "n": [200, 8, 200, 15], "observed": [40, 2, 5, 10]})
    out = qm.suppress_small_cells(t).set_index("group")
    assert not out.loc["a", "suppressed"] and out.loc["a", "observed"] == 40
    assert out.loc["b", "suppressed"] and np.isnan(out.loc["b", "n"])   # small n
    assert out.loc["c", "suppressed"]                                    # 5 events
    assert out.loc["d", "suppressed"]                                    # 15 - 10 = 5 survivors


def test_suppression_can_target_columns_and_denominators():
    t = pd.DataFrame({"group": ["a"], "n": [500], "deaths": [120], "readmits": [4], "survivors": [400],
                      "readmit_rate": [0.01]})
    out = qm.suppress_small_cells(t, count_cols=(("readmits", "survivors"),), measure_cols=["readmits", "readmit_rate"])
    assert np.isnan(out.loc[0, "readmits"]) and np.isnan(out.loc[0, "readmit_rate"])
    assert out.loc[0, "deaths"] == 120 and out.loc[0, "n"] == 500
    # A second pass must not clobber the flag column or earlier results
    again = qm.suppress_small_cells(out, count_cols=("deaths",))
    assert again.loc[0, "suppressed"] and again.loc[0, "deaths"] == 120
