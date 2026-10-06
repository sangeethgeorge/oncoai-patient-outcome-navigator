# tests/test_model_training.py

import numpy as np
import pandas as pd
import pytest

from oncoai_prototype.modeling import model_training as mt


def test_feature_selection_never_sees_test_rows(synthetic_features_df, monkeypatch):
    seen = []
    real_select = mt.select_features_shap

    def recording_select(X_train, y_train, top_n=mt.TOP_N, seed=mt.SEED):
        seen.append(set(X_train.index))
        return real_select(X_train, y_train, top_n, seed)

    monkeypatch.setattr(mt, "select_features_shap", recording_select)
    monkeypatch.setattr(mt, "grouped_cv_auc", lambda *a, **k: (0.0, 0.0))  # keep the test fast
    # Cross-fitting scores every stay by design (analytics only); this test is about the published model
    monkeypatch.setattr(mt, "cross_fitted_predictions",
                        lambda X, y, groups, ids, *a, **k: pd.DataFrame({"icustay_id": ids.to_numpy(), "fold": 0,
                                                                          "pred_prob": np.linspace(0.1, 0.9, len(ids))}))
    result = mt.run_training_pipeline(synthetic_features_df, top_n=4)

    df = synthetic_features_df
    train_subjects = set(df.loc[result["X_train"].index, "subject_id"])
    test_rows = set(df.index[~df["subject_id"].isin(train_subjects)])
    assert test_rows, "expected a non-empty test split"
    for rows in seen:
        assert rows.isdisjoint(test_rows)


def test_training_pipeline_end_to_end(synthetic_features_df):
    result = mt.run_training_pipeline(synthetic_features_df, top_n=4)
    m = result["metrics"]

    assert m["n_stays"] == len(synthetic_features_df)
    assert m["n_train"] + m["n_test"] == m["n_stays"]
    assert m["test_roc_auc_ci_low"] <= m["test_roc_auc"] <= m["test_roc_auc_ci_high"]
    # Signal lives in mean_lab_0 / mean_lab_1; the model should find it and beat the age-only baseline
    assert {"mean_lab_0", "mean_lab_1"} <= set(result["features"])
    assert m["test_roc_auc"] > m["baseline_roc_auc"]
    assert len(result["features"]) == 4


def test_training_pipeline_rejects_duplicate_stays(synthetic_features_df):
    dup = pd.concat([synthetic_features_df, synthetic_features_df.head(5)])
    with pytest.raises(ValueError, match="duplicate rows"):
        mt.run_training_pipeline(dup)


def test_calibration_slope_is_one_for_calibrated_probs():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert mt.calibration_slope(y, p) == pytest.approx(1.0, abs=0.08)


def test_feature_ranges_are_aggregates(synthetic_features_df):
    ranges = mt.feature_ranges(synthetic_features_df[["age", "mean_lab_0"]])
    assert set(ranges) == {"age", "mean_lab_0"}
    assert ranges["age"]["low"] <= ranges["age"]["default"] <= ranges["age"]["high"]
