# tests/test_cross_fitted.py

import numpy as np
from sklearn.metrics import roc_auc_score

from oncoai_prototype.modeling import model_training as mt


def _inputs(df):
    X = df.drop(columns=["subject_id", "hadm_id", "icustay_id", "mortality_30d"])
    return X, df["mortality_30d"], df["subject_id"], df["icustay_id"]


def test_every_stay_scored_exactly_once(synthetic_features_df):
    X, y, groups, ids = _inputs(synthetic_features_df)
    oof = mt.cross_fitted_predictions(X, y, groups, ids, top_n=4, n_splits=5)
    assert sorted(oof["icustay_id"]) == sorted(ids)
    assert oof["pred_prob"].between(0, 1).all()


def test_patient_scored_by_one_fold_only(synthetic_features_df):
    # Patients with two stays must land in the same fold, so the scoring model never saw them
    X, y, groups, ids = _inputs(synthetic_features_df)
    oof = mt.cross_fitted_predictions(X, y, groups, ids, top_n=4, n_splits=5)
    folds = oof.merge(synthetic_features_df[["icustay_id", "subject_id"]]).groupby("subject_id")["fold"].nunique()
    assert (folds == 1).all()


def test_selection_never_sees_scored_fold(synthetic_features_df, monkeypatch):
    seen = []
    real_select = mt.select_features_shap
    monkeypatch.setattr(mt, "select_features_shap",
                        lambda X_tr, y_tr, top_n=mt.TOP_N, seed=mt.SEED: seen.append(set(X_tr.index)) or real_select(X_tr, y_tr, top_n, seed))
    X, y, groups, ids = _inputs(synthetic_features_df)
    oof = mt.cross_fitted_predictions(X, y, groups, ids, top_n=4, n_splits=5)
    id_to_row = dict(zip(ids, X.index))
    for fold, rows in enumerate(seen):
        scored = {id_to_row[i] for i in oof.loc[oof["fold"] == fold, "icustay_id"]}
        assert rows.isdisjoint(scored)


def test_out_of_fold_predictions_carry_signal(synthetic_features_df):
    X, y, groups, ids = _inputs(synthetic_features_df)
    oof = mt.cross_fitted_predictions(X, y, groups, ids, top_n=4, n_splits=5)
    y_oof = y.set_axis(ids).loc[oof["icustay_id"]]
    assert roc_auc_score(y_oof, oof["pred_prob"]) > 0.7
    # Calibrated in the large: total expected deaths close to observed
    assert np.isclose(oof["pred_prob"].sum(), y.sum(), rtol=0.15)
