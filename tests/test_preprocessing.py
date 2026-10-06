#tests/test_preprocessing.py

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.preprocessing import (
    train_test_impute_split,
    scale_features,
    preprocess_for_inference,
    NON_FEATURE_COLS,
)


def test_split_has_no_patient_overlap(synthetic_features_df):
    df = synthetic_features_df
    X_train, X_test, y_train, y_test = train_test_impute_split(df, target_col="mortality_30d")

    train_subjects = set(df.loc[X_train.index, "subject_id"])
    test_subjects = set(df.loc[X_test.index, "subject_id"])
    assert train_subjects.isdisjoint(test_subjects)
    assert len(X_train) + len(X_test) == len(df)


def test_split_is_stratified(synthetic_features_df):
    df = synthetic_features_df
    _, _, y_train, y_test = train_test_impute_split(df, target_col="mortality_30d")
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert 0.15 < len(y_test) / len(df) < 0.25


def test_split_drops_ids_and_imputes_with_train_medians(synthetic_features_df):
    df = synthetic_features_df
    X_train, X_test, _, _ = train_test_impute_split(df, target_col="mortality_30d")

    assert not set(NON_FEATURE_COLS) & set(X_train.columns)
    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()

    train_median = df.loc[X_train.index, "mean_lab_5"].median()
    missing_in_test = df.loc[X_test.index, "mean_lab_5"].isna()
    if missing_in_test.any():
        assert (X_test.loc[missing_in_test[missing_in_test].index, "mean_lab_5"] == train_median).all()


def test_scale_features(synthetic_features_df):
    X_train, X_test, _, _ = train_test_impute_split(synthetic_features_df, target_col="mortality_30d")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert isinstance(scaler, StandardScaler)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-6)


def test_preprocess_for_inference(synthetic_features_df):
    df = synthetic_features_df.drop(columns=["mortality_30d"])
    df_processed = preprocess_for_inference(df)

    for col in NON_FEATURE_COLS:
        assert col not in df_processed.columns
    assert not df_processed.isnull().values.any()
