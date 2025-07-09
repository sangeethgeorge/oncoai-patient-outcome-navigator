#tests/test_preprocessing.py

import os
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.preprocessing import (
    train_test_impute_split,
    scale_features,
    preprocess_for_inference
)

# Configuration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")

@pytest.fixture(scope="module")
def sample_df():
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
        return df.sample(n=100, random_state=42)  # Use a small random subset
    else:
        pytest.skip("ML-ready cohort not found. Skipping preprocessing tests.")

def test_train_test_impute_split(sample_df):
    X_train, X_test, y_train, y_test = train_test_impute_split(sample_df)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert not X_train.isnull().values.any()
    assert not X_test.isnull().values.any()
    assert y_train.isin([0, 1]).all()

def test_scale_features(sample_df):
    X_train, X_test, y_train, y_test = train_test_impute_split(sample_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert isinstance(scaler, StandardScaler)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0, atol=1e-1)  # Check zero-centered

def test_preprocess_for_inference(sample_df):
    df = sample_df.drop(columns=["mortality_30d"])
    df_processed = preprocess_for_inference(df)

    assert not df_processed.isnull().values.any()
    assert "subject_id" not in df_processed.columns

