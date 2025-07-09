#tests/test_io_utils.py

import os
import pandas as pd
import tempfile
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.io_utils import (
    load_dataset,
    save_dataset,
    save_model,
    load_model_and_scaler,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")

@pytest.fixture
def sample_df():
    if os.path.exists(DATA_PATH):
        return pd.read_parquet(DATA_PATH).sample(n=10, random_state=42)  # keep it small
    else:
        pytest.skip("ML-ready cohort not found. Skipping test.")

def test_save_and_load_dataset(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.parquet")
        save_dataset(sample_df, path)
        loaded_df = load_dataset(path)
        pd.testing.assert_frame_equal(sample_df, loaded_df)

def test_load_dataset_missing_file():
    df = load_dataset("non_existent_file.parquet")
    assert df.empty

# Load/save data and models
# src/oncoai_prototype/utils/io_utils.py

import os
import pandas as pd
import joblib

def load_dataset(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        print(f"❌ Dataset not found at {path}")
        return pd.DataFrame()

def save_dataset(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved dataset to {path} — shape: {df.shape}")

def save_model(model, scaler, path: str):
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"✅ Model and scaler saved to {path}")

def load_model_and_scaler(path: str):
    if os.path.exists(path):
        artifact = joblib.load(path)
        return artifact['model'], artifact['scaler']
    else:
        raise FileNotFoundError(f"Model file not found at {path}")
def test_save_and_load_dataset(sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.parquet")
        save_dataset(sample_df, path)
        loaded_df = load_dataset(path)

        # 🔑 Reset both indices before comparing
        pd.testing.assert_frame_equal(
            sample_df.reset_index(drop=True),
            loaded_df.reset_index(drop=True)
        )

