#tests/test_io_utils.py

import os
import tempfile
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.io_utils import (
    load_dataset,
    save_dataset,
    save_model,
    load_model_and_scaler,
)


def test_save_and_load_dataset(synthetic_features_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.parquet")
        save_dataset(synthetic_features_df, path)
        loaded_df = load_dataset(path)

        pd.testing.assert_frame_equal(
            synthetic_features_df.reset_index(drop=True),
            loaded_df.reset_index(drop=True)
        )

def test_load_dataset_missing_file():
    df = load_dataset("non_existent_file.parquet")
    assert df.empty

def test_save_and_load_model(synthetic_features_df):
    X = synthetic_features_df[["age", "mean_lab_0", "mean_lab_1"]]
    y = synthetic_features_df["mortality_30d"]

    model = LogisticRegression().fit(X, y)
    scaler = StandardScaler().fit(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        save_model(model, scaler, model_path)

        loaded_model, loaded_scaler = load_model_and_scaler(model_path)

        assert isinstance(loaded_model, LogisticRegression)
        assert isinstance(loaded_scaler, StandardScaler)
        assert len(loaded_model.predict(X)) == X.shape[0]
        assert loaded_scaler.transform(X).shape == X.shape
