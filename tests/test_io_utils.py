#tests/test_io_utils.py

import os
import pandas as pd
import tempfile
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import duckdb
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.io_utils import (
    load_dataset,
    save_dataset,
    save_model,
    load_model_and_scaler,
)
from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_and_impute
)

from oncoai_prototype.data_processing.feature_engineering import load_data as load_actual_data


# --- Fixture to generate a small numeric dataset from PostgreSQL ---
@pytest.mark.filterwarnings("ignore:.*disp.*deprecated.*:DeprecationWarning")
@pytest.fixture(scope="module")
def numeric_sample_df():
    load_dotenv()

    try:
        duckdb.sql("INSTALL postgres_scanner;")
    except duckdb.CatalogException:
        pass
    duckdb.sql("LOAD postgres_scanner;")

    raw_data = load_actual_data()

    vitals = filter_high_coverage(raw_data['vitals'], label_col='vitals_label', min_coverage=0.95)
    labs = filter_high_coverage(raw_data['labs'], label_col='labs_label', min_coverage=0.70)

    vitals_feat = compute_time_series_features(
        vitals, time_col='charttime', value_col='vitals_valuenum',
        label_col='vitals_label', icu_id_col='icustay_id'
    )
    labs_feat = compute_time_series_features(
        labs, time_col='charttime', value_col='labs_valuenum',
        label_col='labs_label', icu_id_col='icustay_id'
    )

    merged = merge_features(raw_data['cohort'], vitals_feat, labs_feat)
    df_cleaned = filter_and_impute(merged, min_col_coverage=0.8)

    if 'mortality_30d' not in df_cleaned.columns:
        pytest.skip("Skipping test: 'mortality_30d' not found in cleaned data.")

    numeric_df = df_cleaned.select_dtypes(include=[np.number])
    return numeric_df.sample(n=min(10, len(numeric_df)), random_state=42)


# --- Tests ---

def test_save_and_load_dataset(numeric_sample_df):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_data.parquet")
        save_dataset(numeric_sample_df, path)
        loaded_df = load_dataset(path)

        # Ensure consistent structure
        pd.testing.assert_frame_equal(
            numeric_sample_df.reset_index(drop=True),
            loaded_df.reset_index(drop=True)
        )

def test_load_dataset_missing_file():
    df = load_dataset("non_existent_file.parquet")
    assert df.empty

def test_save_and_load_model(numeric_sample_df):
    X = numeric_sample_df.drop(columns=["mortality_30d"], errors="ignore")
    y = numeric_sample_df["mortality_30d"]

    model = LogisticRegression().fit(X, y)
    scaler = StandardScaler().fit(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        save_model(model, scaler, model_path)

        loaded_model, loaded_scaler = load_model_and_scaler(model_path)

        # Confirm deserialized objects work
        preds = loaded_model.predict(X)
        scaled = loaded_scaler.transform(X)

        assert isinstance(loaded_model, LogisticRegression)
        assert isinstance(loaded_scaler, StandardScaler)
        assert len(preds) == X.shape[0]
        assert scaled.shape == X.shape
