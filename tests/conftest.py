# tests/conftest.py
# Shared fixtures: synthetic data for unit tests, and a live-Postgres fixture for `-m db` tests.

import os
import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv


@pytest.fixture
def synthetic_features_df():
    """Feature-table shaped like onco_features_cleaned.parquet, with signal in two features.
    A few patients have two stays so patient-grouped splitting is actually exercised."""
    rng = np.random.default_rng(0)
    n = 400
    subject_id = np.arange(n) + 1000
    subject_id[::10] = subject_id[1::10][: len(subject_id[::10])]  # 40 patients with 2 stays
    df = pd.DataFrame({
        "subject_id": subject_id,
        "hadm_id": np.arange(n) + 5000,
        "icustay_id": np.arange(n) + 9000,
        "age": rng.integers(18, 90, n).astype(float),
        "n_cancer_codes": rng.integers(1, 5, n).astype(float),
    })
    for i in range(10):
        df[f"mean_lab_{i}"] = rng.normal(0, 1, n)
    logit = -1.8 + 1.5 * df["mean_lab_0"] - 1.0 * df["mean_lab_1"] + 0.02 * (df["age"] - 60)
    df["mortality_30d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[rng.choice(n, 30, replace=False), "mean_lab_5"] = np.nan
    return df


# --- Live database (skipped unless ONCOAI_POSTGRES_CONN_STR points at a reachable MIMIC DB) ---
@pytest.fixture(scope="session")
def db_conn_str():
    load_dotenv()
    conn_str = os.getenv("ONCOAI_POSTGRES_CONN_STR")
    if not conn_str:
        pytest.skip("ONCOAI_POSTGRES_CONN_STR not set")
    from oncoai_prototype.utils.db_utils import run_postgres_sql, redact
    try:
        run_postgres_sql(conn_str, "SELECT 1")
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {redact(str(e))}")
    return conn_str


@pytest.fixture(scope="session")
def db_data(db_conn_str):
    from oncoai_prototype.data_processing.feature_engineering import load_data
    return load_data()


@pytest.fixture(scope="session")
def db_features_df(db_data):
    from oncoai_prototype.data_processing.feature_engineering import build_features
    return build_features(db_data)
