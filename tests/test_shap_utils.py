# tests/test_shap_utils.py

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import duckdb
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.shap_utils import run_shap_explainer
from oncoai_prototype.utils.preprocessing import train_test_impute_split
from oncoai_prototype.utils.feature_utils import (
    filter_high_coverage,
    compute_time_series_features,
    merge_features,
    filter_and_impute
)
from oncoai_prototype.data_processing.feature_engineering import load_data as load_actual_data
import shap


# --- Fixture to prepare SHAP-compatible data from PostgreSQL ---
@pytest.fixture(scope="module")
def shap_ready_df():
    load_dotenv()

    try:
        duckdb.sql("INSTALL postgres_scanner;")
    except duckdb.CatalogException:
        pass
    duckdb.sql("LOAD postgres_scanner;")

    data = load_actual_data()

    vitals = filter_high_coverage(data["vitals"], label_col="vitals_label", min_coverage=0.95)
    labs = filter_high_coverage(data["labs"], label_col="labs_label", min_coverage=0.70)

    vitals_feat = compute_time_series_features(
        df=vitals,
        time_col="charttime",
        value_col="vitals_valuenum",
        label_col="vitals_label",
        icu_id_col="icustay_id"
    )
    labs_feat = compute_time_series_features(
        df=labs,
        time_col="charttime",
        value_col="labs_valuenum",
        label_col="labs_label",
        icu_id_col="icustay_id"
    )

    df = merge_features(data["cohort"], vitals_feat, labs_feat)
    df_cleaned = filter_and_impute(df, min_col_coverage=0.8)

    if "mortality_30d" not in df_cleaned.columns:
        pytest.skip("Target column 'mortality_30d' missing in SHAP-ready data.")

    return df_cleaned.sample(n=min(200, len(df_cleaned)), random_state=42)


# --- SHAP test using real data ---
@pytest.mark.filterwarnings("ignore:.*disp.*deprecated.*:DeprecationWarning")
def test_run_shap_explainer_with_real_data(shap_ready_df):
    X_train, X_test, y_train, y_test = train_test_impute_split(shap_ready_df, target_col="mortality_30d")

    # Only keep numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test[X_train.columns]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    top_n_to_check = 3

    # Run SHAP
    with tempfile.TemporaryDirectory() as tmpdir:
        run_shap_explainer(
            model=model,
            X_scaled=X_test_scaled,
            X_df=X_test,
            output_dir=tmpdir,
            top_n=top_n_to_check
        )

        # Assert summary plot exists
        summary_path = os.path.join(tmpdir, "shap_summary_beeswarm.png")
        assert os.path.exists(summary_path)

        # SHAP explanation
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        top_idx = np.argsort(shap_values.values.sum(axis=1))[::-1][:top_n_to_check]

        for i in top_idx:
            waterfall_path = os.path.join(tmpdir, f"shap_waterfall_patient_{i}.png")
            assert os.path.exists(waterfall_path)

