# tests/test_shap_utils.py

import os
import tempfile
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from oncoai_prototype.utils.shap_utils import run_shap_explainer
from oncoai_prototype.utils.preprocessing import train_test_impute_split
import numpy as np
import shap 


# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")

@pytest.fixture(scope="module")
def real_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
        return df.sample(n=200, random_state=42)  # limit to 200 for fast test
    else:
        pytest.skip("ML-ready cohort not found.")

def test_run_shap_explainer_with_real_data(real_data):
    # Preprocess
    X_train, X_test, y_train, y_test = train_test_impute_split(real_data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    top_n_to_check = 3 # Define top_n here

    # Run SHAP on test data
    with tempfile.TemporaryDirectory() as tmpdir:
        run_shap_explainer(
            model=model,
            X_scaled=X_test_scaled,
            X_df=X_test,
            output_dir=tmpdir,
            top_n=top_n_to_check # Use the defined top_n
        )

        # Assertions
        summary_path = os.path.join(tmpdir, "shap_summary_beeswarm.png")
        assert os.path.exists(summary_path)

        # Generate SHAP values to determine expected filenames for waterfall plots
        # This part replicates the logic in run_shap_explainer to get the actual indices
        explainer = shap.Explainer(model, X_test_scaled)
        shap_values = explainer(X_test_scaled)
        top_idx = np.argsort(shap_values.values.sum(axis=1))[::-1][:top_n_to_check]

        # Assert that a waterfall plot exists for each of the top_n_to_check patients
        for i in top_idx:
            waterfall_path = os.path.join(tmpdir, f"shap_waterfall_patient_{i}.png")
            assert os.path.exists(waterfall_path)
