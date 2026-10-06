# tests/test_shap_utils.py

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from oncoai_prototype.utils.shap_utils import run_shap_explainer
from oncoai_prototype.utils.preprocessing import train_test_impute_split


@pytest.mark.filterwarnings("ignore:.*disp.*deprecated.*:DeprecationWarning")
def test_run_shap_explainer_writes_plots(synthetic_features_df):
    X_train, X_test, y_train, y_test = train_test_impute_split(synthetic_features_df, target_col="mortality_30d")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)

    top_n_to_check = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        run_shap_explainer(
            model=model,
            X_scaled=X_test_scaled.values,
            X_df=X_test_scaled,
            output_dir=tmpdir,
            top_n=top_n_to_check
        )

        assert os.path.exists(os.path.join(tmpdir, "shap_summary_beeswarm.png"))

        shap_values = shap.Explainer(model, X_test_scaled)(X_test_scaled)
        top_idx = np.argsort(shap_values.values.sum(axis=1))[::-1][:top_n_to_check]
        for i in top_idx:
            assert os.path.exists(os.path.join(tmpdir, f"shap_waterfall_patient_{i}.png"))
