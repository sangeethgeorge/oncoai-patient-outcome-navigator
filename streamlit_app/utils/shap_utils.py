# utils/shap_utils.py

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHAP_IMG_PATH = os.path.join(PROJECT_ROOT, "shap_temp_waterfall.png")

def generate_shap_explanation(model, scaler, X_input, background):
    explainer = shap.LinearExplainer(model, background)
    shap_vals = explainer.shap_values(X_input)
    return shap.Explanation(
        values=shap_vals[0],
        base_values=explainer.expected_value,
        data=X_input.iloc[0].values,
        feature_names=X_input.columns.tolist()
    )

def plot_waterfall(shap_expl):
    shap.plots.waterfall(shap_expl, show=False)
    plt.title("SHAP Waterfall: Patient Risk Factors")
    plt.savefig(SHAP_IMG_PATH, bbox_inches='tight')
    plt.close()
    return SHAP_IMG_PATH

def create_shap_table(user_input_df, shap_expl):
    df = pd.DataFrame({
        "Feature": user_input_df.columns,
        "Input Value": user_input_df.iloc[0].values,
        "SHAP Value": shap_expl.values,
        "Impact": ["⬆️ Increases Risk" if v > 0 else "⬇️ Decreases Risk" for v in shap_expl.values]
    })
    df["|SHAP|"] = np.abs(df["SHAP Value"])
    return df.sort_values(by="|SHAP|", ascending=False).drop(columns="|SHAP|")
