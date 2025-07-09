# SHAP plotting and explanation logic
# src/oncoai_prototype/utils/shap_util.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_shap_explainer(model, X_scaled: np.ndarray, X_df: pd.DataFrame, output_dir: str, top_n: int = 10):
    os.makedirs(output_dir, exist_ok=True)

    if X_scaled.shape[0] == 0:
        print("❌ No samples to explain.")
        return

    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    # Global summary plot
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary (Beeswarm)")
    plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"))
    plt.close()

    # Top N high-risk
    top_idx = np.argsort(shap_values.values.sum(axis=1))[::-1][:top_n]
    for i in top_idx:
        fig = shap.plots.waterfall(shap_values[i], show=False)
        plt.savefig(os.path.join(output_dir, f"shap_waterfall_patient_{i}.png"))
        plt.close()


