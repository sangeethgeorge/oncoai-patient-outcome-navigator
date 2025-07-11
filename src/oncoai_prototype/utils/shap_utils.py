# SHAP plotting and explanation logic
# src/oncoai_prototype/utils/shap_util.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import mlflow

def run_shap_explainer(
    model,
    X_scaled: np.ndarray,
    X_df: pd.DataFrame,
    output_dir: str,
    X_raw: pd.DataFrame = None,
    log_to_mlflow: bool = False,
    mlflow_path: str = None,
    top_n: int = 10
):
    os.makedirs(output_dir, exist_ok=True)

    if X_df.shape[0] == 0:
        print("❌ No samples to explain.")
        return

    # 🔓 Unwrap MLflow PyFunc model if needed
    if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
        model_to_explain = model._model_impl.python_model.model
    else:
        model_to_explain = model

    explainer = shap.Explainer(model_to_explain, X_df)
    shap_values = explainer(X_df)

    # Beeswarm plot
    summary_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    if log_to_mlflow and mlflow_path:
        mlflow.log_artifact(summary_path, artifact_path=mlflow_path)

    # Waterfall plots for top N impactful predictions
    top_idx = np.argsort(shap_values.values.sum(axis=1))[::-1][:top_n]
    for i in top_idx:
        fig_path = os.path.join(output_dir, f"shap_waterfall_patient_{i}.png")
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[i], show=False)
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()

        if log_to_mlflow and mlflow_path:
            mlflow.log_artifact(fig_path, artifact_path=mlflow_path)
