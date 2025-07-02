# src/predict.py

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import mlflow
import os
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
project_root = '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator'
data_file_path = os.path.join(project_root, "data", "onco_features_cleaned.parquet")
model_load_path = os.path.join(project_root, "models", "logreg_model.joblib")
shap_output_path = os.path.join(project_root, "shap_plots", "inference")

os.makedirs(shap_output_path, exist_ok=True)

# --- Load data ---
def load_data():
    try:
        df = pd.read_parquet(data_file_path)
        print(f"✅ Data loaded from {data_file_path}")
        return df
    except FileNotFoundError:
        print(f"❌ Data file not found at {data_file_path}")
        return pd.DataFrame()

# --- Preprocess data ---
def preprocess(df):
    df = df.drop(columns=['icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob', 'dod', 'intime', 'outtime', 'icd9_code'], errors='ignore')
    if 'mortality_30d' in df.columns:
        df = df.drop(columns=['mortality_30d'])
    df = pd.get_dummies(df, drop_first=True)
    return df

# --- Load model and scaler ---
def load_model():
    artifact = joblib.load(model_load_path)
    print(f"✅ Model loaded from {model_load_path}")
    return artifact['model'], artifact['scaler']

# --- Run prediction and explainability ---
def run_inference():
    df = load_data()
    if df.empty:
        return

    X = preprocess(df)
    model, scaler = load_model()

    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    df_results = df.copy()
    df_results['predicted_mortality_30d'] = y_pred
    df_results['predicted_probability'] = y_prob
    print(df_results[['predicted_mortality_30d', 'predicted_probability']].head())

    # SHAP explanations
    explainer = shap.Explainer(model, X_scaled, feature_names=X.columns.tolist())
    shap_values = explainer(X_scaled)

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features=X, show=False)
    plt.savefig(os.path.join(shap_output_path, "shap_summary_inference.png"), bbox_inches='tight')
    plt.close()

    # Waterfall and dependence plots for top 5 high-risk patients
    top_indices = np.argsort(y_prob)[-5:][::-1]
    for rank, i in enumerate(top_indices):
        id_tag = f"patient_{i}_rank_{rank+1}"

        # Waterfall plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[i], show=False)
        plt.title(f"SHAP Waterfall: {id_tag}")
        plt.savefig(os.path.join(shap_output_path, f"waterfall_{id_tag}.png"), bbox_inches='tight')
        plt.close()

        # Dependence plots
        shap_vals = shap_values.values[i]
        top_features = np.argsort(np.abs(shap_vals))[::-1][:3]
        for feat_idx in top_features:
            feat_name = X.columns[feat_idx]
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feat_name, shap_values.values, features=X, feature_names=X.columns.tolist(), show=False)
            plt.title(f"Dependence: {id_tag} - {feat_name}")
            plt.savefig(os.path.join(shap_output_path, f"dependence_{id_tag}_{feat_name}.png"), bbox_inches='tight')
            plt.close()

    print("✅ Inference and SHAP explanations completed.")

if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name="inference"):
        run_inference()
        mlflow.log_artifacts(shap_output_path, artifact_path="inference_shap_plots")
    print("✨ Inference run logged to MLflow.")
