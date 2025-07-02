# src/predict.py

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import os
import mlflow

def load_model(model_path="models/logreg_model.joblib"):
    loaded = joblib.load(model_path)
    return loaded['model'], loaded['scaler']

def load_input(input_path):
    df = pd.read_parquet(input_path)
    return df

def preprocess_input(df, scaler, expected_columns):
    df = df.drop(columns=['icustay_id', 'subject_id', 'hadm_id', 'icd9_code', 'admittime', 'dob', 'dod', 'intime', 'outtime'], errors='ignore')
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=expected_columns, fill_value=0)
    df_scaled = scaler.transform(df)
    return df, df_scaled

def predict(model, X):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return y_pred, y_prob

def explain_predictions(model, X_scaled, X_df, output_dir="outputs", top_n=10):
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.Explainer(model, X_scaled, feature_names=X_df.columns)
    shap_values = explainer(X_scaled)

    risk_scores = model.predict_proba(X_scaled)[:, 1]
    top_indices = np.argsort(risk_scores)[-top_n:][::-1]

    print(f"\n📊 Generating SHAP plots for top {top_n} high-risk patients...")

    for i in top_indices:
        patient_id = f"patient_{i}"

        plt.figure()
        shap.plots.waterfall(shap_values[i], show=False)
        wf_path = os.path.join(output_dir, f"waterfall_{patient_id}.png")
        plt.title(f"Waterfall Plot - {patient_id}")
        plt.savefig(wf_path)
        mlflow.log_artifact(wf_path)
        plt.close()

        top_features = np.argsort(np.abs(shap_values.values[i]))[::-1][:3]
        for feat_idx in top_features:
            feat_name = X_df.columns[feat_idx]
            plt.figure()
            shap.dependence_plot(feat_name, shap_values.values, X_df, feature_names=X_df.columns, show=False)
            dp_path = os.path.join(output_dir, f"dependence_{patient_id}_{feat_name}.png")
            plt.title(f"Dependence Plot - {patient_id} - {feat_name}")
            plt.savefig(dp_path)
            mlflow.log_artifact(dp_path)
            plt.close()

    print(f"✅ SHAP plots saved and logged to MLflow from {output_dir}/")

def save_output(original_df, y_pred, y_prob, output_path):
    original_df['predicted_mortality_30d'] = y_pred
    original_df['predicted_probability'] = y_prob
    original_df.to_parquet(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")
    mlflow.log_artifact(output_path)

def main():
    parser = argparse.ArgumentParser(description="Run mortality prediction on new patient data.")
    parser.add_argument("--input", required=True, help="Path to input .parquet file")
    parser.add_argument("--output", required=True, help="Path to save prediction output")
    parser.add_argument("--model", default="models/logreg_model.joblib", help="Path to saved model")
    parser.add_argument("--shap_dir", default="outputs", help="Directory to save SHAP plots")
    args = parser.parse_args()

    mlflow.set_experiment("OncoAI-Inference")

    with mlflow.start_run():
        print("🔍 Loading model...")
        model, scaler = load_model(args.model)
        mlflow.log_param("model_path", args.model)

        print("📦 Loading input data...")
        df_input = load_input(args.input)
        mlflow.log_param("input_path", args.input)

        expected_columns = scaler.feature_names_in_

        print("⚙️ Preprocessing input...")
        X_df, X_scaled = preprocess_input(df_input, scaler, expected_columns)

        print("🤖 Making predictions...")
        y_pred, y_prob = predict(model, X_scaled)
        mlflow.log_metric("mean_predicted_risk", float(np.mean(y_prob)))
        mlflow.log_metric("high_risk_count", int(np.sum(y_prob > 0.5)))

        print("📈 Explaining predictions with SHAP...")
        explain_predictions(model, X_scaled, X_df, output_dir=args.shap_dir)

        print("💾 Saving predictions...")
        save_output(df_input, y_pred, y_prob, args.output)

if __name__ == "__main__":
    main()
