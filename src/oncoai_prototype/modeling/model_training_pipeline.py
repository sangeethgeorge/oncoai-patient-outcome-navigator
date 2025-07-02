# src/model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os

# --- Configuration ---
project_root = '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator'
data_file_path = os.path.join(project_root, "data", "onco_features_cleaned.parquet")
model_save_base_path = os.path.join(project_root, "models")
shap_plots_base_path = os.path.join(project_root, "shap_plots")

os.makedirs(os.path.dirname(data_file_path), exist_ok=True)
os.makedirs(model_save_base_path, exist_ok=True)
os.makedirs(shap_plots_base_path, exist_ok=True)

# --- Data Loading Function ---
def load_dataset(path: str = data_file_path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
        print(f"✅ Dataset loaded successfully from {path}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {path}. Please ensure the file exists and the path is correct.")
        return pd.DataFrame()

# --- Data Preprocessing Functions ---
def train_test_impute_split(df: pd.DataFrame, label_col: str = "mortality_30d") -> tuple:
    df = df.drop(columns=['icustay_id', 'subject_id', 'hadm_id', 'admittime', 'dob', 'dod', 'intime', 'outtime', 'icd9_code'], errors='ignore')
    y = df[label_col]
    X = df.drop(columns=[label_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    for col in X_train.select_dtypes(include=np.number).columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    print("✅ Data split and imputed successfully.")
    return X_train, X_test, y_train, y_test

def check_for_leakage(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_copy = X.copy()
    y_copy = y.copy()
    X_copy.index = range(len(X_copy))
    y_copy.index = range(len(y_copy))
    combined_df = pd.concat([X_copy, y_copy], axis=1)
    numeric_cols = combined_df.select_dtypes(include=np.number).columns
    combined_df_numeric = combined_df[numeric_cols]
    if y.name not in combined_df_numeric.columns:
        print(f"⚠️ Warning: Target column '{y.name}' not found in numeric columns for leakage check. Skipping correlation check.")
        return X
    corr = combined_df_numeric.corr()[y.name].drop(y.name, errors='ignore')
    high_corr_threshold = 0.95
    high_corr = corr[abs(corr) > high_corr_threshold]
    if not high_corr.empty:
        print(f"\n⚠️ Potential Leakage Detected (correlation > {high_corr_threshold}):")
        print(high_corr)
        leaky_columns = high_corr.index.tolist()
        X = X.drop(columns=leaky_columns, errors='ignore')
        print(f"Dropped potential leakage columns: {leaky_columns}")
    else:
        print("\nNo significant data leakage detected based on high correlation.")
    return X

# --- Model Training and Evaluation ---
def train_logistic_regression(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series) -> tuple:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\n🧠 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n📊 ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("✅ Logistic Regression model trained and evaluated.")
    return model, X_train, y_train, X_test, y_test, y_pred, y_prob

# --- Model Saving Function ---
def save_model(model, scaler, output_path: str):
    joblib.dump({"model": model, "scaler": scaler}, output_path)
    print(f"\n✅ Saved model and scaler to {output_path}")

# --- Main MLflow Execution Block ---
if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model_save_path_for_run = os.path.join(model_save_base_path, f"logreg_model_run_{run_id}.joblib")
        print(f"Starting MLflow Run with ID: {run_id}")
        df = load_dataset()
        if df.empty:
            print("❌ Dataset is empty. Cannot proceed with training. Exiting MLflow run.")
            mlflow.end_run(status="FAILED")
        else:
            X_train, X_test, y_train, y_test = train_test_impute_split(df)
            X_train_ohe = pd.get_dummies(X_train, drop_first=True)
            X_test_ohe = pd.get_dummies(X_test, drop_first=True)
            missing_cols_in_test = set(X_train_ohe.columns) - set(X_test_ohe.columns)
            for c in missing_cols_in_test:
                X_test_ohe[c] = 0
            X_test_ohe = X_test_ohe[X_train_ohe.columns]
            X_train_leakage_checked = check_for_leakage(X_train_ohe, y_train)
            X_test_leakage_checked = X_test_ohe[X_train_leakage_checked.columns]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_leakage_checked)
            X_test_scaled = scaler.transform(X_test_leakage_checked)
            print("✅ Features prepared (one-hot encoded and scaled).")
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_param("model_type", "LogisticRegression")
            model, X_train_final_scaled, y_train_final, X_test_final_scaled, y_test_final, y_pred, y_prob = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
            auc = roc_auc_score(y_test_final, y_prob)
            mlflow.log_metric("roc_auc", auc)
            save_model(model, scaler, output_path=model_save_path_for_run)
            if X_train_leakage_checked.shape[0] > 0:
                mlflow.sklearn.log_model(model, "logreg_model", input_example=X_train_leakage_checked.head(10))
            else:
                mlflow.sklearn.log_model(model, "logreg_model")
    print("\n✨ MLflow run completed successfully. Check your MLflow UI for details.")
