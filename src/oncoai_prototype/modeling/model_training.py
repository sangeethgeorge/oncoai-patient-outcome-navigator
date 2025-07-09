# src/oncoai_prototype/modeling/model_training.py

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.types.utils import _infer_schema

# Custom utility imports
from oncoai_prototype.utils.io_utils import load_dataset
from oncoai_prototype.utils.preprocessing import train_test_impute_split, scale_features
from oncoai_prototype.utils.leakage import check_for_leakage

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "onco_features_cleaned.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models") # Used for temporary feature_names file
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Model Training ---
def train_logistic_regression(X_train, y_train, X_test, y_test):
    # X_train, X_test here are expected to be DataFrames
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    return model, y_pred, y_prob

# --- Pipeline Orchestration ---
def run_training_pipeline():
    print("Loading ML-ready cohort...")
    df = load_dataset(DATA_PATH)
    if df.empty:
        print("Dataset not found or empty.")
        return None, None, None, None, None, None, None # Return None for all expected values

    # Define columns to drop (identifiers)
    # IMPORTANT: Adjust this list based on your actual data and what truly are non-feature columns
    columns_to_drop = ['subject_id', 'hadm_id', 'icustay_id']
    # Drop columns that are not meant to be features for the model or scaler
    df_features = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    print("Splitting + imputing data...")
    # Pass df_features instead of df
    X_train, X_test, y_train, y_test = train_test_impute_split(df_features)

    print("Checking for data leakage...")
    X_train_processed = check_for_leakage(X_train, y_train)
    X_test_processed = X_test.reindex(columns=X_train_processed.columns)

    print("Scaling features...")
    scaler, X_train_scaled_array, X_test_scaled_array = scale_features(X_train_processed, X_test_processed)

    X_train_scaled_df = pd.DataFrame(X_train_scaled_array, columns=X_train_processed.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_array, columns=X_test_processed.columns)

    print("Training logistic regression model...")
    model, y_pred, y_prob = train_logistic_regression(X_train_scaled_df, y_train, X_test_scaled_df, y_test)

    print("Model training complete.")

    return model, y_test, y_pred, y_prob, X_train_processed, scaler, X_train_scaled_df

# --- Main Execution ---
if __name__ == "__main__":
    mlflow.set_experiment("OncoAI-Mortality-Prediction")
    with mlflow.start_run(run_name=f"logreg_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        result = run_training_pipeline()

        if result[0] is None:
            print("Training aborted due to missing or invalid data.")
        else:
            model, y_test, y_pred, y_prob, X_train_for_mlflow_orig, scaler, X_train_scaled_for_mlflow_df = result

            auc = roc_auc_score(y_test, y_prob)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_metric("roc_auc", auc)

            # Save feature names to a temporary file locally
            feature_names_file = os.path.join(MODEL_DIR, "feature_names.txt")
            with open(feature_names_file, "w") as f:
                for item in X_train_for_mlflow_orig.columns:
                    f.write(f"{item}\n")

            # Log the feature_names.txt file as an artifact directly associated with the run
            mlflow.log_artifact(local_path=feature_names_file, artifact_path="extra_files")

            # --- Log the main model ---
            model_signature = infer_signature(X_train_scaled_for_mlflow_df, model.predict(X_train_scaled_for_mlflow_df))
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="OncoAICancerMortalityPredictor",
                signature=model_signature,
                input_example=X_train_scaled_for_mlflow_df.iloc[:5]
            )

            mlflow.register_model(
                model_uri=model_info.model_uri,
                name="OncoAICancerMortalityPredictor"
            )

            # --- Log and register the scaler ---
            scaler_input_example = X_train_for_mlflow_orig.iloc[:5]
            scaler_output_example_array = scaler.transform(scaler_input_example)
            scaler_output_example_df = pd.DataFrame(scaler_output_example_array, columns=X_train_for_mlflow_orig.columns)

            scaler_signature = ModelSignature(
                inputs=_infer_schema(scaler_input_example),
                outputs=_infer_schema(scaler_output_example_df)
            )

            scaler_info = mlflow.sklearn.log_model(
                sk_model=scaler,
                name="OncoAIScaler",
                signature=scaler_signature,
                input_example=scaler_input_example
            )
            mlflow.register_model(
                model_uri=scaler_info.model_uri,
                name="OncoAIScaler"
            )

            # Cleanup local artifact file
            os.remove(feature_names_file)

            print("Training run logged to MLflow.")