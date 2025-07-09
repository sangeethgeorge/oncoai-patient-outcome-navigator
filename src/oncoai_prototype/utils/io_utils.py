# Load/save data and models
# src/oncoai_prototype/utils/io_utils.py

import os
import pandas as pd
import joblib

def load_dataset(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        print(f"❌ Dataset not found at {path}")
        return pd.DataFrame()

def save_dataset(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved dataset to {path} — shape: {df.shape}")

def save_model(model, scaler, path: str):
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"✅ Model and scaler saved to {path}")

def load_model_and_scaler(path: str):
    if os.path.exists(path):
        artifact = joblib.load(path)
        return artifact['model'], artifact['scaler']
    else:
        raise FileNotFoundError(f"Model file not found at {path}")

