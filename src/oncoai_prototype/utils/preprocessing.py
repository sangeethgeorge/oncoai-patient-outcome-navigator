# Shared preprocessing (split, scale, encode, impute)
# src/oncoai_prototype/utils/preprocessing.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def train_test_impute_split(df: pd.DataFrame, target_col: str = "mortality_30d", test_size=0.2, random_state=42):
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train = X_train.fillna(X_train.median(numeric_only=True))
    X_test = X_test.fillna(X_train.median(numeric_only=True))  # use train stats

    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def preprocess_for_inference(df: pd.DataFrame):
    df = df.drop(columns=["subject_id", "hadm_id", "icustay_id"], errors="ignore")
    return df.fillna(df.median(numeric_only=True))
