# Shared preprocessing (split, scale, encode, impute)
# src/oncoai_prototype/utils/preprocessing.py

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

ID_COLS = ['icustay_id', 'subject_id', 'hadm_id']
NON_FEATURE_COLS = ID_COLS + ['admittime', 'dob', 'dod', 'intime', 'outtime', 'icd9_code', 'icd9_codes']


def grouped_stratified_split(y: pd.Series, groups: pd.Series, test_size=0.2, random_state=42):
    """One fold of StratifiedGroupKFold: no group (patient) lands in both sets."""
    n_splits = round(1 / test_size)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    return y.index[train_idx], y.index[test_idx]


def train_test_impute_split(df: pd.DataFrame, target_col: str = "mortality_30d", group_col: str = "subject_id",
                            test_size=0.2, random_state=42):
    """Patient-grouped, stratified split. X keeps df's index so callers can map rows back to IDs."""
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col] + [c for c in NON_FEATURE_COLS if c in df.columns])
    y = df[target_col]

    train_idx, test_idx = grouped_stratified_split(y, df[group_col], test_size, random_state)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    medians = X_train.median(numeric_only=True)  # train stats only
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, X_test, y_train, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def preprocess_for_inference(df: pd.DataFrame):
    df = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
    return df.fillna(df.median(numeric_only=True))
