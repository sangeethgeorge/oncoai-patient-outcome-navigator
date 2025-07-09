# src/oncoai_prototype/utils/feature_utils.py
# Time series aggregation (mean, slope, etc.)

import pandas as pd

def filter_high_coverage(df: pd.DataFrame, min_coverage: float = 0.85):
    coverage = df.notna().mean()
    high_cov_cols = coverage[coverage >= min_coverage].index.tolist()
    return df[high_cov_cols]

def compute_time_series_features(df: pd.DataFrame, groupby_col: str = "icustay_id"):
    aggs = ['mean', 'min', 'max', 'std']
    features = df.groupby(groupby_col).agg(aggs)
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    return features.reset_index()

def merge_features(cohort: pd.DataFrame, vitals: pd.DataFrame, labs: pd.DataFrame):
    merged = cohort.merge(vitals, on="icustay_id", how="left")
    merged = merged.merge(labs, on="icustay_id", how="left")
    return merged

def filter_and_impute(df: pd.DataFrame, min_col_coverage=0.8):
    coverage = df.notna().mean()
    df = df.loc[:, coverage > min_col_coverage]
    df = df.dropna()
    return df
