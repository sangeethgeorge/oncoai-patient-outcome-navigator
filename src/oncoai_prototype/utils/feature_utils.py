# src/oncoai_prototype/utils/feature_utils.py
# Time series aggregation (mean, slope, etc.)

import re
import pandas as pd
import numpy as np


def filter_high_coverage(df: pd.DataFrame, label_col: str, group_col: str = 'icustay_id', min_coverage: float = 0.95):
    total_stays = df[group_col].nunique()
    label_coverage = df.groupby(label_col)[group_col].nunique() / total_stays
    high_cov_labels = label_coverage[label_coverage >= min_coverage].index.tolist()

    return df[df[label_col].isin(high_cov_labels)].copy()


def compute_time_series_features(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    label_col: str,
    icu_id_col: str
) -> pd.DataFrame:
    """Per stay and measurement: mean, min, max, and least-squares slope (units per hour)."""
    df = df[[icu_id_col, label_col, time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df[value_col] = df[value_col].astype(float)

    keys = [icu_id_col, label_col]
    grouped = df.groupby(keys)[value_col]
    stats = grouped.agg(['mean', 'min', 'max'])

    # Vectorized OLS slope: cov(t, v) / var(t), with t in hours. NaN when all times coincide.
    df['_t'] = (df[time_col] - df.groupby(keys)[time_col].transform('min')).dt.total_seconds() / 3600.0
    df['_dt'] = df['_t'] - df.groupby(keys)['_t'].transform('mean')
    df['_dv'] = df[value_col] - df.groupby(keys)[value_col].transform('mean')
    df['_cov'] = df['_dt'] * df['_dv']
    df['_var'] = df['_dt'] ** 2
    sums = df.groupby(keys)[['_cov', '_var']].sum()
    stats['slope'] = (sums['_cov'] / sums['_var']).where(sums['_var'] > 0)

    wide_df = stats.unstack(label_col)
    wide_df.columns = [re.sub(r"[^a-z0-9]+", "_", f"{stat}_{label}".lower()).strip("_") for stat, label in wide_df.columns]
    return wide_df.reset_index()


def merge_features(cohort: pd.DataFrame, vitals: pd.DataFrame, labs: pd.DataFrame):
    if cohort["icustay_id"].duplicated().any():
        raise ValueError("Cohort has duplicate icustay_id rows; expected one row per ICU stay.")
    merged = cohort.merge(vitals, on="icustay_id", how="left", validate="one_to_one")
    merged = merged.merge(labs, on="icustay_id", how="left", validate="one_to_one")
    return merged


def filter_low_coverage_columns(df: pd.DataFrame, feature_cols: list, min_col_coverage=0.8):
    """Drop feature columns observed in too few stays. Rows are kept; imputation
    happens in the training pipeline using training-set statistics only."""
    coverage = df[feature_cols].notna().mean()
    dropped = coverage[coverage <= min_col_coverage].index
    return df.drop(columns=dropped)
