# src/etl_feature_engineering.py

import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def connect_to_postgres():
    duckdb.sql("INSTALL postgres_scanner;")
    duckdb.sql("LOAD postgres_scanner;")

    pg_conn_str = "dbname='mimic-iii' user='sangeethgeorge' password='12345' host='localhost' port='5432'"
    return pg_conn_str

def load_data():
    pg_conn = connect_to_postgres()

    queries = {
        'labs': f"""
            SELECT *
            FROM postgres_scan('{pg_conn}', 'public', 'all_labs_48h')
        """,
        'vitals': f"""
            SELECT *
            FROM postgres_scan('{pg_conn}', 'public', 'all_vitals_48h')
        """,
        'cohort': f"""
            SELECT *
            FROM postgres_scan('{pg_conn}', 'public', 'oncology_icu_base')
        """
    }

    return {
        'labs': duckdb.sql(queries['labs']).df(),
        'vitals': duckdb.sql(queries['vitals']).df(),
        'cohort': duckdb.sql(queries['cohort']).df(),
    }

def filter_high_coverage(df, label_col, value_col, threshold=0.7):
    total_unique_stays = df['icustay_id'].nunique()
    coverage = df.groupby(label_col)['icustay_id'].nunique() / total_unique_stays
    high_coverage_labels = coverage[coverage >= threshold].index.tolist()
    df_filtered = df[df[label_col].isin(high_coverage_labels)].copy()
    return df_filtered

def compute_time_series_features(df, group_cols, value_col):
    df['charttime'] = pd.to_datetime(df['charttime'])

    def _features(group):
        times = (group['charttime'] - group['charttime'].min()).dt.total_seconds().values / 3600.0
        values = group[value_col].values

        if len(values) < 1:
            return pd.Series({'mean': np.nan, 'min': np.nan, 'max': np.nan, 'slope': np.nan})

        feats = {
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
        }

        if len(values) > 1:
            model = LinearRegression().fit(times.reshape(-1, 1), values)
            feats['slope'] = model.coef_[0]
        else:
            feats['slope'] = np.nan

        return pd.Series(feats)

    features_df = df.groupby(group_cols).apply(_features, include_groups=False).reset_index()
    wide_df = features_df.pivot(index='icustay_id', columns=group_cols[-1])
    wide_df.columns = [f"{stat}_{label}".lower().replace(' ', '_') for stat, label in wide_df.columns]
    wide_df.reset_index(inplace=True)
    return wide_df

def merge_features(cohort_df, vitals_df, labs_df):
    merged = cohort_df.merge(vitals_df, on='icustay_id', how='left')
    merged = merged.merge(labs_df, on='icustay_id', how='left')
    return merged

def filter_and_impute(df, label_col='mortality_30d', completeness_threshold=0.9):
    completeness = df.count() / len(df)
    high_completeness_cols = completeness[completeness >= completeness_threshold].index.tolist()
    df_filtered = df[high_completeness_cols]

    min_non_nan = int(len(high_completeness_cols) * completeness_threshold)
    df_filtered = df_filtered.dropna(thresh=min_non_nan).copy()

    numeric_cols = df_filtered.select_dtypes(include='number').columns
    if label_col in numeric_cols:
        numeric_cols = numeric_cols.drop(label_col)
    df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(df_filtered[numeric_cols].median())

    return df_filtered

def build_onco_cohort():
    data = load_data()
    vitals = filter_high_coverage(data['vitals'], 'vitals_label', 'vitals_valuenum')
    labs = filter_high_coverage(data['labs'], 'labs_label', 'labs_valuenum')

    vitals_features = compute_time_series_features(vitals, ['icustay_id', 'vitals_label'], 'vitals_valuenum')
    labs_features = compute_time_series_features(labs, ['icustay_id', 'labs_label'], 'labs_valuenum')

    full_df = merge_features(data['cohort'], vitals_features, labs_features)
    clean_df = filter_and_impute(full_df)

    return clean_df

if __name__ == "__main__":
    onco_cohort_ML = build_onco_cohort()
    print("✅ Final shape:", onco_cohort_ML.shape)
    onco_cohort_ML.to_parquet("data/onco_cohort_ML.parquet", index=False)
    print("✅ Saved cleaned ML-ready cohort to data/onco_cohort_ML.parquet")
