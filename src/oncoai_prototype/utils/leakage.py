# src/oncoai_prototype/utils/leakage.py
# Feature leakage detection utilities

import pandas as pd

def check_for_leakage(df: pd.DataFrame, target_col: str = "mortality_30d") -> pd.DataFrame:
    leaks = [
        col for col in df.columns
        if target_col.lower() in col.lower() and col != target_col
    ]
    if leaks:
        print(f"⚠️ Potential data leakage in columns: {leaks}")
        df = df.drop(columns=leaks)
    else:
        print("✅ No significant leakage detected.")
    return df


def assert_unique_ids(df: pd.DataFrame, id_col: str = "icustay_id"):
    """Duplicate rows of the same unit leak across train/test splits."""
    n_dupes = df[id_col].duplicated().sum()
    if n_dupes:
        raise ValueError(f"{n_dupes} duplicate rows on '{id_col}'; expected one row per {id_col}.")


def assert_no_group_overlap(train_groups, test_groups, group_col: str = "subject_id"):
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise ValueError(f"{len(overlap)} {group_col} values appear in both train and test.")
