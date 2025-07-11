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