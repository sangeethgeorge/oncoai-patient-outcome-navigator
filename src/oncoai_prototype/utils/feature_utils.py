# src/oncoai_prototype/utils/feature_utils.py
# Time series aggregation (mean, slope, etc.)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


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
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    def featurize(group):
        # Your featurize function already handles dropping these,
        # but including include_groups=False in apply is better
        # for future compatibility and silencing the warning.
        # group = group.drop(columns=[icu_id_col, label_col], errors='ignore') # This line is now less critical but harmless

        times = (group[time_col] - group[time_col].min()).dt.total_seconds().values / 3600.0
        values = group[value_col].values

        if len(values) == 0:
            return pd.Series({'mean': np.nan, 'min': np.nan, 'max': np.nan, 'slope': np.nan})

        stats = {
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
        }

        if len(values) > 1:
            model = LinearRegression().fit(times.reshape(-1, 1), values)
            stats['slope'] = model.coef_[0]
        else:
            stats['slope'] = np.nan

        return pd.Series(stats)

    grouped = df.groupby([icu_id_col, label_col])
    # FIX: Add 'include_groups=False' to silence the FutureWarning
    long_df = grouped.apply(featurize, include_groups=False).reset_index() 

    wide_df = long_df.pivot(index=icu_id_col, columns=label_col)
    wide_df.columns = [f"{stat}_{label}".lower().replace(" ", "_") for stat, label in wide_df.columns]
    wide_df.reset_index(inplace=True)

    return wide_df


def merge_features(cohort: pd.DataFrame, vitals: pd.DataFrame, labs: pd.DataFrame):
    merged = cohort.merge(vitals, on="icustay_id", how="left")
    merged = merged.merge(labs, on="icustay_id", how="left")
    return merged

def filter_and_impute(df: pd.DataFrame, min_col_coverage=0.8):
    coverage = df.notna().mean()
    df = df.loc[:, coverage > min_col_coverage]
    df = df.dropna()
    return df
