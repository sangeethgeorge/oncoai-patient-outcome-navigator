"""Outcome, utilization and risk-stratification summaries over one-row-per-stay data.

Pure functions: they take DataFrames/arrays and return aggregates. Predicted risk comes from
cross-fitted (out-of-fold) probabilities, so no stay is scored by a model that saw it.

The predicted risk comes from a model of first-48 h labs and vitals (plus age when feature selection
keeps it), not a case-mix model (no admission type, comorbidity, cancer group or code status), so
observed/predicted ratios here check the model's calibration in subgroups. They are not standardized mortality ratios and say nothing about quality of care.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

TIER_LABELS = ["low", "medium", "high"]
MIN_CELL = 11  # CMS cell-size suppression: no published count between 1 and 10


def tier_cutpoints(pred_prob, quantiles=(1 / 3, 2 / 3)) -> list:
    """Cut-points at predicted-risk tertiles, rounded so they can be written into the report."""
    return [round(float(q), 3) for q in np.quantile(pred_prob, quantiles)]


def assign_risk_tiers(pred_prob, cuts) -> pd.Categorical:
    """low: p < cuts[0]; medium: cuts[0] <= p < cuts[1]; high: p >= cuts[1]."""
    bins = [-np.inf, *cuts, np.inf]
    return pd.cut(pd.Series(pred_prob), bins=bins, labels=TIER_LABELS, right=False)


def byar_ci(observed, expected, alpha=0.05):
    """Byar's approximation to the exact Poisson CI for an observed/expected ratio."""
    o = np.asarray(observed, dtype=float)
    e = np.asarray(expected, dtype=float)
    z = norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.where(o > 0, o * (1 - 1 / (9 * o) - z / (3 * np.sqrt(o))) ** 3, 0.0) / e
        o1 = o + 1
        upper = o1 * (1 - 1 / (9 * o1) + z / (3 * np.sqrt(o1))) ** 3 / e
    return lower, upper


def calibration_by_group(df: pd.DataFrame, by: str, outcome="mortality_30d", pred="pred_prob") -> pd.DataFrame:
    """Per group: stays, observed and predicted deaths, observed/predicted with a 95% CI, and whether the
    model under- or over-predicts there (CI excludes 1)."""
    g = df.groupby(by, observed=True).agg(n=(outcome, "size"), observed=(outcome, "sum"), predicted=(pred, "sum"))
    g["observed_rate"] = g["observed"] / g["n"]
    g["predicted_rate"] = g["predicted"] / g["n"]
    g["obs_pred_ratio"] = g["observed"] / g["predicted"]
    g["ci_low"], g["ci_high"] = byar_ci(g["observed"], g["predicted"])
    g["calibration"] = np.select([g["ci_low"] > 1, g["ci_high"] < 1], ["model under-predicts", "model over-predicts"],
                                 default="consistent")
    return g.reset_index()


def calibration_by_decile(df: pd.DataFrame, outcome="mortality_30d", pred="pred_prob", bins=10) -> pd.DataFrame:
    """Observed vs mean predicted risk in equal-size bins of predicted risk."""
    d = df.assign(risk_decile=pd.qcut(df[pred].rank(method="first"), bins, labels=range(1, bins + 1)))
    return d.groupby("risk_decile", observed=True).agg(
        n=(outcome, "size"), observed=(outcome, "sum"), observed_rate=(outcome, "mean"),
        mean_predicted=(pred, "mean")).reset_index()


def tier_summary(df: pd.DataFrame, tier_col="risk_tier", outcome="mortality_30d", pred="pred_prob") -> pd.DataFrame:
    """Observed vs predicted mortality by risk tier."""
    return df.groupby(tier_col, observed=True).agg(
        n=(outcome, "size"), deaths=(outcome, "sum"), observed_rate=(outcome, "mean"),
        mean_predicted=(pred, "mean")).reset_index()


def outcome_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Observed outcomes and utilization per group. Length of stay is split by hospital survival
    (death shortens a stay); ICU readmission within 48 h is over ICU survivors only."""
    alive = df["hospital_expire_flag"] == 0
    survivors = df[~df["icu_death"].astype(bool)]
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "deaths_30d": g["mortality_30d"].sum(),
        "icu_deaths": g["icu_death"].sum(),
        "icu_los_survivors": df[alive].groupby(by, observed=True)["icu_los_days"].median(),
        "icu_los_decedents": df[~alive].groupby(by, observed=True)["icu_los_days"].median(),
        "hosp_los_survivors": df[alive].groupby(by, observed=True)["hosp_los_days"].median(),
        "hosp_los_decedents": df[~alive].groupby(by, observed=True)["hosp_los_days"].median(),
        "icu_survivors": survivors.groupby(by, observed=True).size(),
        "readmits_48h": survivors.groupby(by, observed=True)["icu_readmit_48h"].sum(),
    })
    out["mortality_30d_rate"] = out["deaths_30d"] / out["n"]
    out["readmit_48h_rate"] = out["readmits_48h"] / out["icu_survivors"]
    return out.reset_index()


def suppress_small_cells(table: pd.DataFrame, n_col="n", count_cols=("observed",), min_n=MIN_CELL,
                         measure_cols=None) -> pd.DataFrame:
    """Blank the measures of any row whose stay count, event count or non-event count is between 1 and
    min_n - 1; a rate published next to n would otherwise reveal the small count. count_cols may be
    (count, denominator) pairs when an event is counted over a subset, e.g. readmissions over ICU survivors.
    measure_cols limits the blanking to those columns (default: every numeric column)."""
    out = table.copy()
    small = (out[n_col] > 0) & (out[n_col] < min_n)
    for c in count_cols:
        events, denom = (out[c[0]], out[c[1]]) if isinstance(c, tuple) else (out[c], out[n_col])
        small |= ((events > 0) & (events < min_n)) | ((denom - events > 0) & (denom - events < min_n))
    if measure_cols is None:
        measure_cols = [c for c in out.columns
                        if c != "suppressed" and pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c])]
    out[measure_cols] = out[measure_cols].astype(float)
    out.loc[small, measure_cols] = np.nan
    out["suppressed"] = out.get("suppressed", False) | small
    return out
