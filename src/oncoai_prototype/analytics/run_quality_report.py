"""Build reports/quality_measures.md and its figures from the dbt marts.

Run after `dbt build` (see README). Reads the row-level DuckDB warehouse read-only and writes
aggregates only; every table that carries counts goes through suppress_small_cells.
"""
import os
from datetime import date

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oncoai_prototype.analytics import quality_measures as qm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DUCKDB_PATH = os.getenv("ONCOAI_DUCKDB_PATH", os.path.join(PROJECT_ROOT, "data", "processed", "oncoai.duckdb"))
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "quality_measures.md")
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

INK, MUTED, SURFACE, SERIES, REFERENCE = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#8a8984"
GROUP_ORDER = ["hematologic", "solid_metastatic", "solid_nonmetastatic"]


def load_marts(path=DUCKDB_PATH) -> dict:
    with duckdb.connect(path, read_only=True) as con:
        return {
            "stays": con.sql("SELECT * FROM fct_stay_outcomes").df(),
            "coverage": con.sql("SELECT * FROM dq_measurement_coverage").df(),
            "items": con.sql("SELECT * FROM dim_measurement_item").df(),
            "features": _model_features(),
        }


def _model_features(path=os.path.join(PROJECT_ROOT, "models", "feature_names.txt")) -> list:
    try:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


def to_markdown(df: pd.DataFrame, formats: dict | None = None) -> str:
    """Small markdown table writer; suppressed cells render as '<11'."""
    formats = formats or {}

    def cell(col, v):
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return "<11" if col not in ("ci_low", "ci_high") else "–"
        if col in formats:
            return formats[col].format(v)
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(c, r[c]) for c in df.columns) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


COUNT = "{:.0f}"
PCT = "{:.1%}"
DAYS = "{:.1f}"
CAL_FORMATS = {"n": COUNT, "observed": COUNT, "predicted": DAYS, "observed_rate": PCT, "predicted_rate": PCT,
               "obs_pred_ratio": "{:.2f}", "ci_low": "{:.2f}", "ci_high": "{:.2f}"}
OUTCOME_FORMATS = {"n": COUNT, "deaths_30d": COUNT, "mortality_30d_rate": PCT, "icu_deaths": COUNT,
                   "icu_los_survivors": DAYS, "icu_los_decedents": DAYS, "hosp_los_survivors": DAYS,
                   "hosp_los_decedents": DAYS, "icu_survivors": COUNT, "readmit_48h_rate": PCT}


def calibration_table(stays: pd.DataFrame, by: str) -> pd.DataFrame:
    t = qm.calibration_by_group(stays, by).sort_values("n", ascending=False)
    t = qm.suppress_small_cells(t, count_cols=("observed",))
    t.loc[t["suppressed"], "calibration"] = "suppressed (small cell)"
    return t[[by, "n", "observed", "predicted", "observed_rate", "predicted_rate", "obs_pred_ratio",
              "ci_low", "ci_high", "calibration"]]


def outcome_table(stays: pd.DataFrame, by: str) -> pd.DataFrame:
    t = qm.outcome_summary(stays, by)
    t = qm.suppress_small_cells(t, count_cols=("deaths_30d", "icu_deaths"))
    t = qm.suppress_small_cells(t, count_cols=(("readmits_48h", "icu_survivors"),),
                                measure_cols=["readmits_48h", "readmit_48h_rate"])
    return t[[by, "n", "deaths_30d", "mortality_30d_rate", "icu_deaths", "icu_los_survivors", "icu_los_decedents",
              "hosp_los_survivors", "hosp_los_decedents", "icu_survivors", "readmit_48h_rate"]]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def decile_calibration_plot(deciles: pd.DataFrame, path: str):
    shown = deciles.dropna(subset=["observed_rate"])
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=SURFACE)
    top = min(1.0, max(shown["observed_rate"].max(), shown["mean_predicted"].max()) * 1.1)
    ax.plot([0, top], [0, top], "--", color=REFERENCE, lw=1.2, label="Perfect calibration")
    ax.plot(shown["mean_predicted"], shown["observed_rate"], "-o", color=SERIES, lw=2, ms=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="Decile of predicted risk")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Mean predicted 30-day mortality", color=MUTED)
    ax.set_ylabel("Observed 30-day mortality", color=MUTED)
    ax.set_title("Calibration by decile of predicted risk", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def subgroup_forest_plot(tables: dict, path: str):
    """Observed/predicted with 95% CI per subgroup; one panel per grouping, shared x-axis."""
    rows = [(title, t.dropna(subset=["obs_pred_ratio"])) for title, t in tables.items()]
    heights = [max(len(t), 1) for _, t in rows]
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.5, 0.45 * sum(heights) + 1.8), facecolor=SURFACE,
                             sharex=True, gridspec_kw={"height_ratios": heights})
    for ax, (title, t) in zip(np.atleast_1d(axes), rows):
        label_col = t.columns[0]
        y = np.arange(len(t))[::-1]
        ax.axvline(1, color=REFERENCE, lw=1, ls="--")
        ax.hlines(y, t["ci_low"], t["ci_high"], color=SERIES, lw=2)
        ax.scatter(t["obs_pred_ratio"], y, s=60, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_yticks(y, [f"{str(v).replace('_', ' ')} (n={n:.0f})" for v, n in zip(t[label_col], t["n"])],
                      fontsize=8, color=INK)
        ax.set_ylim(-0.7, len(t) - 0.3)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        _style(ax)
    np.atleast_1d(axes)[-1].set_xlabel("Observed / predicted deaths (95% CI)", color=MUTED)
    fig.text(0.01, 0.01, "Model calibration check, not a quality comparison: the model does not adjust for case mix.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_report(marts: dict) -> tuple[str, dict]:
    stays = marts["stays"].copy()
    stays["cancer_group"] = pd.Categorical(stays["cancer_group"], categories=GROUP_ORDER, ordered=True)
    for c in ("icu_death", "icu_readmit_48h", "icu_readmit_same_adm"):
        stays[c] = stays[c].astype(float)

    cuts = qm.tier_cutpoints(stays["pred_prob"])
    stays["risk_tier"] = qm.assign_risk_tiers(stays["pred_prob"], cuts).set_axis(stays.index)
    tiers = qm.suppress_small_cells(qm.tier_summary(stays), count_cols=("deaths",))
    deciles = qm.suppress_small_cells(qm.calibration_by_decile(stays), count_cols=("observed",))

    by_group = outcome_table(stays, "cancer_group")
    by_unit = outcome_table(stays, "first_careunit").sort_values("n", ascending=False)
    by_heme = outcome_table(stays[stays["cancer_group"] == "hematologic"], "heme_subtype").sort_values("n", ascending=False)
    cal = {"Cancer group": calibration_table(stays, "cancer_group"),
           "Primary site": calibration_table(stays, "primary_site"),
           "ICU type": calibration_table(stays, "first_careunit")}

    flags = marts["coverage"].query("flag_coverage_gap or flag_implausible_values").sort_values(
        ["slice_type", "measurement", "slice"])
    # Coverage reveals measured and unmeasured counts, so both must clear the cell-size rule
    unmeasured = flags["n_stays"] - flags["n_stays_measured"]
    publishable = (flags["n_stays_measured"] >= qm.MIN_CELL) & ((unmeasured == 0) | (unmeasured >= qm.MIN_CELL))
    n_flags_suppressed = int((~publishable).sum())
    flags = flags.loc[publishable, ["slice_type", "slice", "measurement", "n_stays", "coverage", "overall_coverage",
                                    "pct_implausible"]]

    items = marts["items"]
    multi_unit = items[items["units"].fillna("").str.lower().str.split(", ").map(lambda u: len(set(u)) > 1)]
    labs, charts = items[items["source"] == "lab"], items[items["source"] == "chart"]
    eligible = labs[labs["pct_stays"] >= 0.70]

    features = marts.get("features", [])
    model_inputs = ("first-48 h labs and vitals" + (" plus age" if "age" in features else "")
                    + (f" ({len(features)} features, listed in models/feature_names.txt)" if features else ""))
    n, deaths = len(stays), int(stays["mortality_30d"].sum())
    n_both = int(stays["hematologic_and_solid_flag"].sum())
    n_both_txt = str(n_both) if n_both == 0 or n_both >= qm.MIN_CELL else "<11"

    multi_unit = multi_unit.assign(n_stays=multi_unit["n_stays"].where(multi_unit["n_stays"] >= qm.MIN_CELL))
    multi_unit_md = (to_markdown(multi_unit[["source", "itemid", "label", "units", "n_stays"]],
                                 {"itemid": COUNT, "n_stays": COUNT}) if len(multi_unit) else "None.")
    flags_md = (to_markdown(flags, {"n_stays": COUNT, "coverage": "{:.0%}", "overall_coverage": "{:.0%}",
                                    "pct_implausible": "{:.2%}"}) if len(flags) else "None.")

    md = f"""# Outcomes, utilization and risk stratification: oncology ICU cohort

Generated {date.today().isoformat()} by `python -m oncoai_prototype.analytics.run_quality_report` from the dbt
marts. Aggregates only; any count between 1 and 10 (and any rate that would reveal it) is suppressed (`<11`).
Definitions are in [docs/business_rules.md](../docs/business_rules.md).

**Cohort:** {n:,} first ICU stays of at least 48 h in adults with a malignancy;
{deaths:,} died within 30 days of ICU admission ({deaths / n:.1%}).

## 1. Outcomes and utilization by cancer group

Hematologic: any lymphoma, leukemia or myeloma code ({n_both_txt} of these also carry a solid-tumor code). Solid
metastatic: secondary or disseminated disease coded (Charlson definition). Length of stay is split by survival to
hospital discharge, because death shortens a stay. ICU readmission is the SCCM indicator: return to an ICU within 48 h
in the same hospitalization, among patients discharged alive from the ICU.

{to_markdown(by_group, OUTCOME_FORMATS)}

### Hematologic subtypes

Acute leukemia (any acute leukemia code), then lymphoma, myeloma and chronic leukemia, first match wins.

{to_markdown(by_heme[["heme_subtype", "n", "deaths_30d", "mortality_30d_rate", "icu_deaths"]], OUTCOME_FORMATS)}

### By ICU type

{to_markdown(by_unit, OUTCOME_FORMATS)}

These are observed rates. They are **not risk-adjusted**: units differ in who they admit (for example, medical vs
post-operative patients), so differences between rows describe case mix as much as care.

## 2. Risk stratification

Tiers are tertiles of cross-fitted predicted risk (low < {cuts[0]:.1%} ≤ medium < {cuts[1]:.1%} ≤ high). They are
descriptive. A tier meant to trigger an action, such as a goals-of-care conversation, needs a threshold agreed with
clinicians.

{to_markdown(tiers[["risk_tier", "n", "deaths", "observed_rate", "mean_predicted"]],
             {"n": COUNT, "deaths": COUNT, "observed_rate": PCT, "mean_predicted": PCT})}

![Calibration by decile](figures/calibration_deciles.png)

## 3. Model calibration by subgroup

Observed ÷ predicted deaths, with Byar's 95% CI. **This checks the model; it is not a standardized mortality ratio.**
The model's inputs are {model_inputs}. It has no admission type, comorbidity, cancer group or stage, or code status.
A ratio above 1 means the model under-predicts for that group, usually because something it can't see (such as metastatic
disease) carries risk.

### By cancer group

{to_markdown(cal["Cancer group"], CAL_FORMATS)}

### By primary site

{to_markdown(cal["Primary site"], CAL_FORMATS)}

### By ICU type

{to_markdown(cal["ICU type"], CAL_FORMATS)}

![Calibration by subgroup](figures/calibration_subgroups.png)

## 4. Data health

**LOINC coverage:** {labs['loinc_code'].notna().mean():.0%} of the {len(labs)} lab items in the 48 h window carry a
LOINC code from MIMIC's lab dictionary, including {eligible['loinc_code'].notna().mean():.0%} of the {len(eligible)} labs
recorded in ≥ 70% of stays. MIMIC-III chart items have no LOINC; a verified crosswalk covers the
{charts['loinc_code'].notna().sum()} chart items the model can use (of {len(charts)} seen).

**Items recorded in more than one unit** (ignoring case): {len(multi_unit)}.

{multi_unit_md}

**Monitor flags** (coverage ≥ 15 points from the cohort rate, or > 1% implausible values):

{flags_md}

{f"{n_flags_suppressed} further flag(s) involve fewer than 11 stays and are not shown; triage notes are in docs/business_rules.md §5." if n_flags_suppressed else ""}
"""
    return md, {"n": n, "deaths": deaths, "cuts": cuts, "deciles": deciles, "calibration": cal,
                "by_group": by_group, "by_unit": by_unit, "by_heme": by_heme, "tiers": tiers}


if __name__ == "__main__":
    md, summary = build_report(load_marts())
    os.makedirs(FIG_DIR, exist_ok=True)
    decile_calibration_plot(summary["deciles"], os.path.join(FIG_DIR, "calibration_deciles.png"))
    subgroup_forest_plot({"Cancer group": summary["calibration"]["Cancer group"],
                          "ICU type": summary["calibration"]["ICU type"]},
                         os.path.join(FIG_DIR, "calibration_subgroups.png"))
    with open(REPORT_PATH, "w") as f:
        f.write(md)
    print(f"Wrote {REPORT_PATH} ({summary['n']} stays, {summary['deaths']} deaths)")
