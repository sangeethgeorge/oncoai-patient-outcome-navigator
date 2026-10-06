# reports/

Generated outputs. `reports/shap_plots/` is git-ignored: per-patient SHAP waterfall plots are derived from MIMIC data and must not be committed.

- `shap_plots/inference/`: plots from `python -m oncoai_prototype.modeling.predict`
- `quality_measures.md`: risk tiers, observed/expected mortality and data-health results, from `python -m oncoai_prototype.analytics.run_quality_report`
- `figures/`: the funnel and tier-calibration plots referenced by `quality_measures.md`

Aggregate results that are safe to publish (held-out metrics, cohort size) go in `models/metrics.json` and the top-level README.
`quality_measures.md` and `figures/` are aggregate-only and are committed. Every count passes through small-cell
suppression first: a row whose stay, event or non-event count is between 1 and 10 is blanked, following the CMS cell-size convention.
