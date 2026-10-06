# reports/

Generated outputs. `reports/shap_plots/` is git-ignored: per-patient SHAP waterfall plots are derived from MIMIC data and must not be committed.

- `shap_plots/inference/`: plots from `python -m oncoai_prototype.modeling.predict`

Aggregate results that are safe to publish (held-out metrics, cohort size) go in `models/metrics.json` and the top-level README.
