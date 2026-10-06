# src/oncoai_prototype

| Module | Purpose |
| :-- | :-- |
| `data_loading/01_mimic-iii_dataload.sql` | Load MIMIC-III v1.4 CSVs into Postgres (`psql -v mimic_dir=<absolute path>`) |
| `data_loading/02_define_onco_cohort.sql` | `oncology_icu_base`: one row per first ICU stay ≥ 48 h in adult cancer patients, with the 30-day mortality label |
| `data_loading/03a/03b_extract_*_48h.sql` | `all_labs_48h` / `all_vitals_48h`: measurements in `[intime, intime + 48h)` |
| `data_processing/feature_engineering.py` | Read the views through DuckDB; build mean/min/max/hourly-slope features; write `data/processed/onco_features_cleaned.parquet` |
| `modeling/model_training.py` | Patient-grouped split, train-only SHAP feature selection, logistic regression, metrics with bootstrap CI, MLflow logging, export to `models/` |
| `modeling/predict.py` | Batch inference and SHAP plots from the registered MLflow model |
| `utils/leakage.py` | Guards: duplicate-ID and train/test group-overlap assertions, target-named column check |
| `utils/preprocessing.py` | Grouped stratified split, train-median imputation, scaling |
| `utils/feature_utils.py` | Coverage filters, vectorized time-series features, one-to-one merges |
| `utils/db_utils.py` | DuckDB ↔ Postgres connections, read-only SQL helper, password redaction for error messages |
| `utils/shap_utils.py` | Beeswarm and waterfall plot generation |
| `utils/io_utils.py` | Parquet and model I/O |

Pipeline order: SQL 01 → 02 → 03a/03b → `feature_engineering` → `model_training`. See the top-level README for the commands.
