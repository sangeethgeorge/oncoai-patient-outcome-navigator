# data/

Local working data. **Everything in this folder except this README is git-ignored and must never be committed.**
MIMIC-III is credentialed data under the PhysioNet Data Use Agreement, which forbids redistribution, including derived row-level files.

| Folder | Contents | Produced by |
| :-- | :-- | :-- |
| `raw/mimic-iii-full/` | MIMIC-III v1.4 CSVs, as downloaded from PhysioNet (`<TABLE>.csv/<TABLE>.csv`) | You (PhysioNet download) |
| `processed/` | `all_labs_48h.parquet`, `all_vitals_48h.parquet`, `onco_features_cleaned.parquet` (one row per ICU stay) | `python -m oncoai_prototype.data_processing.feature_engineering` |
| `external/` | Reference papers and other non-MIMIC material | You |
