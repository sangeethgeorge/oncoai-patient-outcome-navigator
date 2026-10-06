# models/

Published model artifacts, written by `python -m oncoai_prototype.modeling.model_training` and loaded by the Streamlit app.
They contain **no patient-level data**: only fitted coefficients, scaling parameters, and aggregate statistics.

| File | Contents |
| :-- | :-- |
| `model.pkl` | MLflow `PythonModel` wrapper around the fitted `LogisticRegression` (input: standardized features) |
| `scaler.pkl` | MLflow `PythonModel` wrapper around the fitted `StandardScaler` (training-set means and SDs) |
| `feature_names.txt` | The 10 selected features, in model input order |
| `feature_ranges.json` | Training-set 5th/50th/95th percentiles per feature (dashboard input ranges and defaults) |
| `metrics.json` | Cohort size and held-out metrics (ROC-AUC with 95% CI, PR-AUC, Brier, calibration slope, CV, age-only baseline), plus cross-fitted (`oof_*`) AUC, Brier and calibration slope |

The `.pkl` files are pickles: load them only from this repository. Retraining overwrites all five files.
Commit and push them together to update the deployed app.
