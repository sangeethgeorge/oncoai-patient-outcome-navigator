# streamlit_app/

`onco_dashboard.py` is the interactive risk dashboard ([live app](https://oncoai-db.streamlit.app)).

```bash
poetry run streamlit run streamlit_app/onco_dashboard.py
```

| `ONCOAI_MODE` | Model source |
| :-- | :-- |
| `github` (default) | Downloads `models/model.pkl`, `scaler.pkl` and `feature_names.txt` from this repo's `main` branch. Used on Streamlit Cloud. |
| `mlflow` | Loads the latest `OncoAICancerMortalityPredictor` and `onco_scaler` from the local MLflow registry. Use it to test a new model before pushing. |

The input widgets come from `models/feature_names.txt` and `models/feature_ranges.json`, and the model card from `models/metrics.json`.
All five files are loaded together as one release. The app re-checks the published release every
5 minutes (a hash of `feature_names.txt` + `metrics.json`) and reloads when it changes, so a pushed model goes live without a reboot.
If the files are ever out of sync (a model feature without an input range), the app refuses to predict instead of guessing values.
SHAP contributions are relative to the training-set mean.
