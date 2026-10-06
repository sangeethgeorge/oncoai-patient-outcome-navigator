# OncoAI Patient Outcome Navigator

🧠 Early 30-day mortality risk for critically ill cancer patients from the first 48 hours of ICU labs and vitals (MIMIC-III)

🔗 **Try the Streamlit App:** [oncoai-db.streamlit.app](https://oncoai-db.streamlit.app)

⚠️ **For research and educational use only.** Not for clinical decision-making.

---

## 🔍 Overview

Critically ill cancer patients face high ICU mortality, yet early risk signals are scattered across fragmented EHR data. OncoAI is a research prototype that:

* Defines an oncology ICU cohort from ICD-9 codes in MIMIC-III v1.4
* Summarizes the first 48 h of ICU labs and vitals into time-series features (mean, min, max, hourly slope)
* Trains an interpretable 30-day mortality classifier with leakage-safe evaluation
* Explains each prediction with SHAP in an interactive Streamlit dashboard

---

## 🧱 Project Architecture

```text
 MIMIC-III v1.4 CSVs
        │  01_mimic-iii_dataload.sql
        ▼
 PostgreSQL ──► 02_define_onco_cohort.sql      one row per ICU stay
        │       03a/03b_extract_*_48h.sql       labs + vitals in [intime, intime+48h)
        │  DuckDB postgres_scanner
        ▼
 feature_engineering.py ──► data/processed/onco_features_cleaned.parquet   (local only)
        │
        ▼
 model_training.py   patient-grouped split → train-only SHAP feature selection
        │            → logistic regression → metrics + MLflow run
        ▼
 models/  model.pkl · scaler.pkl · feature_names.txt · feature_ranges.json · metrics.json
        │
        ▼
 streamlit_app/onco_dashboard.py   risk estimate + SHAP explanation + model card
```

---

## 📊 Project Status (October 2026)

| Module              | Status        | Notes |
| :------------------ | :------------ | :---- |
| Cohort definition   | ✅ Complete   | First ICU stay ≥ 48 h per patient; one row per stay, enforced by a unique index |
| ETL                 | ✅ Complete   | SQL views for the 48 h lab/vital windows; DuckDB reads them from Postgres |
| Feature engineering | ✅ Complete   | Mean/min/max/hourly slope for 25 labs plus heart rate, respiratory rate and hemoglobin |
| Modeling            | ✅ Complete   | Logistic regression; patient-grouped split, train-only feature selection, MLflow tracking |
| SHAP explanations   | ✅ Complete   | Global (beeswarm) and per-patient (waterfall) plots |
| Streamlit app       | ✅ Live       | [oncoai-db.streamlit.app](https://oncoai-db.streamlit.app) |
| Tests               | ✅ Complete   | Unit tests on synthetic data; `-m db` integration tests against the local database |
| ICU-note NLP / LLM summaries | 💡 Planned | Not yet implemented |
| Survival analysis (R)       | 💡 Planned | Not yet implemented |
| Docker                      | 💡 Planned | |

---

## 📁 Data Source

* **Dataset:** [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/) (credentialed access via PhysioNet)
* **Tables:** `patients`, `admissions`, `icustays`, `diagnoses_icd`, `d_icd_diagnoses`, `labevents`, `d_labitems`, `chartevents`, `d_items`
* **License:** PhysioNet Credentialed Health Data License 1.5.0

> **No MIMIC data is distributed in this repository**, raw or derived. `data/`, `mlruns/`, `reports/shap_plots/`
> and notebooks are git-ignored. To reproduce, you need your own PhysioNet credentials and a local copy of MIMIC-III.
> The published `models/` files contain only model coefficients and aggregate feature ranges.

### Cohort definition

| Criterion | Rule |
| :-- | :-- |
| Population | First ICU stay per patient, in an admission with a malignant/neoplasm ICD-9 code (140–239) |
| Observation window | First 48 h after ICU admission (`[intime, intime + 48h)`) |
| Inclusion | ICU stay ≥ 48 h, so the outcome can't occur inside the feature window; age 18–89 |
| Outcome | Death within 30 days of ICU admission (`dod <= intime + 30 days`) |
| Unit | One row per ICU stay (enforced by a unique index) |

**Cohort size:** 2,674 ICU stays (2,674 patients), 672 deaths within 30 days (25.1%).

### Model performance

Held-out test set of 535 ICU stays (20%, split by patient, stratified). Features were chosen
on the training split only (top 10 by mean |SHAP| of an XGBoost model), then fed to a
standardized logistic regression.

| Metric | Model (10 lab/vital features) | Baseline (age only) |
| :-- | :-- | :-- |
| ROC-AUC (95% bootstrap CI) | **0.764** (0.718–0.807) | 0.583 (0.532–0.635) |
| PR-AUC (prevalence 0.251) | 0.595 | 0.366 |
| Brier score | 0.172 | 0.208 |
| Calibration slope | 1.03 | 1.88 |
| 5-fold grouped CV ROC-AUC (training split) | 0.741 ± 0.038 | — |

Selected features: min/mean BUN, mean MCHC, bicarbonate and glucose slopes, mean anion gap,
max heart rate, min RDW, mean respiratory rate, min creatinine. ICD-derived predictors are excluded
because MIMIC assigns ICD codes at discharge, after the prediction time.

> **About the earlier 0.842 figure:** earlier versions of this project reported ROC-AUC 0.842. That number
> was inflated by ICU stays duplicated across train and test, feature selection that used test rows, and a
> cohort (stays ≤ 48 h) whose outcome could fall inside the feature window. It has been retired.
> See [What changed in October 2026](#-what-changed-in-october-2026).

---

## 📈 Dashboard Features

| Feature                     | Description |
| :-------------------------- | :---------- |
| Feature input form          | Enter the model's 48 h lab/vital summaries (ranges from the training set) |
| 30-day mortality prediction | Logistic regression risk estimate |
| SHAP explanation            | Waterfall plot and table of per-feature contributions vs. the training mean |
| Model card                  | Held-out metrics, baseline comparison and cohort size |

---

## ⚙️ Tech Stack

| Layer      | Stack |
| :--------- | :---- |
| Data       | PostgreSQL 17, SQL, DuckDB (`postgres_scanner`), pandas, PyArrow |
| ML         | scikit-learn, XGBoost (feature ranking), SHAP, MLflow |
| UI         | Streamlit, Matplotlib |
| Dev        | Poetry, pytest, GitHub |

---

## 🗂 Repository Structure

```text
src/oncoai_prototype/
  data_loading/      SQL: MIMIC load, cohort definition, 48 h lab/vital extraction
  data_processing/   feature_engineering.py: one feature row per ICU stay
  modeling/          model_training.py (train + evaluate + export), predict.py (batch inference)
  utils/             db, feature, preprocessing, leakage-check, SHAP and I/O helpers
streamlit_app/       onco_dashboard.py
models/              published model artifacts (see models/README.md)
tests/               unit tests (synthetic data) + `db` integration tests
data/, notebooks/, reports/   local-only working folders (contents git-ignored)
```

---

## 🚀 Setup Instructions

Requires Python 3.11, Poetry, PostgreSQL, and credentialed access to MIMIC-III v1.4.

```bash
# 1. Clone and install
git clone https://github.com/sangeethgeorge/oncoai-patient-outcome-navigator.git
cd oncoai-patient-outcome-navigator
poetry install

# 2. Configure the database connection (never commit .env)
cp .env.example .env   # then fill in ONCOAI_POSTGRES_CONN_STR (a read-only role is enough)

# 3. Load the MIMIC-III CSVs into Postgres. Run as a superuser: server-side COPY needs
#    an absolute path, and the Postgres server must be able to read the files.
psql -U postgres -d mimic-iii -v mimic_dir="$PWD/data/raw/mimic-iii-full" \
     -f src/oncoai_prototype/data_loading/01_mimic-iii_dataload.sql

# 4. Build the cohort and the 48 h extraction views, then let the app role read them
for f in 02_define_onco_cohort 03a_extract_all_labs_48h 03b_extract_all_vitals_48h; do
  psql -U postgres -d mimic-iii -f src/oncoai_prototype/data_loading/$f.sql
done
psql -U postgres -d mimic-iii -c "GRANT SELECT ON oncology_icu_base, all_labs_48h, all_vitals_48h TO <app_role>;"

# 5. Features -> training (writes models/*.pkl, feature_ranges.json, metrics.json)
poetry run python -m oncoai_prototype.data_processing.feature_engineering
poetry run python -m oncoai_prototype.modeling.model_training

# 6. Tests: unit tests need no database; `-m db` runs the integration checks
poetry run pytest -q
poetry run pytest -q -m db

# 7. Dashboard (ONCOAI_MODE=mlflow uses the local MLflow registry; the default, github,
#    downloads models/ from this repository)
ONCOAI_MODE=mlflow poetry run streamlit run streamlit_app/onco_dashboard.py
```

---

## ☁️ Deployment

The app runs on Streamlit Community Cloud from `streamlit_app/onco_dashboard.py`. Dependencies
come from `requirements.txt`, which is exported from `poetry.lock`. In the default `github` mode it
downloads the files in `models/` from this repository's `main` branch. To deploy a retrained model,
commit the regenerated `models/` files and push to `main`.

---

## 🛠 What changed in October 2026

A repository-wide review fixed several problems that had inflated the reported performance:

* **Duplicated ICU stays:** the cohort query produced one row per cancer ICD code, which turned
  1,800 stays into 3,340 rows; copies of the same stay landed in both train and test.
  It now aggregates codes per admission.
* **Vitals timestamps:** `CHARTEVENTS.CHARTTIME` was loaded as `DATE`, which put every vital at
  midnight and broke the 48 h window and the slopes. It's now loaded as `TIMESTAMP`.
* **Outcome inside the feature window:** the cohort kept stays ≤ 48 h; it now requires ≥ 48 h.
* **Selection leakage:** SHAP feature selection had used test rows. It now runs on the
  training split only, including inside each CV fold.
* **Look-ahead features:** discharge-assigned ICD counts are no longer predictors.
* **Data hygiene:** MIMIC-derived files, MLflow runs and a `.env` file were removed from git and
  from the git history.

---

## ⚠️ Limitations
* Single-center retrospective data (Beth Israel Deaconess Medical Center, 2001–2012); no external validation.
* ICD-9 codes identify cancer diagnoses, not active treatment, stage or code status.
* Feature selection and the train/test split use one random seed (42). The cross-validation spread (±0.038) shows how much results vary.
* The dashboard takes manually entered values; it is a demonstration, not a clinical tool.
* Do not use for clinical inference or decision-making.

---

## 📜 License
MIT License – see [LICENSE](LICENSE). The MIT license covers this code only. MIMIC-III data is governed by the PhysioNet Credentialed Health Data License.

## 🙏 Acknowledgements & Citation

MIMIC-III is provided by the MIT Laboratory for Computational Physiology. If you use this work, please cite:

* Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. *Scientific Data* 3, 160035 (2016). https://doi.org/10.1038/sdata.2016.35
* Johnson A, Pollard T, Mark R. MIMIC-III Clinical Database (version 1.4). *PhysioNet* (2016). https://doi.org/10.13026/C2XW26
* Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220 (2000).

Built with scikit-learn, XGBoost, SHAP, MLflow, DuckDB and Streamlit.
