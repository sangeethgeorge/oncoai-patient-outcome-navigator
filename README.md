# OncoAI: Oncology ICU Outcomes Analytics & Risk Stratification

🧠 Cohort definition, curated data marts with data-quality tests, outcomes and utilization measures, and 30-day mortality risk stratification for critically ill cancer patients (MIMIC-III)

🔗 **Try the Streamlit App:** [oncoai-db.streamlit.app](https://oncoai-db.streamlit.app)

⚠️ **For research and educational use only.** Not for clinical decision-making.

---

## 🔍 Overview

Critically ill cancer patients face high ICU mortality, yet early risk signals are scattered across fragmented EHR data. OncoAI is a research project that:

* **Defines the cohort from standard code sets:** Charlson ICD-9-CM cancer definitions (Quan 2005), grouped into hematologic, metastatic solid and non-metastatic solid, with documented [business rules](docs/business_rules.md)
* **Curates analytics-ready tables with dbt:** staging, intermediate and mart models; LOINC-coded measurements; 40+ data tests; and a data-health monitor
* **Reports outcomes and utilization the way critical-care studies do:** 30-day mortality, length of stay split by survival, and 48 h ICU readmission among ICU survivors (the SCCM indicator), with CMS-style small-cell suppression
* **Stratifies 30-day mortality risk** from the first 48 h of labs and vitals, using a leakage-safe, patient-grouped and cross-fitted evaluation, and checks the model's calibration by subgroup
* **Explains each prediction with SHAP** in an interactive Streamlit dashboard

📄 **Start here:** [executive brief](docs/executive_brief.md) (one page, plain language) · [quality measures report](reports/quality_measures.md) · [business rules](docs/business_rules.md)

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
        │            → 5-fold cross-fitted risk for every stay ──► data/processed/oof_predictions.parquet
        ▼
 models/  model.pkl · scaler.pkl · feature_names.txt · feature_ranges.json · metrics.json
        │                                           │
        ▼                                           ▼
 streamlit_app/onco_dashboard.py         dbt/ (dbt-duckdb, Postgres attached read-only)
 risk estimate + SHAP + model card         seeds: ICD-9 cancer categories · chart-item LOINC · plausibility limits
                                           staging → intermediate → marts: fct_stay_outcomes,
                                           dim_measurement_item, dq_measurement_coverage  + 40+ tests
                                                    │
                                                    ▼
                                         analytics/run_quality_report.py ──► reports/quality_measures.md
                                         (aggregates only, cells < 11 suppressed)    + reports/figures/
```

---

## 📊 Project Status (October 2026)

| Module              | Status        | Notes |
| :------------------ | :------------ | :---- |
| Cohort definition   | ✅ Complete   | Charlson ICD-9-CM malignancy codes; first ICU stay ≥ 48 h per patient; one row per stay, enforced by a unique index |
| ETL                 | ✅ Complete   | SQL views for the 48 h lab/vital windows; DuckDB reads them from Postgres |
| Curated data layer  | ✅ Complete   | dbt-duckdb: staging/intermediate/marts, LOINC mapping, 40+ data tests, data-health monitor |
| Outcomes & utilization | ✅ Complete | Mortality by cancer group and ICU type, LOS by survival, 48 h ICU readmission ([report](reports/quality_measures.md)) |
| Feature engineering | ✅ Complete   | Mean/min/max/hourly slope for 26 labs and vital signs |
| Modeling            | ✅ Complete   | Logistic regression; patient-grouped split, train-only feature selection, MLflow tracking |
| SHAP explanations   | ✅ Complete   | Global (beeswarm) and per-patient (waterfall) plots |
| Streamlit app       | ✅ Live       | [oncoai-db.streamlit.app](https://oncoai-db.streamlit.app) |
| Tests               | ✅ Complete   | pytest unit tests on synthetic data; `-m db` integration tests; dbt data tests |
| Case-mix-adjusted SMR | 💡 Planned  | Needs a first-24 h severity model; see [business rules §6](docs/business_rules.md#why-this-is-not-a-standardized-mortality-ratio) |
| ICU-note NLP / LLM summaries | 💡 Planned | Not yet implemented |
| Survival analysis (R)       | 💡 Planned | Not yet implemented |
| Docker                      | 💡 Planned | |

---

## 📁 Data Source

* **Dataset:** [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/) (credentialed access via PhysioNet)
* **Tables:** `patients`, `admissions`, `icustays`, `diagnoses_icd`, `labevents`, `d_labitems`, `chartevents`, `d_items`
* **License:** PhysioNet Credentialed Health Data License 1.5.0

> **No MIMIC data is distributed in this repository**, raw or derived. `data/`, `mlruns/`, `reports/shap_plots/`
> the dbt DuckDB warehouse and notebooks are git-ignored. To reproduce, you need your own PhysioNet credentials and a local copy of MIMIC-III.
> The published `models/` files contain only model coefficients and aggregate feature ranges; `reports/` holds aggregates
> with every count between 1 and 10 suppressed.

### Cohort definition

| Criterion | Rule |
| :-- | :-- |
| Population | First ICU stay per patient, in an admission with a Charlson malignancy code: 140–172, 174–195.8, 200–208, metastatic 196–199.1, plus neuroendocrine 209.0–209.3 and 209.7. Non-melanoma skin (173), in situ, benign and uncertain-behavior codes are excluded ([details](docs/business_rules.md#2-icd-9-cancer-codes)) |
| Observation window | First 48 h after ICU admission (`[intime, intime + 48h)`) |
| Inclusion | ICU stay ≥ 48 h, so the outcome can't occur inside the feature window; age 18–89 |
| Outcome | Death within 30 days of ICU admission (`dod <= intime + 30 days`) |
| Unit | One row per ICU stay (enforced by a unique index) |

**Cohort size:** 2,671 ICU stays (2,671 patients), 765 deaths within 30 days (28.6%).

| Cancer group | Stays | 30-day mortality |
| :-- | --: | --: |
| Solid tumor, metastatic | 1,171 | 35.9% |
| Solid tumor, non-metastatic | 937 | 19.3% |
| Hematologic | 563 | 29.1% (acute leukemia: 40.7%) |

### Model performance

Held-out test set of 535 ICU stays (20%, split by patient, stratified). Features were chosen
on the training split only (top 10 by mean |SHAP| of an XGBoost model), then fed to a
standardized logistic regression.

| Metric | Model (10 features) | Baseline (age only) |
| :-- | :-- | :-- |
| ROC-AUC (95% bootstrap CI) | **0.768** (0.722–0.812) | 0.549 (0.494–0.604) |
| PR-AUC (prevalence 0.286) | 0.584 | 0.321 |
| Brier score | 0.164 | 0.202 |
| Calibration slope | 1.04 | 0.86 |
| 5-fold grouped CV ROC-AUC (training split) | 0.740 ± 0.019 | — |
| Cross-fitted ROC-AUC, all 2,671 stays (5 folds) | 0.752 | — |

Selected features: min/mean BUN, mean creatinine, mean/max MCHC, mean anion gap, bicarbonate slope,
min RDW, max heart rate, and age. ICD-derived predictors are excluded because MIMIC assigns ICD codes
at discharge, after the prediction time.

**Risk tiers** (tertiles of cross-fitted risk): observed 30-day mortality is 9.8% in the low tier,
26.4% in the medium tier and 49.8% in the high tier (predicted: 10.9%, 23.8%, 51.6%). The model
under-predicts for metastatic disease (observed/predicted 1.18, 95% CI 1.07–1.30), which it can't see.
See [reports/quality_measures.md](reports/quality_measures.md).

> **About the earlier 0.842 figure:** earlier versions of this project reported ROC-AUC 0.842. That number
> was inflated by ICU stays duplicated across train and test, feature selection that used test rows, and a
> cohort (stays ≤ 48 h) whose outcome could fall inside the feature window. It has been retired.
> See [What changed in October 2026](#-what-changed-in-october-2026).

---

## 🧪 Curated data layer and data quality

The `dbt/` project (dbt-duckdb) attaches the MIMIC Postgres read-only and builds a local DuckDB warehouse:

| Layer | Models | What it does |
| :-- | :-- | :-- |
| Seeds | `icd9_cancer_category`, `chart_item_loinc`, `measurement_plausible_range` | Reference data we author: ICD-9 range → cancer category, a verified LOINC crosswalk for chart vitals, wide physiologic limits |
| Staging | `stg_*` | One model per source: renames, casts, an integer ICD-9 key for range joins |
| Intermediate | `int_stay_cancer_dx`, `int_icu_utilization`, `int_measurements_48h` | Cancer group and primary site; LOS, ICU death and 48 h readmission; labs and vitals in long format with LOINC codes |
| Marts | `fct_stay_outcomes`, `dim_measurement_item`, `dq_measurement_coverage` | Analytic base table; data dictionary of items, units and LOINC coverage; a coverage and plausibility monitor by ICU type and CareVue/MetaVision |

**Tests:**
- Keys, accepted values, ranges and relationships.
- The 48 h window holds for every measurement.
- The cohort rule and the category seed agree.
- Every feature-eligible measurement is LOINC-coded.
- Readmission is defined only for ICU survivors.
- Every stay has exactly one cross-fitted prediction.

Two `warn` tests surface items recorded in more than one unit, and the monitor's flags. Their triage is in [docs/business_rules.md](docs/business_rules.md#5-units-plausibility-and-missing-data).
Column descriptions in the YAML double as the data dictionary (`dbt docs generate`).

**Found by these checks:** the original cohort filter (ICD-9 140–239 plus a title match on "malignant"/"neoplasm")
included 390 benign or pre-cancerous stays. It also silently excluded 387 stays with leukemia, lymphoma, myeloma or melanoma, whose ICD titles use neither word.
Correcting it raised measured 30-day mortality from 25.1% to 28.6%.

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
| Data       | PostgreSQL 17, SQL, DuckDB (`postgres_scanner`), dbt (dbt-duckdb), pandas, PyArrow |
| Standards  | ICD-9-CM (Charlson/Quan code sets), LOINC |
| ML         | scikit-learn, XGBoost (feature ranking), SHAP, MLflow |
| UI         | Streamlit, Matplotlib |
| Dev        | Poetry, pytest, GitHub |

---

## 🗂 Repository Structure

```text
src/oncoai_prototype/
  data_loading/      SQL: MIMIC load, cohort definition, 48 h lab/vital extraction
  data_processing/   feature_engineering.py: one feature row per ICU stay
  modeling/          model_training.py (train + evaluate + cross-fit + export), predict.py (batch inference)
  analytics/         quality_measures.py (tiers, calibration, outcome summaries, suppression), run_quality_report.py
  utils/             db, feature, preprocessing, leakage-check, SHAP and I/O helpers
dbt/                 sources, seeds, staging/intermediate/mart models, data tests
docs/                business_rules.md, executive_brief.md
streamlit_app/       onco_dashboard.py
models/              published model artifacts (see models/README.md)
reports/             quality_measures.md + figures/ (aggregate, published); shap_plots/ (git-ignored)
tests/               unit tests (synthetic data) + `db` integration tests
data/, notebooks/    local-only working folders (contents git-ignored)
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

# 5. Features -> training (writes models/*.pkl, feature_ranges.json, metrics.json,
#    and data/processed/oof_predictions.parquet)
poetry run python -m oncoai_prototype.data_processing.feature_engineering
poetry run python -m oncoai_prototype.modeling.model_training

# 6. Curated layer + data tests (dbt reads ONCOAI_POSTGRES_CONN_STR from the environment)
set -a; source .env; set +a
(cd dbt && poetry run dbt deps && poetry run dbt build)

# 7. Outcomes / risk-stratification report (aggregates only)
poetry run python -m oncoai_prototype.analytics.run_quality_report

# 8. Tests: unit tests need no database; `-m db` runs the integration checks
poetry run pytest -q
poetry run pytest -q -m db

# 9. Dashboard (ONCOAI_MODE=mlflow uses the local MLflow registry; the default, github,
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

A repository-wide review fixed several problems that had inflated the reported performance.
A later clinical review aligned the cohort and measures with published definitions:

* **Cohort codes:** the title-text filter was replaced with the Charlson ICD-9-CM malignancy codes.
  This removed 390 benign, in-situ or non-melanoma skin stays and added 387 hematologic and melanoma stays that the filter had missed.
  The cohort went from 2,674 stays (25.1% mortality) to 2,671 (28.6%), and the model was retrained (ROC-AUC 0.764 → 0.768).
* **Measures:** reporting is by cancer group (hematologic / metastatic solid / non-metastatic solid). ICU readmission follows the SCCM 48 h definition among ICU survivors,
  and length of stay is split by survival. Observed/predicted ratios are presented as model calibration, not as a standardized mortality ratio, because the model is not a case-mix model.

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
* ICD-9 codes identify cancer diagnoses, not active treatment, stage or code status. They are assigned at discharge, with no present-on-admission flag.
* The mortality model is not a case-mix model, so unit-level comparisons of mortality are not risk-adjusted (see [business rules §6](docs/business_rules.md#why-this-is-not-a-standardized-mortality-ratio)).
* Feature selection and the train/test split use one random seed (42). The cross-validation spread (±0.019) shows how much results vary.
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
