# OncoAI Patient Outcome Navigator

🧠 Early risk prediction for critically ill cancer patients with MIMIC-III EHR data

🔗 **Try the Streamlit App:** [oncoai-db.streamlit.app](https://oncoai-db.streamlit.app)

---

## 🔍 Overview

Critically ill cancer patients face high ICU mortality risk, yet early prediction remains challenging due to fragmented EHR data. OncoAI helps clinical researchers, data scientists, and translational teams explore how early labs and vitals may signal short-term outcomes through an interpretable, modular dashboard.

OncoAI is a modular research prototype that explores how early ICU data in oncology patients can be used to:

* Define an oncology ICU cohort from ICD-9 codes in MIMIC-III v1.4
* Engineer time-series features from labs and vitals
* Train a 30-day mortality classifier with interpretability via SHAP
* Use NLP and LLMs (spaCy, GPT-4) to summarize ICU notes
* Visualize patient-level predictions in an interactive Streamlit dashboard

⚠️ **For research and educational use only.** Not for clinical decision-making.

---

## 🧱 Project Architecture

## Project Architecture

```text
        ┌────────────────────┐
        │  MIMIC-III v1.4    │
        └────────┬───────────┘
                 │
        ┌────────▼──────────┐
        │  SQL + Pandas ETL │
        └────────┬──────────┘
        ┌────────▼────────────┐     ┌──────────────────────┐
        │ Feature Engineering │◄───▶│ NLP/LLM Note Summary │
        └────────┬────────────┘     └──────────────────────┘
                 │
        ┌────────▼────────────┐
        │ Mortality Classifier│
        │ + SHAP Explanations │
        └────────┬────────────┘
                 │
        ┌────────▼────────────┐
        │ Streamlit Dashboard │
        └─────────────────────┘

```
---


## 📊 Project Progress (as of July 2025)

| Module                 | Status        | Notes                                          |
| :--------------------- | :------------ | :--------------------------------------------- |
| Cohort Definition      | ✅ Complete   | One row per first ICU stay ≥ 48 h (see below)  |
| ETL + Preprocessing    | ✅ Complete   | Modular utilities for lab/vital cleaning       |
| Feature Engineering    | ✅ Complete   | Aggregates + trends (e.g., CRP mean, MAP min)  |
| Modeling               | ✅ Complete   | Logistic regression; patient-grouped split, train-only feature selection, MLflow |
| SHAP Integration       | ✅ Complete   | `shap_utils.py` includes beeswarm, waterfall plots |
| Streamlit App          | ✅ Prototype Live | `onco_dashboard.py` serves patient risk view |
| NLP Module             | 🧪 In Progress | spaCy & GPT-4 tested for sample summaries      |
| R Survival Analysis    | 🧪 Simulated  | Kaplan-Meier example with dummy risk groups    |
| Docker/Cloud Deploy    | ⚙️ Planned    | Next phase (Week 6)                            |

---

## 📁 Data Source

* **Dataset:** MIMIC-III Clinical Database v1.4 (full, credentialed access via PhysioNet)
* **Tables:** `patients`, `admissions`, `icustays`, `diagnoses_icd`, `d_icd_diagnoses`, `labevents`, `d_labitems`, `chartevents`, `d_items`
* **License:** PhysioNet Credentialed Health Data License 1.5.0

> **No MIMIC data is distributed in this repository**, raw or derived. `data/`, `mlruns/` and notebooks
> are git-ignored. To reproduce, you need your own PhysioNet credentials and a local copy of MIMIC-III.

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

> Earlier versions of this project reported ROC-AUC 0.842. That figure was inflated by duplicated
> ICU stays across train/test, feature selection on test rows, and a cohort (stays ≤ 48 h) whose
> outcome could fall inside the feature window. It has been retired.

---

## 📈 Dashboard Features

| Feature                     | Description                                                   |
| :-------------------------- | :------------------------------------------------------------ |
| Feature input form          | Enter the model's 48 h lab/vital summaries (ranges from the training set) |
| 30-day mortality prediction | Logistic regression risk estimate                             |
| SHAP explanation            | Waterfall plot and table of per-feature contributions vs. the training mean |
| Model card                  | Held-out metrics, baseline comparison and cohort size         |

Planned (not yet in the app): ICU-note summaries (spaCy/LLM) and R survival analysis.

---

## ⚙️ Tech Stack

| Layer    | Stack                                   |
| :------- | :-------------------------------------- |
| Data     | Pandas, SQLAlchemy, PostgreSQL          |
| ML       | scikit-learn, SHAP, MLflow              |
| NLP      | spaCy, scispaCy, OpenAI GPT-4           |
| Survival | R (survival, survminer)                 |
| UI       | Streamlit, Plotly                       |
| DevOps   | Docker (planned), GitHub, Poetry        |

---

## 🚀 Setup Instructions

```bash
# 1. Clone and install
git clone https://github.com/sangeethgeorge/oncoai-patient-outcome-navigator.git
cd oncoai-patient-outcome-navigator
poetry install

# 2. Configure the database connection (never commit .env)
cp .env.example .env   # then fill in ONCOAI_POSTGRES_CONN_STR

# 3. Load MIMIC-III CSVs into Postgres (superuser; server-side COPY needs an absolute path)
psql -U postgres -d mimic-iii -v mimic_dir="$PWD/data/raw/mimic-iii-full" \
     -f src/oncoai_prototype/data_loading/01_mimic-iii_dataload.sql

# 4. Build the cohort and the 48 h extraction views
for f in 02_define_onco_cohort 03a_extract_all_labs_48h 03b_extract_all_vitals_48h; do
  psql -U postgres -d mimic-iii -f src/oncoai_prototype/data_loading/$f.sql
done

# 5. Features -> training (writes models/*.pkl, feature_ranges.json, metrics.json)
poetry run python -m oncoai_prototype.data_processing.feature_engineering
poetry run python -m oncoai_prototype.modeling.model_training

# 6. Tests: unit tests need no database; `-m db` runs the integration checks
poetry run pytest -q
poetry run pytest -q -m db

# 7. Dashboard
poetry run streamlit run streamlit_app/onco_dashboard.py
```

---

## ☁️ Deployment
✅ Prototype runs locally with streamlit run

🧪 Docker containerization in progress

🧪 Streamlit Cloud deployment planned (near future)

---

## ⚠️ Limitations
* Single-center retrospective data (Beth Israel Deaconess, 2001–2012); no external validation.
* ICD-9 codes identify cancer diagnoses, not active treatment or stage.
* The dashboard takes manually entered values; it is a demonstration, not a clinical tool.
* Do not use for clinical inference or decision-making.

---

## 📜 License
MIT License – see LICENSE

## 🙏 Acknowledgements
MIT Lab for Computational Physiology (MIMIC-III)

scikit-learn, SHAP, Streamlit, spaCy, HuggingFace, OpenAI

