# OncoAI Patient Outcome Navigator

🧠 AI-powered clinical dashboard for 30-day mortality prediction in oncology ICU patients using MIMIC-III demo data

---

## 🔍 Overview

OncoAI is a modular research prototype that explores how early ICU data in oncology patients can be used to:

* Define an oncology cohort using ICD-9/10 codes in MIMIC-III demo
* Engineer time-series features from labs and vitals
* Train a 30-day mortality classifier with interpretability via SHAP
* Use NLP and LLMs (spaCy, GPT-4) to summarize ICU notes
* Visualize patient-level predictions in an interactive Streamlit dashboard

⚠️ **For academic/demo use only** — all results are simulated using limited MIMIC-III demo data.

---

## 🧱 Project Architecture

## Project Architecture

```text
        ┌────────────────────┐
        │   MIMIC-III Demo   │
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
| Cohort Definition      | ✅ Complete   | Oncology cohort from diagnoses_icd             |
| ETL + Preprocessing    | ✅ Complete   | Modular utilities for lab/vital cleaning       |
| Feature Engineering    | ✅ Complete   | Aggregates + trends (e.g., CRP mean, MAP min)  |
| Modeling               | ✅ Complete   | Logistic regression with SHAP and MLflow logging |
| SHAP Integration       | ✅ Complete   | `shap_utils.py` includes beeswarm, waterfall plots |
| Streamlit App          | ✅ Prototype Live | `onco_dashboard.py` serves patient risk view |
| NLP Module             | 🧪 In Progress | spaCy & GPT-4 tested for sample summaries      |
| R Survival Analysis    | 🧪 Simulated  | Kaplan-Meier example with dummy risk groups    |
| Docker/Cloud Deploy    | ⚙️ Planned    | Next phase (Week 6)                            |

---

## 📁 Data Source

* **Dataset:** MIMIC-III Demo v2.2
* **Tables:** `patients`, `admissions`, `diagnoses_icd`, `labevents`, `chartevents`, `noteevents`
* **License:** PhysioNet Credentialed Health Data License 1.5.0

---

## 📈 Dashboard Features

| Feature                   | Description                                    |
| :------------------------ | :--------------------------------------------- |
| Patient Viewer            | Selects a demo patient from cohort             |
| Lab/Vital Trends          | Time-series plots for CRP, Hgb, MAP, etc.      |
| 30-day Mortality Prediction | Logistic regression model output               |
| SHAP Explainability       | Beeswarm and waterfall plots for top risk drivers |
| NLP Summary (LLM)         | GPT-4 condenses ICU notes (beta)               |
| Natural Language Rationale | LLM-generated explanation of prediction        |
| Survival Simulation (R)   | Simulated KM curves for high/low risk          |

---

## 🧪 Sample Results

| Metric                    | Value / Description                                      |
| :------------------------ | :------------------------------------------------------- |
| Oncology Patients         | 5 mock patients with cancer ICDs                         |
| AUC (LogReg, simulated)   | ~0.70                                                    |
| SHAP Top Drivers          | ↑ CRP, ↓ MAP, ↓ Hgb                                      |
| GPT-4 ICU Summary (example) | “This patient had progressive anemia, rising CRP, and hypotension prior to death.” |

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
# 1. Clone the repo
git clone [https://github.com/sangeethgeorge/oncoai-patient-outcome-navigator.git](https://github.com/sangeethgeorge/oncoai-patient-outcome-navigator.git)
cd oncoai-patient-outcome-navigator

# 2. Install Python dependencies
poetry install  # or pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run streamlit_app/onco_dashboard.py
```
---

## ☁️ Deployment
✅ Prototype runs locally with streamlit run

🧪 Docker containerization in progress

🧪 Streamlit Cloud deployment planned (Week 6)

---

## ⚠️ Limitations
Based on MIMIC-III demo, not real patient data

All predictions, summaries, and visuals are illustrative only

Do not use for clinical inference or decision-making

---

## 📜 License
MIT License – see LICENSE

## 🙏 Acknowledgements
MIT Lab for Computational Physiology (MIMIC-III)

scikit-learn, SHAP, spaCy, Streamlit

HuggingFace, OpenAI for NLP infrastructure
