# OncoAI Patient Outcome Navigator

**AI-driven clinical dashboard for predicting 30-day mortality in oncology ICU patients using MIMIC-III data**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-App-green.svg) ![scikit-learn](https://img.shields.io/badge/sklearn-Model-orange.svg) ![spaCy](https://img.shields.io/badge/NLP-spaCy%20%7C%20GPT4-lightgrey.svg)

---

## Overview

**OncoAI** is a research prototype that leverages electronic health record (EHR) data to:

- Identify oncology ICU cohorts from MIMIC-III/IV using ICD-9/ICD-10 codes
- Extract labs, vitals, and clinical notes to build structured and unstructured features
- Train predictive models for 30-day mortality
- Generate explainability outputs using SHAP
- Summarize ICU patient notes using NLP + LLMs (spaCy, GPT-4)
- Visualize patient-level insights in an interactive **Streamlit dashboard**

> For academic/research use only — built using the MIMIC demo dataset.

---

## Project Architecture

```text
        ┌────────────────────┐
        │   MIMIC-III Demo    │
        └────────┬───────────┘
                 │
        ┌────────▼──────────┐
        │  SQL + Pandas ETL │◄─────────────┐
        └────────┬──────────┘              │
        ┌────────▼────────────┐     ┌──────▼───────┐
        │ Feature Engineering │◄───▶│ Clinical NLP │
        └────────┬────────────┘     └──────┬───────┘
                 │                         │
        ┌────────▼──────────┐      ┌───────▼────────┐
        │   ML Modeling     │      │  GPT Summary   │
        └────────┬──────────┘      └───────┬────────┘
                 │                         │
        ┌────────▼──────────────┐          │
        │  R Survival Modeling  │◄─────────┘
        └────────┬──────────────┘
                 │
        ┌────────▼────────────┐
        │ Streamlit Dashboard │
        └─────────────────────┘
```
---

## Data Source

- **Dataset**: [MIMIC-III Demo v2.2](https://physionet.org/content/MIMIC-III-demo/2.2/)
- **License**: PhysioNet Credentialed Health Data License 1.5.0
- **Data Tables Used**:
  - `patients`, `admissions` – demographics and mortality labels
  - `diagnoses_icd` – oncology cohort extraction
  - `labevents`, `chartevents` – vitals and lab values
  - `noteevents` – free-text clinical notes (ICU)

---

## Dashboard Features

| Feature                             | Description |
|------------------------------------|-------------|
| Patient Lookup                  | Select a mock patient to view clinical data |
| Lab/Vital Trends                | Time-series plots of selected labs (e.g., Hgb, CRP) |
| Mortality Risk Prediction       | Model output (e.g., 0.72 risk score) with SHAP explanation |
| GPT-based Note Summarization   | Condensed patient history from ICU notes |
| Survival Curves (R)             | Simulated Kaplan-Meier curves (high vs low risk) |
| Model Interpretability          | SHAP beeswarm + top contributing features |
| LLM Explanation Layer           | Natural language summary of "why this patient is high-risk" |

---

## Tech Stack

| Layer              | Tools & Libraries |
|-------------------|-------------------|
| **Data**          | PostgreSQL, Pandas, SQLAlchemy |
| **Modeling**      | Scikit-learn, SHAP |
| **NLP**           | spaCy, SciSpacy, GPT-4 (OpenAI API) |
| **Stats (R)**     | survival, survminer |
| **Frontend**      | Streamlit, Plotly |
| **Deployment**    | Docker, Streamlit Cloud, GitHub |

---

## Results Snapshot (Demo Mode)

| Task                      | Result |
|---------------------------|--------|
| Oncology cohort           | 5 simulated patients from demo ICD codes |
| 30-day mortality model    | Logistic regression (AUC ~0.70, simulated) |
| SHAP insights             | CRP ↑, Hgb ↓, MAP ↓ as top risk drivers |
| GPT Summary (example)     | “This patient had progressive anemia, rising CRP, and hypotension prior to death.” |

---

## Installation

```bash
# Clone repo
git clone https://github.com/sangeethgeorge/oncoai-patient-outcome-navigator.git
cd oncoai-patient-outcome-navigator

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run streamlit_app/app.py

```
---

## Deployment

- Streamlit Cloud: streamlit_app/app.py
- Docker containerization support (optional)
- R scripts for survival analysis in /notebooks/

---

## Limitations

This project uses the MIMIC-III demo dataset, which is highly restricted in patient volume and data diversity. While the pipelines and dashboard logic are realistic and reusable, the model outputs and subgroup insights are illustrative only. No clinical inference should be drawn from this prototype.

---

## Contributing

Pull requests and issue discussions are welcome. Please see CONTRIBUTING.md for guidelines (coming soon).

---

## License

MIT License – see LICENSE file for details.

---

## Acknowledgements

- MIT Lab for Computational Physiology for MIMIC-III
- HuggingFace & OpenAI for LLM tools
- scikit-learn, SHAP, and spaCy communities

---
