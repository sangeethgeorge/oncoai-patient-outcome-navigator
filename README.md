# 🧠 OncoAI Patient Outcome Navigator (Prototype)

An interactive, AI-driven dashboard for predicting oncology patient outcomes using structured and unstructured EHR data. Built as a full-stack proof-of-concept using the [MIMIC-IV demo dataset](https://physionet.org/content/mimic-iv-demo/2.2/), this project demonstrates end-to-end capabilities in data extraction, predictive modeling, clinical NLP, statistical validation, and user-facing dashboard design.

---

## 🎯 Project Goals

- Extract and process a mock oncology cohort from MIMIC-IV demo using ICD-10 codes
- Engineer time-series features from labs, vitals, and demographics
- Train ML models to predict 30-day mortality with SHAP explainability
- Use LLMs (e.g., GPT-4) to summarize clinical notes for interpretability
- Visualize patient risk, lab/vital trends, and GPT-generated summaries in a Streamlit dashboard
- Simulate statistical survival modeling using R (Kaplan-Meier, Cox regression)

---

## 🏗️ Project Architecture

```text
        ┌────────────────────┐
        │   MIMIC-IV Demo    │
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

---

## 🧪 Data Source

- **Dataset**: [MIMIC-IV Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/)
- **License**: PhysioNet Credentialed Health Data License 1.5.0
- **Data Tables Used**:
  - `patients`, `admissions` – demographics and mortality labels
  - `diagnoses_icd` – oncology cohort extraction
  - `labevents`, `chartevents` – vitals and lab values
  - `noteevents` – free-text clinical notes (ICU)

---

## 📊 Dashboard Features

| Feature                             | Description |
|------------------------------------|-------------|
| 🔍 Patient Lookup                  | Select a mock patient to view clinical data |
| 🧬 Lab/Vital Trends                | Time-series plots of selected labs (e.g., Hgb, CRP) |
| 📉 Mortality Risk Prediction       | Model output (e.g., 0.72 risk score) with SHAP explanation |
| 🧠 GPT-based Note Summarization   | Condensed patient history from ICU notes |
| 📈 Survival Curves (R)             | Simulated Kaplan-Meier curves (high vs low risk) |
| 🧰 Model Interpretability          | SHAP beeswarm + top contributing features |
| 💡 LLM Explanation Layer           | Natural language summary of "why this patient is high-risk" |

---

## 💻 Tech Stack

| Layer              | Tools & Libraries |
|-------------------|-------------------|
| **Data**          | PostgreSQL, Pandas, SQLAlchemy |
| **Modeling**      | Scikit-learn, SHAP |
| **NLP**           | spaCy, SciSpacy, GPT-4 (OpenAI API) |
| **Stats (R)**     | survival, survminer |
| **Frontend**      | Streamlit, Plotly |
| **Deployment**    | Docker, Streamlit Cloud, GitHub |

---

## 🧪 Results Snapshot (Demo Mode)

| Task                      | Result |
|---------------------------|--------|
| Oncology cohort           | 5 simulated patients from demo ICD codes |
| 30-day mortality model    | Logistic regression (AUC ~0.70, simulated) |
| SHAP insights             | CRP ↑, Hgb ↓, MAP ↓ as top risk drivers |
| GPT Summary (example)     | “This patient had progressive anemia, rising CRP, and hypotension prior to death.” |

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/sangeethgeorge/oncoai-patient-outcome-navigator.git
cd oncoai-patient-outcome-navigator

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run streamlit_app/app.py

---

## 🚀 Deployment

- Streamlit Cloud: streamlit_app/app.py
- Docker containerization support (optional)
- R scripts for survival analysis in /notebooks/

---

## 📌 Limitations

This project uses the MIMIC-IV demo dataset, which is highly restricted in patient volume and data diversity. While the pipelines and dashboard logic are realistic and reusable, the model outputs and subgroup insights are illustrative only. No clinical inference should be drawn from this prototype.

---

## 🤝 Contributing

Pull requests and issue discussions are welcome. Please see CONTRIBUTING.md for guidelines (coming soon).

---

## 📄 License

MIT License – see LICENSE file for details.

---

## ✨ Acknowledgements

- MIT Lab for Computational Physiology for MIMIC-III
- HuggingFace & OpenAI for LLM tools
- scikit-learn, SHAP, and spaCy communities

---
