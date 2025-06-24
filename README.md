# 💊📈 OncoAI Patient Outcome Navigator


**Predict oncology patient outcomes using EHR data + LLM-based explanations.**

OncoAI is a full-stack AI platform that predicts clinical outcomes in ICU cancer patients using structured EHR data (labs, vitals, diagnoses) and unstructured notes. It combines machine learning, NLP, and survival analysis to deliver interpretable insights via an interactive Streamlit dashboard.

---

## 🚀 Project Goals

- ✅ Predict 30-day mortality and other clinical outcomes
- ✅ Integrate structured data (MIMIC-III: labs, vitals, diagnoses) + unstructured notes
- ✅ Use LLMs to summarize patient history and explain predictions
- ✅ Visualize risk factors and trends for clinical use

---

## 🏗️ Project Architecture

Data (MIMIC-III) → ETL → Feature Engineering → ML Model → LLM Explanation → Streamlit Dashboard


- SQL + Pandas for cohort building and data wrangling  
- Scikit-learn for modeling and SHAP-based interpretation  
- spaCy + GPT-4 for clinical note parsing and summarization  
- R for Cox regression and survival plots  
- Streamlit for front-end dashboard

---

## 🗃️ Repository Structure

oncoai/
├── data/ # Raw & processed data (excluded from GitHub)
├── notebooks/ # Exploratory data analysis (Jupyter, R)
├── src/ # Core Python modules (etl, modeling, dashboard)
│ ├── etl.py
│ ├── features.py
│ ├── model.py
│ ├── gpt_summarizer.py
│ └── app.py
├── reports/ # SHAP plots, survival curves, markdown outputs
├── requirements.txt # Python dependencies
├── README.md
└── streamlit_app/ # Streamlit dashboard UI

---

## 📊 Dashboard Features

- **Patient ID Lookup**  
  View structured data and NLP summaries

- **Risk Score + Explanation**  
  SHAP values, Cox hazard ratios, key drivers

- **Vitals/Labs Trend Plots**  
  Time series visualization (e.g. CRP, WBC, BP)

- **GPT-based Summary**  
  Plain-English summaries of patient notes

---

## 📦 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/oncoai-patient-outcome-navigator.git
cd oncoai-patient-outcome-navigator

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Activate R environment for survival analysis
# See R/README.md for details

# Launch Streamlit dashboard
streamlit run streamlit_app/app.py

```
---

## 🧪 Example Use Case
"Given an ICU patient with stage IV NSCLC, abnormal CRP and WBC levels, and history of infection noted in discharge summaries, what is their predicted 30-day mortality risk and how is it explained?"

→ OncoAI returns:

30-day mortality risk = 67%

Top risk drivers: high CRP, neutropenia, elevated lactate

GPT summary: “Patient experienced sepsis 2 days post-chemo; elevated CRP levels noted; recent hypotension…”

---

## 📚 Data Source
MIMIC-III v1.4 – Medical Information Mart for Intensive Care
Requires credentialed access via PhysioNet.

---

## 🛠️ Tech Stack

| Category | Tools Used |
| --- | --- |
| Data Wrangling | SQL, Pandas|
| Modeling | Scikit-learn, SHAP |
| NLP | spaCy, GPT-4, clinicalBERT (optional) |
| Stats | R (survival, survminer) |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud or Docker + EC2|

## 📈 Roadmap

-  Baseline model with labs/vitals
-  GPT-based note summarization
-  Add comorbidity extraction from notes
-  Integrate SEER or MIMIC-IV for validation
-  Add treatment prediction module

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
