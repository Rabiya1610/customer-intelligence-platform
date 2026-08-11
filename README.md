# 📊 Customer Churn Intelligence Platform

An enterprise-grade, end-to-end Machine Learning system engineered to predict, explain, and mitigate customer churn using **XGBoost**, **FastAPI**, **Streamlit**, and **Pandas**.

---

## 🌟 Architecture Overview

`
[ Raw Customer Data ]
         │
         ▼
[ Feature Engineering ] ──> Domain Ratios (e.g., ticket_ratio) & Categorical Encoding
         │
         ▼
[ XGBoost ML Engine ]   ──> Stratified 5-Fold Cross-Validation (0.9953 ROC-AUC)
         │
         ├───> [ Model Explainability ] ──> Gain Importance Analysis
         ├───> [ Batch Scoring ]        ──> Actionable Risk Buckets
         ├───> [ FastAPI REST API ]     ──> Sub-second /predict Endpoint
         └───> [ Streamlit Web UI ]     ──> Interactive Plotly Gauges
`

---

## ✨ Key Features & Highlights

- **Domain Feature Engineering:** Created high-signal domain features (including 	icket_ratio and charge_per_login) that tripled predictive signal compared to raw interaction counters.
- **High-Performance Classifier:** Trained an XGBoost model using hyperparameter optimization and Stratified 5-Fold Cross-Validation to handle class imbalance, reaching **0.9953 ROC-AUC**.
- **Dual Inference Engine:**
  - **Batch Inference Pipeline:** Scores offline customer datasets and maps records into actionable business risk buckets (Low Risk, Medium Risk, High Risk).
  - **Real-Time REST Microservice:** Sub-second production predictions served via FastAPI with Pydantic request/response payload validation.
- **Executive Visual Dashboard:** Interactive Streamlit dashboard with real-time parameter controls, interactive Plotly risk gauge charts, and churn probability readouts.

---

## 📁 Project Structure

customer-intelligence-platform/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── xgboost_churn_model.pkl
├── notebooks/
│   └── 01_exploratory_data_analysis.py
├── src/
│   ├── api/
│   │   └── main.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── evaluate_explainability.py
│   │   ├── predict_batch.py
│   │   └── train_model.py
│   └── visualization/
│       └── dashboard.py
├── .gitignore
├── README.md
└── requirements.txt

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Ensure **Python 3.10+** is installed on your system.

`powershell
git clone [https://github.com/Rabiya1610/customer-intelligence-platform.git](https://github.com/Rabiya1610/customer-intelligence-platform.git)
cd customer-intelligence-platform

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
`

---

### 2. Running Data & ML Pipelines

Execute the scripts sequentially to process features, train the model, evaluate feature drivers, and run batch predictions.

#### Step A: Feature Engineering
`powershell
python src/features/build_features.py
`

#### Step B: Model Training & Tuning
`powershell
python src/models/train_model.py
`

#### Step C: Evaluate Explainability
`powershell
python src/models/evaluate_explainability.py
`

#### Step D: Batch Inference Pipeline
`powershell
python src/models/predict_batch.py
`

---

### 3. Launching Applications & Services

#### Option A: FastAPI Real-Time Service
`powershell
python -m uvicorn src.api.main:app --reload --port 8000
`
- **Interactive Swagger Docs:** http://127.0.0.1:8000/docs
- **Health Check Endpoint:** http://127.0.0.1:8000/health
- **Prediction Endpoint:** POST http://127.0.0.1:8000/predict

#### Option B: Streamlit Web Dashboard
`powershell
python -m streamlit run src/visualization/dashboard.py
`
- **Dashboard Access:** http://localhost:8501

---

## 📊 Model Evaluation & Metrics

| Metric / Parameter | Value / Finding |
| :--- | :--- |
| **ROC-AUC Score** | **0.9953** |
| **Validation Strategy** | Stratified 5-Fold Cross-Validation |
| **Top Churn Drivers** | Contract Type (34.6%), Ticket Ratio (17.7%), Monthly Charges (11.7%) |

> **Key Behavioral Insight:** Feature importance analysis revealed that the engineered **	icket_ratio** (support_tickets_30d / tenure_months) was over **3x more predictive** than raw ticket counts alone, proving that customer frustration relative to tenure is the strongest driver of churn.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Machine Learning** | XGBoost, Scikit-Learn, Joblib, NumPy |
| **Data Processing** | Pandas |
| **API & Web Services** | FastAPI, Uvicorn, Pydantic |
| **Visualization & UI** | Streamlit, Plotly |
| **Version Control** | Git, GitHub
