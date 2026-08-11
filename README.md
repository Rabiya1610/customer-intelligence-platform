# 📊 Customer Churn Intelligence Platform

An end-to-end, enterprise-grade Machine Learning system designed to predict, explain, and mitigate customer churn. Built with **XGBoost**, **FastAPI**, **Streamlit**, and **Pandas**, this platform bridges the gap between raw data processing, high-performance ML modeling, production REST API serving, and business stakeholder dashboards.

---

## 🌟 Key Architecture & Highlights


[ Raw Customer Data ]
         │
         ▼
[ Feature Engineering ] ──> Creates domain ratios (e.g., ticket_ratio) & encodes categoricals
         │
         ▼
[ XGBoost Model Engine ] ──> Optimized via Stratified 5-Fold Cross-Validation (0.9953 ROC-AUC)
         │
         ├───> [ Model Explainability ]   ──> Ranks top churn drivers (Gain Importance)
         ├───> [ Batch Inference Pipeline ] ──> Processes offline bulk CSVs & maps Risk Buckets
         ├───> [ Real-Time REST API ]     ──> Sub-second inference via FastAPI (/predict)
         └───> [ Interactive Dashboard ]  ──> Visual Streamlit UI with real-time gauge controls



## 📁 Project Structure


customer-intelligence-platform/
│
├── data/
│   ├── raw/                      # Initial raw dataset
│   └── processed/                # Scaled feature matrices & batch prediction outputs
│
├── models/
│   └── xgboost_churn_model.pkl   # Serialized trained model artifact
│
├── notebooks/
│   └── 01_exploratory_data_analysis.py  # EDA script auditing distributions & imbalance
│
├── src/
│   ├── features/
│   │   └── build_features.py            # Feature engineering & scaling pipeline
│   ├── models/
│   │   ├── train_model.py               # Model training & Stratified CV grid search
│   │   ├── evaluate_explainability.py   # Feature importance extraction script
│   │   └── predict_batch.py             # Batch scoring engine & risk categorization
│   ├── api/
│   │   └── main.py                      # FastAPI real-time REST microservice
│   └── visualization/
│       └── dashboard.py                 # Interactive Streamlit web application
│
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation


## 🚀 Quickstart Guide

### 1. Prerequisites & Virtual Environment Setup

Ensure Python 3.10+ is installed on your system.

```powershell
# Clone the repository
git clone [https://github.com/your-username/customer-intelligence-platform.git](https://github.com/your-username/customer-intelligence-platform.git)
cd customer-intelligence-platform

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


### 📊 Model Performance & Key Drivers

| Metric | Score |
| :--- | :--- |
| **ROC-AUC Score** | **0.9953** |
| **Cross-Validation Strategy** | Stratified 5-Fold |
| **Primary Churn Drivers** | Contract Type (34.6%), Ticket Ratio (17.7%), Monthly Charges (11.7%) |

### Top Drivers Insight
Feature importance analysis demonstrated that the engineered **`ticket_ratio`** ($\frac{\text{support\_tickets\_30d}}{\text{tenure\_months}}$) was over **3x more predictive** than raw ticket counts alone, confirming that customer frustration relative to account tenure is the single strongest behavioral indicator of churn risk.


## 🛠️ Tech Stack

- **Machine Learning:** XGBoost, Scikit-Learn, Joblib, NumPy
- **Data Engineering:** Pandas
- **REST API:** FastAPI, Uvicorn, Pydantic
- **Dashboard & Visualization:** Streamlit, Plotly