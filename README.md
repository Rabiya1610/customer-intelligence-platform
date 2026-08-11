📊 Customer Churn Intelligence PlatformAn enterprise-grade, end-to-end Machine Learning system engineered to predict, explain, and mitigate customer churn using XGBoost, FastAPI, Streamlit, and Pandas.🌟 Architecture Overview[ Raw Customer Data ]
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
✨ Key Features & HighlightsDomain Feature Engineering: Created high-signal domain features (including ticket_ratio and charge_per_login) that tripled predictive signal compared to raw interaction counters.High-Performance Classifier: Trained an XGBoost model using hyperparameter optimization and Stratified 5-Fold Cross-Validation to handle class imbalance, reaching 0.9953 ROC-AUC.Dual Inference Engine:Batch Inference Pipeline: Scores offline customer datasets and maps records into actionable business risk buckets (Low Risk, Medium Risk, High Risk).Real-Time REST Microservice: Sub-second production predictions served via FastAPI with Pydantic request/response payload validation.Executive Visual Dashboard: Interactive Streamlit dashboard with real-time parameter controls, interactive Plotly risk gauge charts, and churn probability readouts.📁 Project Structurecustomer-intelligence-platform/
│
├── data/
│   ├── raw/                      # Raw customer datasets
│   └── processed/                # Transformed feature matrices & predictions
│
├── models/
│   ├── xgboost_churn_model.pkl   # Serialized trained model artifact
│   └── .gitkeep
│
├── notebooks/
│   └── 01_exploratory_data_analysis.py  # EDA, distributions, and class auditing
│
├── src/
│   ├── features/
│   │   └── build_features.py            # Feature engineering & scaling pipeline
│   ├── models/
│   │   ├── train_model.py               # Model training & Stratified CV grid search
│   │   ├── evaluate_explainability.py   # Feature importance extraction
│   │   └── predict_batch.py             # Batch scoring engine & risk categorization
│   ├── api/
│   │   └── main.py                      # FastAPI real-time REST microservice
│   └── visualization/
│       └── dashboard.py                 # Interactive Streamlit web application
│
├── .gitignore                    # Environment & artifact ignore rules
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
🚀 Quickstart Guide1. Prerequisites & InstallationEnsure Python 3.10+ is installed on your system.PowerShellgit clone [https://github.com/Rabiya1610/customer-intelligence-platform.git](https://github.com/Rabiya1610/customer-intelligence-platform.git)
cd customer-intelligence-platform

python -m venv .venv
.\.venv\Scripts\activate

pip install -r requirements.txt
2. Running Data & ML PipelinesExecute the scripts sequentially to process features, train the model, evaluate feature drivers, and run batch predictions.Step A: Feature EngineeringPowerShellpython src/features/build_features.py
Step B: Model Training & TuningPowerShellpython src/models/train_model.py
Step C: Evaluate ExplainabilityPowerShellpython src/models/evaluate_explainability.py
Step D: Batch Inference PipelinePowerShellpython src/models/predict_batch.py
3. Launching Applications & ServicesOption A: FastAPI Real-Time ServicePowerShellpython -m uvicorn src.api.main:app --reload --port 8000
Interactive Swagger Docs: http://127.0.0.1:8000/docsHealth Check Endpoint: http://127.0.0.1:8000/healthPrediction Endpoint: POST http://127.0.0.1:8000/predictOption B: Streamlit Web DashboardPowerShellpython -m streamlit run src/visualization/dashboard.py
Dashboard Access: http://localhost:8501📊 Model Evaluation & MetricsMetric / ParameterValue / FindingROC-AUC Score0.9953Validation StrategyStratified 5-Fold Cross-ValidationTop Churn DriversContract Type (34.6%), Ticket Ratio (17.7%), Monthly Charges (11.7%)Key Behavioral Insight: Feature importance analysis revealed that the engineered ticket_ratio ($\frac{\text{support\_tickets\_30d}}{\text{tenure\_months}}$) was over 3x more predictive than raw ticket counts alone, proving that customer frustration relative to tenure is the strongest driver of churn.🛠️ Tech StackCategoryTechnologiesMachine LearningXGBoost, Scikit-Learn, Joblib, NumPyData ProcessingPandasAPI & Web ServicesFastAPI, Uvicorn, PydanticVisualization & UIStreamlit, PlotlyVersion ControlGit, GitHub