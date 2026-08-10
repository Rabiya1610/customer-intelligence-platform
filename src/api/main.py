import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI app instance
app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Real-time churn risk prediction microservice powered by XGBoost",
    version="1.0.0"
)

# Global model artifact path
MODEL_PATH = os.path.join("models", "xgboost_churn_model.pkl")
model = None

@app.on_event("startup")
def load_model():
    """Loads trained XGBoost model into memory when server starts."""
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("XGBoost Model successfully loaded into API memory.")

# Input validation schema using Pydantic
class CustomerFeatures(BaseModel):
    age: float = Field(..., example=0.7584)
    tenure_months: float = Field(..., example=-0.5210)
    monthly_charges: float = Field(..., example=1.1205)
    total_charges: float = Field(..., example=-0.3412)
    support_tickets_30d: float = Field(..., example=1.5420)
    app_logins_30d: float = Field(..., example=-0.8812)
    charge_per_login: float = Field(..., example=2.1050)
    ticket_ratio: float = Field(..., example=1.8540)
    contract_type_One_Year: int = Field(0, alias="contract_type_One-Year", example=0)
    contract_type_Two_Year: int = Field(0, alias="contract_type_Two-Year", example=0)
    payment_method_Credit_Card: int = Field(0, alias="payment_method_Credit Card", example=0)
    payment_method_Electronic_Check: int = Field(0, alias="payment_method_Electronic Check", example=1)

    class Config:
        allow_population_by_field_name = True

@app.get("/health")
def health_check():
    """Health endpoint for monitoring readiness."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    """Generates real-time churn prediction and risk level for single customer."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded.")
    
    # Convert JSON payload into single-row DataFrame
    input_data = customer.dict(by_alias=True)
    df_input = pd.DataFrame([input_data])
    
    # Run model inference
    prediction = int(model.predict(df_input)[0])
    probability = float(model.predict_proba(df_input)[0][1])
    
    # Map probability to risk level
    if probability < 0.35:
        risk_level = "Low Risk"
    elif probability < 0.70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
        
    return {
        "predicted_churn": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk_level
    }