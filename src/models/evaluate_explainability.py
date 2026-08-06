import os
import joblib
import pandas as pd
import numpy as np

def load_model_and_feature_names(model_path: str, data_path: str):
    """Loads saved model artifact and extracts feature column names."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at: {data_path}")
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # Isolate predictor feature names
    feature_names = df.drop(columns=["customer_id", "churn"]).columns.tolist()
    return model, feature_names

def calculate_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extracts and normalizes feature importances from trained model."""
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Gain_Importance": importances
    }).sort_values(by="Gain_Importance", ascending=False).reset_index(drop=True)
    
    # Convert raw gain into percentage contribution
    total_gain = importance_df["Gain_Importance"].sum()
    importance_df["Contribution_%"] = (importance_df["Gain_Importance"] / total_gain) * 100
    
    return importance_df

if __name__ == "__main__":
    model_path = os.path.join("models", "xgboost_churn_model.pkl")
    data_path = os.path.join("data", "processed", "processed_customer_data.csv")
    
    model, feature_names = load_model_and_feature_names(model_path, data_path)
    importance_df = calculate_feature_importance(model, feature_names)
    
    print("\n" + "=" * 60)
    print("XGBOOST FEATURE IMPORTANCE RANKING (TOP CHURN DRIVERS)")
    print("=" * 60)
    print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))