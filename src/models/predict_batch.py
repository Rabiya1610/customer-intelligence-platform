import os
import joblib
import pandas as pd
import numpy as np

def load_unseen_data(filepath: str) -> pd.DataFrame:
    """Loads new customer data requiring scoring."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input batch data not found at: {filepath}")
    return pd.read_csv(filepath)

def prepare_features_for_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Isolates customer IDs and extracts model feature matrix."""
    customer_ids = df["customer_id"]
    
    # Drop non-feature identifier and ground-truth target if present
    drop_cols = [col for col in ["customer_id", "churn"] if col in df.columns]
    X = df.drop(columns=drop_cols)
    
    return X, customer_ids

def run_batch_inference(model_path: str, data_path: str, output_path: str) -> pd.DataFrame:
    """Loads trained model, scores new customer data, and assigns risk buckets."""
    # 1. Ingest input data & model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model artifact not found at: {model_path}")
        
    model = joblib.load(model_path)
    df = load_unseen_data(data_path)
    
    X, customer_ids = prepare_features_for_inference(df)
    
    # 2. Generate predictions and probability scores
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # 3. Construct scoring output DataFrame
    output_df = pd.DataFrame({
        "customer_id": customer_ids,
        "predicted_churn": predictions,
        "churn_probability": np.round(probabilities, 4)
    })
    
    # 4. Map probability scores into actionable business risk buckets
    output_df["risk_level"] = pd.cut(
        output_df["churn_probability"],
        bins=[-0.01, 0.35, 0.70, 1.00],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    
    # 5. Export results to storage
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 55)
    print("BATCH INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 55)
    print(f"Total Customers Scored: {len(output_df)}")
    print(f"High Risk Flagged:      {(output_df['risk_level'] == 'High Risk').sum()}")
    print(f"Output saved to:        {output_path}")
    
    return output_df

if __name__ == "__main__":
    model_path = os.path.join("models", "xgboost_churn_model.pkl")
    # For testing, we run batch scoring on our processed matrix
    data_path = os.path.join("data", "processed", "processed_customer_data.csv")
    output_path = os.path.join("data", "processed", "batch_predictions.csv")
    
    results = run_batch_inference(model_path, data_path, output_path)
    
    print("\nSample Batch Predictions (First 5 Rows):")
    print(results.head())