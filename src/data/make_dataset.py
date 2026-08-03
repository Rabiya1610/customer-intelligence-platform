import os
import numpy as np
import pandas as pd

def generate_raw_customer_data(num_records: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic customer dataset simulating subscription churn mechanics.
    """
    np.random.seed(seed)
    
    customer_ids = [f"CUST-{i:04d}" for i in range(1, num_records + 1)]
    ages = np.random.randint(18, 71, size=num_records)
    tenure_months = np.random.randint(1, 61, size=num_records)
    monthly_charges = np.round(np.random.uniform(20.0, 150.0, size=num_records), 2)
    
    # Calculate total charges with noise
    total_charges = np.round(tenure_months * monthly_charges * np.random.uniform(0.9, 1.05, size=num_records), 2)
    
    contract_types = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=num_records, p=[0.5, 0.3, 0.2])
    payment_methods = np.random.choice(["Credit Card", "Bank Transfer", "Electronic Check"], size=num_records, p=[0.4, 0.3, 0.3])
    
    support_tickets = np.random.poisson(lam=2, size=num_records)
    app_logins = np.random.poisson(lam=20, size=num_records)
    
    # Define underlying probabilistic churn rule (Simulating real-world human decisions)
    churn_score = (
        (support_tickets * 0.35) - 
        (app_logins * 0.08) + 
        (monthly_charges * 0.015) - 
        (tenure_months * 0.03) + 
        np.where(contract_types == "Month-to-Month", 0.8, -0.5)
    )
    
    # Convert log-odds score to probability using sigmoid function
    churn_probs = 1 / (1 + np.exp(-churn_score))
    churn_labels = (churn_probs > np.percentile(churn_probs, 80)).astype(int)  # ~20% churn rate
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "support_tickets_30d": support_tickets,
        "app_logins_30d": app_logins,
        "churn": churn_labels
    })
    
    return df

def save_data(df: pd.DataFrame, output_filepath: str) -> None:
    """Saves DataFrame to CSV ensuring destination path exists."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Dataset saved successfully to: {output_filepath}")

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "raw_customer_data.csv")
    dataset = generate_raw_customer_data(num_records=1000)
    save_data(dataset, raw_data_path)