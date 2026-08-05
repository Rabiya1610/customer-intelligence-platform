import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads raw dataset from storage."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    return pd.read_csv(filepath)

def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RATIONALE: Creates interaction terms based on domain logic:
    - charge_per_login: Cost relative to platform activity.
    - ticket_ratio: Support ticket density relative to account tenure.
    """
    df_feat = df.copy()
    
    # Avoid division by zero using + 1
    df_feat["charge_per_login"] = df_feat["total_charges"] / (df_feat["app_logins_30d"] + 1)
    df_feat["ticket_ratio"] = df_feat["support_tickets_30d"] / (df_feat["tenure_months"] + 1)
    
    return df_feat

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies scaling and categorical encoding based on EDA findings."""
    df_engineered = engineer_domain_features(df)
    
    # 1. Isolate primary key and target label
    customer_ids = df_engineered["customer_id"]
    target = df_engineered["churn"]
    
    features_df = df_engineered.drop(columns=["customer_id", "churn"])
    
    # 2. Identify categorical and numerical column sets
    categorical_cols = ["contract_type", "payment_method"]
    numeric_cols = [c for c in features_df.columns if c not in categorical_cols]
    
    # 3. One-Hot Encode categorical variables (Addresses Step 4 finding)
    encoded_cats = pd.get_dummies(features_df[categorical_cols], drop_first=True, dtype=int)
    
    # 4. Standardize numerical features (Addresses Step 3 finding)
    scaler = StandardScaler()
    scaled_num = pd.DataFrame(
        scaler.fit_transform(features_df[numeric_cols]),
        columns=numeric_cols
    )
    
    # 5. Reassemble processed feature matrix
    processed_df = pd.concat([customer_ids, scaled_num, encoded_cats, target], axis=1)
    return processed_df

def save_processed_data(df: pd.DataFrame, output_filepath: str) -> None:
    """Saves processed dataset to disk."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Processed dataset saved successfully to: {output_filepath}")

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "raw_customer_data.csv")
    processed_path = os.path.join("data", "processed", "processed_customer_data.csv")
    
    raw_df = load_raw_data(raw_path)
    processed_df = preprocess_features(raw_df)
    save_processed_data(processed_df, processed_path)