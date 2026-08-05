import os
import pandas as pd
import numpy as np

def run_raw_data_inspection(filepath: str) -> None:
    """
    Executes the 6-step raw data inspection framework on raw customer records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    print("=" * 60)
    print("STEP 1: STRUCTURAL & SCHEMA AUDIT")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData Types & Memory Info:")
    print(df.info())

    print("\n" + "=" * 60)
    print("STEP 2: QUALITY & NULL AUDIT")
    print("=" * 60)
    null_counts = df.isnull().sum()
    print("Missing Values per Column:")
    print(null_counts)
    print(f"\nDuplicate Rows Count: {df.duplicated().sum()}")

    print("\n" + "=" * 60)
    print("STEP 3: STATISTICAL DISTRIBUTION AUDIT")
    print("=" * 60)
    print(df.describe().T)

    print("\n" + "=" * 60)
    print("STEP 4: CATEGORICAL INTEGRITY AUDIT")
    print("=" * 60)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        print(f"\n--- Distribution for '{col}' ---")
        print(df[col].value_counts(normalize=True))

    print("\n" + "=" * 60)
    print("STEP 5: TARGET DISTRIBUTION AUDIT")
    print("=" * 60)
    if "churn" in df.columns:
        print("Churn Target Ratio:")
        print(df["churn"].value_counts(normalize=True))

    print("\n" + "=" * 60)
    print("STEP 6: FIRST 5 SAMPLE RECORDS")
    print("=" * 60)
    print(df.head())

if __name__ == "__main__":
    raw_csv_path = os.path.join("data", "raw", "raw_customer_data.csv")
    run_raw_data_inspection(raw_csv_path)