import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def load_processed_data(filepath: str) -> pd.DataFrame:
    """Loads processed dataset from storage."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found at: {filepath}")
    return pd.read_csv(filepath)

def split_features_and_target(df: pd.DataFrame):
    """Splits matrix into predictor features (X) and target vector (y)."""
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]
    return X, y

def train_and_tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Trains XGBoost classifier using Stratified 5-Fold Cross-Validation and Hyperparameter Tuning."""
    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
    
    # Define hyperparameter grid for tuning tree depth, learning rate, and estimators
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    
    print("Executing Stratified 5-Fold Cross-Validation & Grid Search Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Cross-Validation ROC-AUC Score: {grid_search.best_score_:.4f}")
    print(f"Best Hyperparameters Selected: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluates the best tuned model on holdout test dataset."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "=" * 50)
    print("HOLDOUT TEST DATASET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy Score:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def save_model_artifact(model: XGBClassifier, output_filepath: str) -> None:
    """Saves serialized model binary artifact to storage."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    joblib.dump(model, output_filepath)
    print(f"\nModel artifact successfully saved to: {output_filepath}")

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "processed_customer_data.csv")
    model_output_path = os.path.join("models", "xgboost_churn_model.pkl")
    
    # 1. Load processed matrix
    df = load_processed_data(data_path)
    X, y = split_features_and_target(df)
    
    # 2. Train-Test Split (80% training, 20% holdout test with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Train, evaluate, and save artifact
    best_model = train_and_tune_model(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)
    save_model_artifact(best_model, model_output_path)
    