"""
Training Pipeline for Tashkent House Price Prediction.
Trains Linear Regression, Random Forest, and Gradient Boosting models,
evaluates them using MAE, RMSE, and R2, and saves the trained models to disk.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import clean_housing_data

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "housing_data_08-02-2021.csv")
REMOTE_DATA_URL = "https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/housing_data_08-02-2021.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_dataset(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Loads dataset from local file if available, otherwise from remote URL."""
    if os.path.exists(filepath):
        print(f"Loading local dataset from: {filepath}")
        return pd.read_csv(filepath)
    print(f"Local dataset not found. Downloading from: {REMOTE_DATA_URL}")
    return pd.read_csv(REMOTE_DATA_URL)


def evaluate(y_true, y_pred, name: str):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"  MAE:      ${mae:,.2f}")
    print(f"  RMSE:     ${rmse:,.2f}")
    print(f"  R2 Score: {r2:.4f} ({r2 * 100:.2f}% variance explained)\n")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Step 1: Loading raw data...")
    df_raw = load_dataset()
    print(f"Loaded {len(df_raw)} records.")

    print("\nStep 2: Cleaning and preprocessing...")
    df_clean = clean_housing_data(df_raw)
    print(f"Shape after cleaning & one-hot encoding: {df_clean.shape}")

    X = df_clean.drop("price", axis=1)
    y = df_clean["price"]

    print("\nStep 3: Splitting train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nStep 4: Training candidate models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = evaluate(y_test, preds, name)

    # Save the top models
    rf_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
    gb_path = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")

    with open(rf_path, "wb") as f:
        pickle.dump(models["Random Forest"], f)

    with open(gb_path, "wb") as f:
        pickle.dump(models["Gradient Boosting"], f)

    with open(features_path, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Models and feature list successfully saved to {MODEL_DIR}/")


if __name__ == "__main__":
    run_pipeline()
