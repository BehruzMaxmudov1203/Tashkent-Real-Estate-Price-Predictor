"""
Data Preprocessing Module for Tashkent Real Estate Dataset.
Handles type conversions, missing value imputation, outlier trimming, and encoding.
"""

import pandas as pd
import numpy as np


def clean_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw housing data by:
    1. Converting 'size' and 'price' to numeric values.
    2. Imputing missing 'size' values with the column mean.
    3. Dropping rows with missing 'price' values.
    4. Trimming outliers outside the 1st and 99th percentiles of price.
    5. One-hot encoding categorical variables (location, district).

    Args:
        df: Raw pandas DataFrame.

    Returns:
        Cleaned, encoded pandas DataFrame ready for modeling.
    """
    df_clean = df.copy()

    # 1. Clean size and price
    df_clean['size'] = pd.to_numeric(df_clean['size'], errors='coerce')
    df_clean['size'] = df_clean['size'].fillna(df_clean['size'].mean())

    df_clean['price'] = pd.to_numeric(df_clean['price'], errors='coerce')
    df_clean = df_clean.dropna(subset=['price'])

    # 2. Outlier removal (1% to 99% quantile)
    q_low = df_clean['price'].quantile(0.01)
    q_high = df_clean['price'].quantile(0.99)
    df_clean = df_clean[(df_clean['price'] >= q_low) & (df_clean['price'] <= q_high)]

    # 3. One-hot encoding for categorical variables
    categorical_cols = ['location', 'district']
    available_cats = [col for col in categorical_cols if col in df_clean.columns]
    
    if available_cats:
        df_clean = pd.get_dummies(df_clean, columns=available_cats, drop_first=True)

    return df_clean
