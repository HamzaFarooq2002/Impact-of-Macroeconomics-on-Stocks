# inflation.py

import pandas as pd
import numpy as np

def adjust_for_inflation(df: pd.DataFrame, price_col: str = "Close", year_col: str = "Date") -> pd.DataFrame:
    """
    Adjusts price values for inflation using sample CPI values.
    Replace the CPI dictionary with actual CPI data for accuracy.
    
    Args:
        df (pd.DataFrame): Input dataframe containing price data
        price_col (str): Name of the column containing price values
        year_col (str): Name of the column containing date information
        
    Returns:
        pd.DataFrame: DataFrame with inflation-adjusted prices
        
    Raises:
        ValueError: If required columns are missing
    """
    if year_col not in df.columns:
        raise ValueError("Date column required for inflation adjustment.")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in the dataframe.")

    df = df.copy()
    df[year_col] = pd.to_datetime(df[year_col])
    df['Year'] = df[year_col].dt.year

    # Dummy CPI index values (base = 2023 = 1.0)
    sample_cpi = {
        2019: 0.92,
        2020: 0.95,
        2021: 0.97,
        2022: 0.99,
        2023: 1.00,
        2024: 1.02,
        2025: 1.05,
    }

    df['Inflation Factor'] = df['Year'].map(sample_cpi)
    df['Inflation Factor'].fillna(1.0, inplace=True)
    df[f'{price_col}_Adj'] = df[price_col] / df['Inflation Factor']

    return df
