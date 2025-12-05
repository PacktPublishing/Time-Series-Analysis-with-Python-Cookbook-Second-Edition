"""
Utility functions for Chapter 11: Additional Statistical Modeling Techniques for Time Series
"""

import pandas as pd
from statsmodels.tsa.api import adfuller

def split_data(data, test_split):
    """Split time series data into training and test sets.
    
    Args:
        data (pd.Series or pd.DataFrame): Time series data to split
        test_split (int): Number of periods to use for testing
        
    Returns:
        tuple: (train_data, test_data)
    """
    t_idx = test_split
    train, test = data[ : -t_idx], data[-t_idx : ]
    print(f'train: {len(train)} , test: {len(test)}')
    return train, test

def check_stationarity(df):
    adf_pv = adfuller(df)[1]
    result = 'Stationary' if adf_pv < 0.05 else "Non-Stationary"
    return result
