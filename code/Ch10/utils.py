"""
Utility functions for Chapter 10: Building Univariate Time Series Models Using Statistical Methods
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import adfuller
from itertools import product

def split_data(data, test_split):
    """
    Split time series data into train and test sets while preserving temporal order.
    
    Parameters:
    data : pandas.DataFrame or Series
        Time series data to split
    test_split : float
        Proportion of data to use for testing (0 to 1)
        
    Returns:
    train, test : tuple of pandas.DataFrame or Series
        Training and test sets maintaining temporal order
    """
    l = len(data)
    t_idx = round(l*(1-test_split))
    train, test = data[ : t_idx], data[t_idx : ]
    print(f'train: {len(train)} , test: {len(test)}')
    return train, test

def check_stationarity(df):
    """
    Test if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Parameters:
    df : pandas Series/DataFrame
        Time series data
        
    Returns:
    tuple : (status, p_value)
        status: 'Stationary' or 'Non-Stationary'
        p_value: ADF test p-value
    """
    results = adfuller(df)[1:3]
    s = 'Non-Stationary'
    if results[0] < 0.05:
        s = 'Stationary'
    print(f"'{s}\t p-value:{results[0]} \t lags:{results[1]}")
    return (s, results[0])

def get_top_models_df(scores, criterion='AIC', top_n=5):
    """
    Rank time series models based on their performance metrics.
    
    Parameters:
    scores : dict
        Dictionary of model results and their metrics
    criterion : str, default='AIC'
        Metric for ranking (AIC, RMSE, MAPE, etc.)
    top_n : int, default=5
        Number of top models to return
        
    Returns:
    DataFrame with top performing models sorted by criterion
    """
    sorted_scores = sorted(scores.items(), 
                           key=lambda item: item[1][criterion])
    
    top_models = sorted_scores[:top_n]

    data = [v for k, v in top_models]
    df = pd.DataFrame(data)
    
    df['model_id'] = [k for k, v in top_models]
    df.set_index('model_id', inplace=True)

    return df

def plot_forecast(model, start, train, test):
    """
    Visualize model forecasts against actual values.
    
    Parameters:
    model : fitted time series model
    start : str or datetime
        Start date for the forecast
    train : pandas.Series
        Training data
    test : pandas.Series
        Test data
    """
    forecast = pd.DataFrame(model.forecast(test.shape[0]), 
                          index=test.index)
    
    ax = train.loc[start:].plot(style='--')
    test.plot(ax=ax)
    forecast.plot(ax=ax, style = '-.')
    ax.legend(['orig_train', 'orig_test', 'forecast'])
    plt.show()

def combinator(items, r=1):
    """
    Generate parameter combinations for grid search.
    
    Parameters:
    items : list of lists
        Lists of parameter values to combine
        
    Returns:
    list of tuples containing all possible parameter combinations
    """
    combo = [i for i in product(*items, repeat=r)]
    return combo
