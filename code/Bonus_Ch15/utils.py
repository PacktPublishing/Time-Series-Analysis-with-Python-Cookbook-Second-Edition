"""
Utility functions for Bonus Chapter 15: Advanced Deep Learning for Time Series
"""

import matplotlib.pyplot as plt
import pandas as pd

def nf_plot_forecast(forecast_df, actuals_df, point_forecast, title):
    """
    Plots NeuralForecast predictions against actuals with prediction intervals.

    Args:
        forecast_df (pd.DataFrame): DataFrame from `nf.predict()` containing forecasts.
        actuals_df (pd.DataFrame): DataFrame with the true values for the forecast horizon.
        point_forecast (str): The column name of the point forecast in `forecast_df`.
        title (str): The title for the plot.
    """
    df = forecast_df.copy()
    df = df.set_index('ds')

    # Extract forecast 
    nf_yhat = df[point_forecast]
    lower_95 = df['NHITS-lo-95']
    upper_95 = df['NHITS-hi-95']
    lower_80 = df['NHITS-lo-80']
    upper_80 = df['NHITS-hi-80']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot Forecast 
    nf_yhat.plot(ax=ax, lw=2, label='Forecast')

    # Plot Actuals (test)
    actuals_df.set_index('ds')['y'].plot(ax=ax, style='--', lw=1.5, label='Actual')

    # Plot the 95% prediction interval
    ax.fill_between(
        df.index, 
        lower_95, 
        upper_95,
        alpha=0.15, # prediction band transparency
        label='95% PI'
    )
    # Plot the 80% prediction interval
    ax.fill_between(
        df.index, 
        lower_80, 
        upper_80,
        alpha=0.4, # prediction band transparency
        label='80% PI'
    )

    ax.set_title(title)
    ax.set_xlabel('Datetime')
    ax.set_ylabel('MW')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()




def plot_darts_forecast(actual_series, forecast_series, title):
    """
    Plots the actual time series against a Darts probabilistic forecast.
    
    Args:
        actual_series (darts.TimeSeries): The ground truth series (e.g., d_test).
        forecast_series (darts.TimeSeries): The probabilistic forecast from a Darts model.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(16, 5))
    
    # Plot the actuals
    actual_series.plot(label='Actual', linestyle='--', alpha=0.65)
    
    # Plot the probabilistic forecast with prediction intervals
    forecast_series.plot(
        label='Forecast',
        color='k',
        linestyle='-',
        alpha=0.25,
        low_quantile=0.05,
        high_quantile=0.95,
        central_quantile=0.5 # This plots the median
    )
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (MW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_statsforecast_prediction(forecast_df, test_df, model_name, ylabel='Value'):
    """
    Plot StatsForecast predictions with prediction intervals.
    
    Parameters:
    -----------
    forecast_df : pd.DataFrame
        Forecast DataFrame from StatsForecast
    test_df : pd.DataFrame
        Test data with actual values
    model_name : str
        Name of the model column (e.g., 'MSTL', 'AutoARIMA')
    ylabel : str, optional
        Y-axis label (default: 'Value')
    
    Returns:
    --------
    pd.Series
        The point forecast (yhat) for use in metric calculations
    """
    forecast = forecast_df.copy()
    test = test_df.copy()
    
    # Set index if needed
    if 'ds' in forecast.columns:
        forecast.set_index('ds', inplace=True)
    if 'ds' in test.columns:
        test.set_index('ds', inplace=True)
    
    # Extract point forecast
    yhat = forecast[model_name]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Forecast and actuals
    yhat.plot(ax=ax, lw=2, label='Forecast')
    test['y'].plot(ax=ax, style='--', lw=1.5, label='Actual')
    
    # Prediction intervals
    ax.fill_between(forecast.index, 
                    forecast[f'{model_name}-lo-95'], 
                    forecast[f'{model_name}-hi-95'],
                    alpha=0.15, label='95% PI')
    ax.fill_between(forecast.index,
                    forecast[f'{model_name}-lo-80'],
                    forecast[f'{model_name}-hi-80'],
                    alpha=0.4, label='80% PI')
    
    ax.set_title(f'{model_name} Forecast vs Actuals with Prediction Intervals')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    return yhat  # Return for metric calculations


def plot_statsmodels_forecast(forecast_result, actuals, alpha=0.05,
                               ylabel='Value', title=None, figsize=(10, 5)):
    """
    Plot statsmodels forecast with prediction intervals against actual values.
    
    Parameters:
    -----------
    forecast_result : statsmodels forecast object
        Forecast result from model.forecast() or model.get_forecast()
        Must have .predicted_mean and .conf_int() methods
    actuals : pd.Series or pd.DataFrame
        Actual values for comparison. If DataFrame, must contain 'y' column
    alpha : float, optional
        Significance level for prediction intervals (default: 0.05 for 95% PI)
    ylabel : str, optional
        Y-axis label (default: 'Value')
    title : str, optional
        Plot title (auto-generated if None)
    figsize : tuple, optional
        Figure size (default: (10, 5))
    
    Returns:
    --------
    tuple of (fig, ax)
        Matplotlib figure and axis objects
    
    Example:
    --------
    >>> from statsmodels.tsa.statespace.sarimax import SARIMAX
    >>> model = SARIMAX(train, order=(1,1,1))
    >>> result = model.fit()
    >>> fc = result.get_forecast(steps=12)
    >>> plot_statsmodels_forecast(fc, test, ylabel='MW')
    """
    # Extract forecast components
    yhat = forecast_result.predicted_mean
    pi = forecast_result.conf_int(alpha=alpha)
    
    # Get confidence interval columns (handles different naming conventions)
    pi_cols = pi.columns.tolist()
    lower_col = pi[pi_cols[0]]  # First column is lower bound
    upper_col = pi[pi_cols[1]]  # Second column is upper bound
    
    # Handle actuals (Series or DataFrame)
    if isinstance(actuals, pd.DataFrame):
        actuals_series = actuals['y']
    else:
        actuals_series = actuals
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot forecast
    yhat.plot(ax=ax, lw=2, label='Forecast', color='C0')
    
    # Plot actuals
    actuals_series.plot(ax=ax, style='--', lw=1.5, label='Actual', color='black')
    
    # Plot prediction interval
    confidence_pct = int((1 - alpha) * 100)
    ax.fill_between(
        pi.index,
        lower_col,
        upper_col,
        alpha=0.2,
        color='C0',
        label=f'{confidence_pct}% PI'
    )
    
    # Formatting
    if title is None:
        title = f'Forecast vs Actuals with {confidence_pct}% Prediction Interval'
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax