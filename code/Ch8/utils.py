"""
Utility functions for Chapter 8: Outlier Detection (Statistical)
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_outliers(outliers, data, method='KNN', halignment='right', valignment='bottom', labels=False):
    """
    Plot time series data with highlighted outliers.
    
    Parameters
    ----------
    outliers : pandas.DataFrame or pandas.Series
        The DataFrame or Series containing the outlier data points.
    data : pandas.DataFrame or pandas.Series
        The complete time series data.
    method : str, default='KNN'
        The outlier detection method used, displayed in the plot title.
    halignment : str, default='right'
        Horizontal alignment for the date labels ('left', 'center', or 'right').
    valignment : str, default='bottom'
        Vertical alignment for the date labels ('top', 'center', or 'bottom').
    labels : bool, default=False
        If True, displays date labels for each outlier point.
        
    Returns
    -------
    None
        The function shows the plot but does not return any value.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
        
    data.plot(ax=ax, alpha=0.6)
    
    # Plot outliers
    if labels:
        outliers.plot(ax=ax, style='rx', markersize=8, legend=False)
        
        # Add text labels for each outlier
        for idx, value in outliers['value'].items():
            ax.text(idx, value, f'{idx.date()}', 
                   horizontalalignment=halignment, 
                   verticalalignment=valignment)
    else:
        outliers.plot(ax=ax, style='rx', legend=False)
    
    ax.set_title(f'NYC Taxi - {method}')
    ax.set_xlabel('date')
    ax.set_ylabel('# of passengers')
    ax.legend(['nyc taxi', 'outliers'])
    
    plt.tight_layout()
    plt.show()

def plot_zscore(data_series, d=3):
    """
    Plot the standardized z-scores with threshold lines using Series index for x-axis.
    
    Parameters:
    - data_series: Series containing z-scores with datetime index
    - d: Threshold in standard deviations (default: 3)
    """
    
    plt.plot(data_series.index, data_series.values, 'k^', markersize=4)
    
    plt.axhline(y=d, color='r', linestyle='--', label=f'+{d} SD')
    plt.axhline(y=-d, color='r', linestyle='--', label=f'-{d} SD')
    
    # Highlight outliers
    outliers = data_series[abs(data_series) > d]
    if not outliers.empty:
        plt.plot(outliers.index, outliers.values, 'ro', markersize=8, label='Outliers')
    
    plt.ylabel('Z-score')
    plt.title('Standardized Taxi Passenger Data with Outlier Thresholds')
    plt.legend()
    
    # Format x-axis for dates
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
