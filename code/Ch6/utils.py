"""
Utility functions for Chapter 6: Handling Missing Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_dataset(folder, file, date_col=None, format=None, index=False):
    '''
    Reads a CSV dataset from a specified folder and converts date columns to datetime.
    
    Parameters:
    folder: the directory containing the file passed a Path object
    file: the CSV filename in that Path object. 
    date_col: specify a column which has datetime
    format: the date format string for parsing dates
    index: True if date_col should be the index
    
    Returns: 
    pandas DataFrame with a DatetimeIndex
    '''
    index_col = date_col if index is True else None
    
    df = pd.read_csv(folder / file, 
                     index_col=index_col, 
                     parse_dates=[date_col],
                     date_format=format)
    return df

def plot_dfs(df1, df2, col, title=None, xlabel=None, ylabel=None):
    '''
    Creates comparative plots of original data versus data with missing values.
    
    Parameters:
    df1: original dataframe without missing data
    df2: dataframe with missing data 
    col: column name that contains missing data in df2 
    title: title for the entire figure
    xlabel: x-axis label for all subplots
    ylabel: y-axis label for the original data subplot
    
    Returns: 
    None - displays the plot using plt.show()
    '''    
    df_missing = df2.rename(columns={col: 'missing'})
    
    columns = df_missing.loc[:, 'missing':].columns.tolist()
    subplots_size = len(columns)
    
    fig, ax = plt.subplots(subplots_size+1, 1, sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    if title:
        fig.suptitle(title)
    
    df1[col].plot(ax=ax[0], figsize=(12, 10))
    ax[0].set_title('Original Dataset')
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)    
    
    for i, colname in enumerate(columns):
        df_missing[colname].plot(ax=ax[i+1])
        ax[i+1].set_title(colname)
    
    fig.tight_layout()
    plt.show()

def rmse_score(df1, df2, col=None):
    '''
    Calculates RMSE scores between original data and multiple versions of processed data.
    
    Parameters:
    df1: original dataframe without missing data
    df2: dataframe with processed data (imputed, filled, etc.)
    col: column name in df1 to compare against processed versions in df2
         If None, the function will fail as it needs to know which column to compare
    
    Returns:
    list: RMSE scores for each processed column compared to the original
    
    Note: The function renames the column specified by 'col' to 'missing' in df2,
    then compares all columns after 'missing' with the original column from df1.
    '''
    if col is None:
        raise ValueError("Column name must be specified")
        
    df_missing = df2.rename(columns={col: 'missing'})
    
    # Get all columns starting from 'missing'
    columns = df_missing.loc[:, 'missing':].columns.tolist()
    
    if len(columns) <= 1:
        raise ValueError("No comparison columns found after the specified column")
    
    scores = []
    for comp_col in columns[1:]:
        rmse = np.sqrt(np.mean((df1[col] - df_missing[comp_col])**2))
        scores.append(rmse)
        print(f'RMSE for {comp_col}: {rmse}')
    
    return scores
