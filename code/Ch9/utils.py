"""
Utility functions for Chapter 9: Time Series Decomposition
"""

import pandas as pd
from pathlib import Path
from statsmodels.datasets import co2, get_rdataset
import matplotlib.pyplot as plt


def load_ch9_datasets(base_path='../../datasets/Ch9'):
    """
    Load and return the three time series datasets used in Chapter 9.

    Parameters
    ----------
    base_path : str, default '../../datasets/Ch9'
        Base folder path where the CSV files are stored.

    Returns
    -------
    airp_df : pandas.DataFrame
        Monthly airline passengers.
    closing_price : pandas.DataFrame
        Daily closing prices.
    co2_df : pandas.DataFrame
        Weekly atmospheric CO2 concentrations.
    """
    base = Path(base_path)

    # Load closing prices
    closing_price = pd.read_csv(
        base / 'closing_price.csv',
        index_col='Date',
        parse_dates=True
    )

    # Load and clean CO2 dataset
    co2_df = co2.load_pandas().data
    co2_df = co2_df.ffill()

    # Load and prepare AirPassengers dataset
    air_passengers = get_rdataset("AirPassengers")
    airp_df = air_passengers.data
    airp_df.index = pd.date_range('1949', '1961', freq='ME')
    airp_df.drop(columns=['time'], inplace=True)
    airp_df.rename(columns={'value': 'passengers'}, inplace=True)

    return airp_df, closing_price, co2_df


def plot_comparison(methods, kpss_results, adf_results, plot_type='line'):
    """
    Plot transformed series and summarize stationarity tests.

    Parameters
    ----------
    methods : list of pandas.Series
        List of transformed time series to compare.
    kpss_results : callable
        Function that takes a Series and returns a dict-like object
        with a 'Decision' field for the KPSS test.
    adf_results : callable
        Function that takes a Series and returns a dict-like object
        with a 'Decision' field for the ADF test.
    plot_type : {'line', 'hist'}, default 'line'
        Type of plot to use for each transformed series.

    Returns
    -------
    None
        Displays a grid of subplots comparing the transformations.
    """
    n = len(methods) // 2 + len(methods) % 2  # rows needed for subplots
    fig, ax = plt.subplots(n, 2, sharex=True, figsize=(20, 10))
    ax = ax.flatten()

    for i, method in enumerate(methods):
        series = method.dropna()  # ensure no NaNs

        # Derive a label for the series
        name = getattr(series, "name", f"method_{i+1}")

        # Perform KPSS and ADF tests
        kpss_result = kpss_results(series)
        adf_result = adf_results(series)

        kpss_decision = kpss_result['Decision']
        adf_decision = adf_result['Decision']

        # Plot series
        series.plot(
            kind=plot_type,
            ax=ax[i],
            legend=False,
            title=(
                f"Method={name}, "
                f"KPSS={kpss_decision}, "
                f"ADF={adf_decision}"
            ),
        )
        ax[i].title.set_size(14)

        # Add rolling mean (52-week window)
        if plot_type == 'line':
            series.rolling(52).mean().plot(ax=ax[i], legend=False)

    # Remove any unused axes
    for j in range(i + 1, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    plt.show()