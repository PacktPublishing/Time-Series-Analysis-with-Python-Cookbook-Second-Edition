import matplotlib.pyplot as plt

def plot_spectrum_with_peaks(period, power_spectrum, peaks_df,
                             title='Power Spectrum with Detected Cycles',
                             xlabel='Period (hours)',
                             ylabel='Power (spectrum)',
                             figsize=(10, 5)):
    """
    Plot power spectrum with annotated peaks.
    
    Parameters:
    -----------
    period : array-like
        Period values corresponding to frequencies
    power_spectrum : array-like
        Power spectral density values
    peaks_df : pd.DataFrame
        DataFrame with columns: 'period_hours', 'labels', 'color'
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    figsize : tuple, optional
        Figure size (default: (10, 5))
    
    Returns:
    --------
    tuple of (fig, ax)
        Matplotlib figure and axis objects
    
    Example:
    --------
    >>> plot_spectrum_with_peaks(period_hours, P_pos, top_peaks,
    ...                          title='Periodogram of Hourly Energy Load')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot spectrum
    ax.plot(period, power_spectrum, lw=2, color='steelblue', alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Annotate peaks
    y_max = ax.get_ylim()[1]
    for _, row in peaks_df.iterrows():
        # Vertical line at peak
        ax.axvline(
            x=row['period_hours'],
            color=row['color'],
            ls='--',
            linewidth=2,
            alpha=0.7
        )
        # Text label
        ax.text(
            x=row['period_hours'],
            y=y_max * 0.9,
            s=row['labels'],
            rotation=90,
            va='top',
            ha='right',
            fontsize=10,
            color=row['color']
        )
    
    ax.grid(True, alpha=0.3, which='both', ls=':')
    plt.tight_layout()
    plt.show()
    
    return fig, ax