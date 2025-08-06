import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict


# Read Data

def read_and_parse_dates(path: str, date_col: str = 'date') -> pd.DataFrame:
    """
    Reads a CSV or Excel file and converts the date column to datetime.

    Parameters
    ----------
    path : str
        Path to the file (CSV or Excel)
    date_col : str
        Column name to parse as datetime

    Returns
    -------
    pd.DataFrame
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xlsx') or path.endswith('.xls'):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    df[date_col] = pd.to_datetime(df[date_col])
    return df

def add_time_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Adds a 'time' column calculated as (date - min_date) / 365.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing a datetime column

    Returns
    -------
    pd.DataFrame
        Updated dataframe with 'time' column
    """
    min_date = df[date_col].min()
    df['time'] = (df[date_col] - min_date).dt.days / 365
    return df

def check_cashflow_consistency(df: pd.DataFrame,
                                cf_col: str = 'total_cf',
                                coupon_col: str = 'coupon',
                                delta_col: str = 'delta_notional',
                                price_col: str = 'price',
                                tol: float = 1e-6) -> None:
    """
    Checks if total cashflow equals coupon + delta_notional * price.

    Parameters
    ----------
    df : pd.DataFrame
    cf_col : str
        Column with recorded total cashflows
    coupon_col : str
    delta_col : str
    price_col : str
    tol : float
        Tolerance for np.isclose comparison

    Returns
    -------
    None
    """
    df['calculated_cf'] = df[coupon_col] + df[delta_col] * df[price_col]
    df['cf_match'] = np.isclose(df['calculated_cf'], df[cf_col], atol=tol)

    mismatches = df[~df['cf_match']]
    print(f"Number of mismatches: {len(mismatches)}")

    if not mismatches.empty:
        display(mismatches[['company', 'date', coupon_col, delta_col, price_col, cf_col, 'calculated_cf']])

# Plotting Functions
def plot_grouped_time_series(df, group_col, time_col, value_cols, 
                             title_prefix="Group", ylabel="Value", figsize_per_group=4):
    """
    Plot time series data for each group in a separate subplot.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    group_col : str
        Column to group by (e.g. 'company').
    time_col : str
        Column representing the time axis (e.g. 'time').
    value_cols : list of str
        Columns to plot as separate lines (e.g. ['total_cf', 'coupon', 'delta_notional']).
    title_prefix : str
        Prefix for subplot titles (e.g. 'Company' will show 'Company A', 'Company B', etc.)
    ylabel : str
        Label for the y-axis.
    figsize_per_group : float
        Vertical size per group in inches.
    """
    groups = df[group_col].unique()
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, figsize_per_group * len(groups)), sharex=True)

    if len(groups) == 1:
        axes = [axes]

    for i, group in enumerate(groups):
        sub_df = df[df[group_col] == group]
        ax = axes[i]

        for col in value_cols:
            ax.plot(sub_df[time_col], sub_df[col], label=col.replace('_', ' ').title(), marker='o')

        ax.set_title(f"{title_prefix} {group}")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

