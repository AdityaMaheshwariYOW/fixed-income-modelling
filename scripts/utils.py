import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def read_and_parse_dates(path: str, date_col: str = 'date') -> pd.DataFrame:
    """
    Reads a CSV or Excel file and parses a specified column as datetime.

    Parameters
    ----------
    path : str
        Path to the file (supports .csv, .xls, .xlsx)
    date_col : str
        Column to parse as datetime

    Returns
    -------
    pd.DataFrame
        Parsed dataframe with datetime column

    Raises
    ------
    ValueError
        If file extension is not supported
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
    Adds a 'time' column to a dataframe, where time is measured
    in years from the earliest date in `date_col`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a datetime column
    date_col : str
        Column name containing datetime objects

    Returns
    -------
    pd.DataFrame
        DataFrame with an added 'time' column (float, years)
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
    print(f"Number of mismatches in cashflow: {len(mismatches)}")

    if not mismatches.empty:
        print(mismatches[['company', 'date', coupon_col, delta_col, price_col, cf_col, 'calculated_cf']])


def plot_grouped_time_series(df: pd.DataFrame, group_col: str, time_col: str,
                             value_cols: List[str], title_prefix: str = "Group",
                             ylabel: str = "Value", figsize_per_group: float = 4.0) -> None:
    """
    Plots time series for each group in a separate subplot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with grouped time series data
    group_col : str
        Column to group by
    time_col : str
        Column for x-axis (time)
    value_cols : list of str
        Columns to plot on y-axis
    title_prefix : str
        Prefix for subplot titles
    ylabel : str
        Label for y-axis
    figsize_per_group : float
        Vertical size per subplot

    Returns
    -------
    None
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
