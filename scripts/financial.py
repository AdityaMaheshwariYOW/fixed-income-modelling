import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict

# NPV Function
def NPV(r, cf, t):
  """Calculate net present value of cashflows.
  Parameters
  ----------
  r : float
  Discount rate
  cf : float, jax.Array / np.array
  Cashflow amounts
  t : float, jax.Array / np.array
  Time points for cashflows (in years: days / 365)
  Notes
  -----
  cf and t should be compatible (broadcastable) shapes
  Returns
  -------
  jax.Array
  Net present value
  """
  # replacing nans with 0 - to avoid nans propagating if using jax.grad
  cf_ = jnp.where(jnp.isnan(cf), 0, cf)
  t_ = jnp.where(jnp.isnan(t), 0, t)
  # return jnp.nansum(cf * jnp.exp(-r * t))
  return jnp.sum(cf_ * jnp.exp(-r * t_))

# IRR Function

def IRR(cf: jnp.ndarray, t: jnp.ndarray, r_init: float = 0.05, max_iter: int = 50, tol: float = 1e-8) -> jnp.ndarray:
    """
    Estimate internal rate of return (IRR) using Newton–Raphson method via jax.lax.scan.
    
    Parameters
    ----------
    cf : jax.Array or np.ndarray
        Cashflow amounts (should match time array in shape)
    t : jax.Array or np.ndarray
        Time points corresponding to each cashflow (in years, e.g., days / 365)
    r_init : float, optional
        Initial guess for the IRR (default is 0.05)
    max_iter : int, optional
        Maximum number of iterations for Newton–Raphson convergence (default is 50)
    tol : float, optional
        Tolerance for convergence (not currently used explicitly, left for future extensions)
    
    Returns
    -------
    jax.Array
        Estimated continuous compounding internal rate of return (IRR)
    
    Notes
    -----
    - Uses JAX autodiff (`jax.grad`) and `lax.scan` for efficient iteration.
    - IRR is defined as the discount rate `r` such that NPV(r, cf, t) ≈ 0.
    - Assumes that a valid root exists and that the Newton method will converge.
    - Assumes all inputs are JAX-compatible arrays (e.g., `jnp.array`).
    """
    # Ensure inputs are arrays
    cf = jnp.asarray(cf)
    t = jnp.asarray(t)

    def body_fn(r, _):
        f = NPV(r, cf, t)
        df = jax.grad(NPV)(r, cf, t)
        r_new = r - f / df
        return r_new, r

    r_final, _ = lax.scan(body_fn, r_init, None, length=max_iter)
    return r_final

def compute_grouped_npv(df, group_col, cashflow_col, time_col, rates):
    """
    Compute NPVs at different discount rates grouped by a specified column, or across all data.

    Parameters
    ----------
    df : pd.DataFrame
        The input data.
    group_col : str
        Column to group by (e.g., 'company'), or "combined" to treat all rows together.
    cashflow_col : str
        The name of the column containing cashflows.
    time_col : str
        The name of the column containing time (in years).
    rates : list of float
        Discount rates to evaluate (e.g., [0.00, 0.05, 0.10]).

    Returns
    -------
    pd.DataFrame
        A pivoted DataFrame with NPVs per group and rate.
    """
    results = []

    if group_col == "combined":
        cf = jnp.array(df[cashflow_col].values)
        t = jnp.array(df[time_col].values)

        for r in rates:
            npv_val = NPV(r, cf, t)
            results.append({
                'group': 'All',
                'rate': r,
                'npv': float(npv_val)
            })

        npv_df = pd.DataFrame(results)
        return npv_df.pivot(index='group', columns='rate', values='npv').rename_axis(columns='rate (%)')

    else:
        for group in df[group_col].unique():
            subset = df[df[group_col] == group]
            cf = jnp.array(subset[cashflow_col].values)
            t = jnp.array(subset[time_col].values)

            for r in rates:
                npv_val = NPV(r, cf, t)
                results.append({
                    group_col: group,
                    'rate': r,
                    'npv': float(npv_val)
                })

        npv_df = pd.DataFrame(results)
        return npv_df.pivot(index=group_col, columns='rate', values='npv').rename_axis(columns='rate (%)')

def evaluate_npv_and_gradients(cf_array, t_array, rates):
    """
    Compute NPV and its gradient with respect to r for given rates.

    Parameters
    ----------
    cf_array : array-like
        Array of cashflows.
    t_array : array-like
        Array of time points (in years).
    rates : list of float
        Discount rates at which to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with NPV and gradient of NPV w.r.t. r at each rate.
    """
    grad_NPV = jax.grad(NPV)
    results = []

    for r in rates:
        npv_val = NPV(r, cf_array, t_array)
        grad_val = grad_NPV(r, cf_array, t_array)
        results.append({
            'rate': r,
            'npv': float(npv_val),
            'dNPV/dr': float(grad_val)
        })

    df = pd.DataFrame(results).set_index('rate')
    df.index.name = 'rate (%)'
    return df

def evaluate_irr(
    data,
    cf_col='total_cf',
    t_col='time',
    group_col='company',
    combine=False,
    change_price=False,
    new_price=1.00,
    date_col='date',
    coupon_col='coupon',
    delta_notional_col='delta_notional',
    price_col='price'
):
    """
    Compute IRR (continuous and simple) from cashflows, optionally modifying entrance price at t=0.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    cf_col : str
        Name of the column with cashflows (e.g., 'total_cf').
    t_col : str
        Name of the column with time values (in years).
    group_col : str
        Column to group by (e.g., 'company').
    combine : bool
        If True, computes IRR over all cashflows combined (ignores grouping).
    change_price : bool
        If True, modifies the entrance price at t=0 before computing IRRs.
    new_price : float
        New price to set at t=0 if change_price is True.
    date_col : str
        Column indicating dates (used for detecting t=0).
    coupon_col : str
        Coupon column used to reconstruct cashflows.
    delta_notional_col : str
        Delta notional column used to reconstruct cashflows.
    price_col : str
        Price column to override at t=0.

    Returns
    -------
    pd.DataFrame
        IRR results (per group or combined).
    pd.DataFrame (optional)
        Modified DataFrame (only returned if `change_price=True`)
    """

    df_mod = data.copy()

    # Optionally override price at t=0 and recompute cashflows
    if change_price:
        min_date = df_mod[date_col].min()
        mask_t0 = df_mod[date_col] == min_date

        df_mod.loc[mask_t0, price_col] = new_price
        df_mod.loc[mask_t0, cf_col] = (
            df_mod.loc[mask_t0, coupon_col] +
            df_mod.loc[mask_t0, delta_notional_col] * df_mod.loc[mask_t0, price_col]
        )
    else:
        df_mod = df_mod.copy()

    # Compute IRRs
    if combine:
        cf = jnp.array(df_mod[cf_col].values)
        t = jnp.array(df_mod[t_col].values)

        r_cont = IRR(cf, t)
        r_simple = jnp.exp(r_cont) - 1

        result_df = pd.DataFrame([{
            'group': 'Combined',
            'IRR (continuous)': float(r_cont),
            'IRR (simple)': float(r_simple)
        }]).set_index('group')
    else:
        irr_results = []
        for group, df_grp in df_mod.groupby(group_col):
            cf = jnp.array(df_grp[cf_col].values)
            t = jnp.array(df_grp[t_col].values)

            r_cont = IRR(cf, t)
            r_simple = jnp.exp(r_cont) - 1

            irr_results.append({
                group_col: group,
                'IRR (continuous)': float(r_cont),
                'IRR (simple)': float(r_simple)
            })

        result_df = pd.DataFrame(irr_results).set_index(group_col)

    return (result_df, df_mod) if change_price else result_df

