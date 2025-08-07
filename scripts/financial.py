import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import lax, grad
from typing import Union, Tuple, List

# -------------------------
# Core Financial Functions
# -------------------------

def NPV(r: float, cf: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Net Present Value (NPV) of a stream of cashflows.

    Parameters
    ----------
    r : float
        Discount rate (continuously compounded).
    cf : jnp.ndarray
        Array of cashflows.
    t : jnp.ndarray
        Array of time points corresponding to the cashflows (in years).

    Returns
    -------
    jnp.ndarray
        The computed NPV value.
    """
    cf_ = jnp.where(jnp.isnan(cf), 0, cf)
    t_ = jnp.where(jnp.isnan(t), 0, t)
    return jnp.sum(cf_ * jnp.exp(-r * t_))

import jax
import jax.numpy as jnp
from jax import lax


def NPV_simple(r: float, cf: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    Discrete compounding NPV: sum(cf_i / (1+r)**t_i)
    """
    cf_ = jnp.where(jnp.isnan(cf), 0, cf)
    t_ = jnp.where(jnp.isnan(t), 0, t)
    return jnp.sum(cf_ / (1 + r) ** t_)


def compute_grouped_npv(df, group_col, cashflow_col, time_col, rates):
    """
    Computes NPV for each group (or for all combined) across different discount rates.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing cashflows.
    group_col : str
        Column name to group by (e.g., 'company' or 'combined').
    cashflow_col : str
        Column containing cashflow values.
    time_col : str
        Column containing time indices (usually integers).
    rates : list of float
        Discount rates to test.

    Returns
    -------
    pd.DataFrame
        DataFrame of NPVs for each group and rate.
    """
    results = []
    
    if group_col == 'combined':
        group_data = [('combined', df)]
    else:
        group_data = df.groupby(group_col)

    for rate in rates:
        for label, group in group_data:
            times = jnp.array(group[time_col].values)
            cfs = jnp.array(group[cashflow_col].values)
            npv_val = NPV(rate, cfs, times)  # Use continuous compounding
            results.append({'group': label, 'rate': rate, 'npv': float(npv_val)})
    
    return pd.DataFrame(results)

def evaluate_npv_and_gradients(cf: jnp.ndarray, t: jnp.ndarray, rates: list[float]) -> pd.DataFrame:
    """
    Compute the Net Present Value (NPV) and its gradient with respect to the discount rate
    for a given stream of cashflows at specified times.

    This is useful for assessing the sensitivity of NPV to changes in the discount rate—
    for example, for understanding duration in bond analytics.

    Parameters
    ----------
    cf : jnp.ndarray
        Array of cashflows. Should be 1D and correspond to the same length as `t`.
    t : jnp.ndarray
        Array of time indices (e.g., in years, quarters, etc.) corresponding to each cashflow.
    rates : list of float
        List of discount rates at which to evaluate the NPV and its derivative.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'rate': The discount rate
        - 'npv': Net Present Value at that rate
        - 'd_npv_dr': Derivative of NPV with respect to rate (i.e., sensitivity/duration proxy)

    Example
    -------
    >>> cf = jnp.array([-1000, 400, 400, 400])
    >>> t = jnp.array([0, 1, 2, 3])
    >>> rates = [0.00, 0.05, 0.10]
    >>> evaluate_npv_and_gradients(cf, t, rates)
    """
    npv_fn = lambda r: NPV(r, cf, t)
    npv_grad = grad(npv_fn)

    results = []
    for r in rates:
        npv_val = npv_fn(r)
        grad_val = npv_grad(r)
        results.append({
            "rate": r,
            "npv": float(npv_val),
            "d_npv_dr": float(grad_val)
        })

    return pd.DataFrame(results)

def irr_newton(
    cf: jnp.ndarray,
    t: jnp.ndarray,
    r_init: float = 0.05,
    max_iter: int = 50,
    compounding: str = 'continuous'
) -> float:
    """
    Compute the internal rate of return (IRR) using Newton-Raphson.

    Parameters
    ----------
    cf : jnp.ndarray
        Cashflow amounts.
    t : jnp.ndarray
        Time points corresponding to each cashflow.
    r_init : float, optional
        Initial guess for the IRR (default 0.05).
    max_iter : int, optional
        Max Newton-Raphson iterations (default 50).
    compounding : {'simple', 'continuous'}
        'simple' for discrete compounding (R),
        'continuous' for continuously compounded rate (r).

    Returns
    -------
    float
        Estimated IRR (simple R or continuous r).
    """
    # Validate compounding
    if compounding not in ('simple', 'continuous'):
        raise ValueError("compounding must be 'simple' or 'continuous'")

    def body_fn(r, _):
        if compounding == 'simple':
            f = NPV_simple(r, cf, t)
            df = jax.grad(NPV_simple)(r, cf, t)
        else:
            f = NPV(r, cf, t)
            df = jax.grad(NPV)(r, cf, t)
        r_new = r - f / df
        return r_new, r

    # Run Newton-Raphson via lax.scan
    r_final, _ = lax.scan(body_fn, r_init, None, length=max_iter)
    return float(r_final)  # return Python float

def irr_simulated_batch(cashflow_matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Compute IRRs for a batch of simulated cashflow paths using vectorized bisection.

    Parameters
    ----------
    cashflow_matrix : jnp.ndarray
        Matrix of shape (n_simulations, n_periods) where each row is a set of cashflows.

    Returns
    -------
    jnp.ndarray
        Array of IRRs, one per simulation.
    """
    def npv(rate, cf):
        times = jnp.arange(cf.shape[0])
        return jnp.sum(cf / (1 + rate) ** times)

    def single_irr(cf):
        def cond(state):
            low, high, i = state
            return (i < 100) & (high - low > 1e-6)

        def body(state):
            low, high, i = state
            mid = (low + high) / 2
            mid_npv = npv(mid, cf)
            return jax.lax.cond(mid_npv > 0,
                                 lambda _: (mid, high, i + 1),
                                 lambda _: (low, mid, i + 1),
                                 operand=None)

        low_final, high_final, _ = lax.while_loop(cond, body, (-0.9999, 1.0, 0))
        return (low_final + high_final) / 2

    return jax.vmap(single_irr)(cashflow_matrix)

def expected_irr_given_price(sim_results: dict, p0: float) -> float:
    """
    Estimate the expected IRR across all simulations and companies, given an entry price.

    Parameters
    ----------
    sim_results : dict
        Dictionary with simulated results, each containing a 'cf_mat' (cashflow matrix).
    p0 : float
        Price to use for time-zero investment (overwriting initial cash outlay).

    Returns
    -------
    float
        Mean IRR across all simulation paths.
    """
    cf_list = []
    for v in sim_results.values():
        cf_adj = v['cf_mat'].at[0].set(-p0)  # set time-zero outlay
        cf_list.append(cf_adj.T)  # reshape to (n_sims, n_periods)

    all_cf = jnp.sum(jnp.stack(cf_list), axis=0)  # sum over companies
    return float(jnp.mean(irr_simulated_batch(all_cf)))

def compute_no_default_irr(cf: jnp.ndarray, times: jnp.ndarray, p0: float) -> float:
    """
    Compute IRR under the assumption of no default (i.e., using total deterministic cashflows).

    Parameters
    ----------
    cf : jnp.ndarray
        Cashflows (e.g., company-level or summed across companies).
    times : jnp.ndarray
        Time points for each cashflow.
    p0 : float
        Price at time 0 (initial investment).

    Returns
    -------
    float
        Estimated IRR (continuous compounding).
    """
    full_cf = cf.at[0].set(-p0)
    return irr_newton(full_cf, times)

def compute_irr_from_dataframe(
    df: pd.DataFrame,
    combine: bool = False,
    change_price: bool = False,
    new_price: float = 1.0
) -> Union[
    Tuple[float, float],
    Tuple[Tuple[float, float], pd.DataFrame],
    Dict[str, Tuple[float, float]],
    Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]
]:
    """
    Compute both discrete (simple) and continuous IRR(s) from a DataFrame containing cashflows and times.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'company', 'total_cf', and 'time' columns.
    combine : bool, optional
        If True, combines all cashflows across companies and returns a single IRR pair.
    change_price : bool, optional
        If True, overrides the price at time 0 with new_price.
    new_price : float, optional
        The new price to use at time 0 (only used if change_price is True).

    Returns
    -------
    Union[
        (R_simple, r_cont),
        ((R_simple, r_cont), df),
        {company: (R_simple, r_cont)},
        ({company: (R_simple, r_cont)}, df)
    ]
        Simple and continuous IRRs, and optionally the modified DataFrame.
    """
    df = df.copy()

    # Apply price override if requested
    if change_price:
        t0 = df['time'].min()
        mask = df['time'] == t0
        df.loc[mask, 'total_cf'] = df.loc[mask, 'delta_notional'] * new_price + df.loc[mask, 'coupon']

    def calc_both(cf_arr, t_arr):
        R = irr_newton(cf_arr, t_arr, compounding='simple')
        r = irr_newton(cf_arr, t_arr, compounding='continuous')
        return (R, r)

    if combine:
        cf = jnp.array(df['total_cf'].values)
        t = jnp.array(df['time'].values)
        irr_pair = calc_both(cf, t)
        return (irr_pair, df) if change_price else irr_pair

    irr_by_company: Dict[str, Tuple[float, float]] = {}
    for name, subdf in df.groupby('company'):
        cf = jnp.array(subdf['total_cf'].values)
        t = jnp.array(subdf['time'].values)
        irr_by_company[name] = calc_both(cf, t)

    return (irr_by_company, df) if change_price else irr_by_company


def expected_irr_given_price(sim_results: dict, p0: float) -> float:
    """
    Estimate the expected IRR across all simulations and companies, given an entry price.

    Parameters
    ----------
    sim_results : dict
        Dictionary with simulated results, each containing a 'cf_mat' (jax array of shape
        (n_simulations, n_periods)).
    p0 : float
        Price to use for time-zero investment (overwriting initial cashflow at t=0).

    Returns
    -------
    float
        Mean IRR across all simulation paths.
    """
    cf_list = []
    for v in sim_results.values():
        cf_mat = v['cf_mat']                     # shape (n_sims, n_periods)
        cf_adj = cf_mat.at[:, 0].set(-p0)        # overwrite t=0 cashflow
        cf_list.append(cf_adj)

    # Stack across companies → shape (n_companies, n_sims, n_periods)
    stacked = jnp.stack(cf_list, axis=0)
    # Sum over companies → shape (n_sims, n_periods)
    total_cf = jnp.sum(stacked, axis=0)

    # Compute IRR per simulation and average
    irr_array = irr_simulated_batch(total_cf)
    irr_array = np.array(irr_array)
    irr_array = irr_array[~np.isnan(irr_array)]
    return float(np.mean(irr_array))
