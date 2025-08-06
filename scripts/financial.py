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
            npv = jnp.sum(cfs / (1 + rate) ** times)
            results.append({'group': label, 'rate': rate, 'npv': float(npv)})
    
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

def irr_newton(cf: jnp.ndarray, t: jnp.ndarray, r_init: float = 0.05, max_iter: int = 50) -> jnp.ndarray:
    """
    Compute the internal rate of return (IRR) using the Newton-Raphson method.

    Parameters
    ----------
    cf : jnp.ndarray
        Cashflow amounts.
    t : jnp.ndarray
        Time points corresponding to each cashflow (in years).
    r_init : float, optional
        Initial guess for the IRR (default is 0.05).
    max_iter : int, optional
        Maximum number of iterations (default is 50).

    Returns
    -------
    jnp.ndarray
        Estimated IRR using continuous compounding.
    """
    def body_fn(r, _):
        f = NPV(r, cf, t)
        df = jax.grad(NPV)(r, cf, t)
        r_new = r - f / df
        return r_new, r

    r_final, _ = lax.scan(body_fn, r_init, None, length=max_iter)
    return r_final

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
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, pd.DataFrame]]:
    """
    Compute IRR(s) from a DataFrame containing cashflows and times.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'company', 'total_cf', and 'time' columns.
    combine : bool, optional
        If True, combines all cashflows across companies and returns a single IRR.
        If False, returns IRR for each company separately.
    change_price : bool, optional
        If True, overrides the price at time 0 with new_price.
    new_price : float, optional
        The new price to use at time 0 (only used if change_price is True).

    Returns
    -------
    Union[jnp.ndarray, Tuple[jnp.ndarray, pd.DataFrame]]
        If combine=True: returns a single IRR.
        If combine=False: returns a list of IRRs and the modified DataFrame.
    """
    df = df.copy()

    if change_price:
        t0 = df['time'].min()
        mask = df['time'] == t0
        df.loc[mask, 'total_cf'] = df.loc[mask, 'delta_notional'] * new_price + df.loc[mask, 'coupon']

    if combine:
        cf = jnp.array(df['total_cf'].values)
        t = jnp.array(df['time'].values)
        irr = irr_newton(cf, t)
        return irr, df if change_price else irr

    else:
        irr_by_group = {}
        for name, subdf in df.groupby('company'):
            cf = jnp.array(subdf['total_cf'].values)
            t = jnp.array(subdf['time'].values)
            irr_by_group[name] = irr_newton(cf, t)
        return irr_by_group, df if change_price else irr_by_group

def expected_irr_given_price_empirical(data: pd.DataFrame, p0: float, N_SIMULATIONS: int = 10000) -> float:
    """
    Simulate portfolio cashflows given an entry price p0, and compute the expected IRR.

    Parameters
    ----------
    p0 : float
        Entry price to set at time 0 for all companies.

    Returns
    -------
    float
        Average IRR across all simulation paths.
    """
    sim_res = {}

    for company in data['company'].unique():
        df_sub = data[data['company'] == company].copy()
        n_periods = df_sub.shape[0]
        t0 = df_sub['date'].min()
        times = (df_sub['date'] - t0).dt.days.values / 365.0

        # Overwrite price at time 0
        df_sub.loc[df_sub.index[0], 'price'] = p0
        coupon = df_sub['coupon'][df_sub['coupon'] > 0].iloc[0]
        delta_notional = df_sub['delta_notional'].iloc[0]

        sim_res[company] = run_simulations_for_company(
            company=company,
            coupon=coupon,
            delta_notional=delta_notional,
            price=p0,
            n_periods=n_periods,
            n_simulations=N_SIMULATIONS,
            pd=PD
        )
        sim_res[company]['times'] = jnp.array(times)

    # Pad and aggregate cashflows across companies
    max_periods = max(v['cf_mat'].shape[1] for v in sim_res.values())
    cf_list = []
    for v in sim_res.values():
        cf = v["cf_mat"]
        pad_width = max_periods - cf.shape[1]
        cf_padded = jnp.pad(cf, ((0, 0), (0, pad_width)))
        cf_padded = cf_padded.at[:, 0].set(-p0)
        cf_list.append(cf_padded)

    total_cf = jnp.sum(jnp.stack(cf_list), axis=0)  # shape: (n_sims, max_periods)

    # Compute IRR per simulation
    irr_array = irr_simulated_batch(total_cf)
    irr_array = np.array(irr_array)
    irr_array = irr_array[~np.isnan(irr_array)]

    return irr_array.mean()
