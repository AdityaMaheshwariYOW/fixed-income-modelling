import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import lax, grad
from typing import Union, Tuple, List, Dict

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
    return r_final  # return Python float

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

def yearly_default_probs(pd: float, years: int = 4) -> List[float]:
    """
    Return [p(default in Year 1), ..., p(default in Year N)] under an i.i.d. annual PD.
    No globals required.

    Args:
        pd: annual probability of default (e.g., 0.02 for 2%)
        years: number of years in the horizon

    Returns:
        List of length `years` with year-by-year default probabilities.
    """
    if years <= 0:
        return []
    if not (0.0 <= pd <= 1.0):
        raise ValueError("pd must be in [0, 1].")
    return [(1.0 - pd) ** (k - 1) * pd for k in range(1, years + 1)]


def build_cashflows_for_company(
    company_name: str,
    companies: Dict[str, Dict[str, float]],
    price: float,
    *,
    par: float = 100.0,
    maturity_years: int = 4,
    per_year: int = 4,
    delay_quarters: int = 0,
    recovery_lag_years: float = 0.5,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Build scenario cashflows for a single company, with optional deployment delay.

    Scenarios:
      - Default at Year 1, 2, 3, 4 (recovery at year + recovery_lag_years from the t=0 clock)
      - No Default (par repaid at maturity_years)

    Deployment delay (fund clock):
      - Initial outlay occurs at t = delay_quarters / per_year
      - Coupons start the *quarter after* purchase
      - Default/maturity timings DO NOT shift (they’re anchored to t=0)

    Returns
    -------
    dict[str, (cf: jnp.ndarray, t: jnp.ndarray)]
        Keys: "Year 1", …, "Year N", "No Default".
        cf and t share the same length and quarterly grid (including recovery points).
    """
    if company_name not in companies:
        raise KeyError(f"company '{company_name}' not found in companies dict")

    if per_year <= 0:
        raise ValueError("per_year must be > 0")
    if maturity_years <= 0:
        raise ValueError("maturity_years must be > 0")
    if delay_quarters < 0:
        raise ValueError("delay_quarters must be >= 0")
    if recovery_lag_years < 0:
        raise ValueError("recovery_lag_years must be >= 0")

    params   = companies[company_name]
    cpn_rate = float(params["Coupon"]) / 100.0
    rr       = float(params["RR"])

    total_periods     = int(maturity_years * per_year)              # coupon/par grid to maturity
    recovery_periods  = int((maturity_years + recovery_lag_years) * per_year)  # to last recovery
    dt                = 1.0 / per_year
    t                 = jnp.arange(0, recovery_periods + 1) * dt    # 0, 1/per_year, ..., maturity+lag

    qtr_coupon        = (cpn_rate * par) / per_year
    scenarios: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]] = {}

    # --- No Default ---
    cf_nd = jnp.zeros_like(t)
    # Outlay at deployment time (fund clock)
    deploy_idx = min(delay_quarters, recovery_periods)  # guard if delay beyond grid
    cf_nd = cf_nd.at[deploy_idx].set(-price)

    # Coupons: from the quarter AFTER purchase up to and including the last coupon quarter
    first_coupon_qtr = delay_quarters + 1
    if first_coupon_qtr <= total_periods:
        cf_nd = cf_nd.at[first_coupon_qtr:total_periods + 1].add(qtr_coupon)

    # Par at maturity (unchanged by delay)
    cf_nd = cf_nd.at[total_periods].add(par)
    scenarios["No Default"] = (cf_nd, t)

    # --- Default in Year k ---
    for year in range(1, maturity_years + 1):
        cf = jnp.zeros_like(t)

        # Outlay at deployment
        cf = cf.at[deploy_idx].set(-price)

        # Coupons between purchase and the end of default year (inclusive)
        last_qtr     = year * per_year
        start_q      = delay_quarters + 1
        end_q        = min(last_qtr, total_periods)
        if start_q <= end_q:
            cf = cf.at[start_q:end_q + 1].add(qtr_coupon)

        # Recovery at k + recovery_lag_years (anchored to t=0)
        recovery_qtr = int((year + recovery_lag_years) * per_year)
        recovery_qtr = min(recovery_qtr, recovery_periods)  # guard
        cf = cf.at[recovery_qtr].add(rr * par)

        scenarios[f"Year {year}"] = (cf, t)

    return scenarios


# Safe IRR wrapper: no defaults, no globals
def safe_irr(
    cf: jnp.ndarray,
    t: jnp.ndarray,
    compounding: str,
    r_init: float
) -> float:
    """
    Newton IRR with basic sanity checks. Returns np.nan if it fails.
    Requires explicit compounding ('simple'|'continuous') and r_init.
    """
    try:
        r = float(irr_newton(cf, t, r_init=r_init, compounding=compounding))
        if not np.isfinite(r) or r <= -0.999:
            return np.nan
        return r
    except Exception:
        return np.nan


# Scenario IRRs (in %) for one company/price with explicit config
def irr_series_for_company(
    company_name: str,
    companies: Dict[str, Dict[str, float]],
    price: float,
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    compounding: str,
    r_init: float
) -> pd.Series:
    """
    Returns a Series of IRRs (in %) for scenarios: Year 1..maturity_years, No Default.
    Uses fund clock (deploy at delay_quarters; default/maturity anchored to t=0).
    """
    scen = build_cashflows_for_company(
        company_name=company_name,
        companies=companies,
        price=price,
        par=par,
        maturity_years=maturity_years,
        per_year=per_year,
        delay_quarters=delay_quarters,
        recovery_lag_years=recovery_lag_years,
    )

    labels = [f"Year {i}" for i in range(1, maturity_years + 1)] + ["No Default"]
    out = {}
    for k in labels:
        cf, t = scen[k]
        r = safe_irr(cf, t, compounding=compounding, r_init=r_init)
        out[k] = 100.0 * r  # percentage
    return pd.Series(out)


# Tables for all companies across a list of prices 
def irr_tables_all_prices(
    companies_dict: Dict[str, Dict[str, float]],
    price_list: List[float],
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    compounding: str,
    r_init: float
) -> Dict[str, pd.DataFrame]:
    """
    For each company: DataFrame indexed by (Year 1..N, No Default), columns = price_list, values = IRR (%).
    """
    idx = [f"Year {i}" for i in range(1, maturity_years + 1)] + ["No Default"]
    out: Dict[str, pd.DataFrame] = {}

    for company in companies_dict.keys():
        cols = {}
        for p in price_list:
            s = irr_series_for_company(
                company_name=company,
                companies=companies_dict,
                price=p,
                par=par,
                maturity_years=maturity_years,
                per_year=per_year,
                delay_quarters=delay_quarters,
                recovery_lag_years=recovery_lag_years,
                compounding=compounding,
                r_init=r_init
            )
            cols[f"{p:.0f}"] = s
        df = pd.DataFrame(cols).loc[idx]
        out[company] = df
    return out


# Long/tidy DF for plotting (maps 'No Default' to 'Year {maturity_years+1}')
def irr_long_dataframe(
    companies_dict: Dict[str, Dict[str, float]],
    price_list: List[float],
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    compounding: str,
    r_init: float
) -> pd.DataFrame:
    """
    Columns: Company, Price, Year, YearPlot, IRR (%).
    YearPlot maps 'No Default' -> 'Year {maturity_years+1}' for consistent x-axis spacing.
    """
    rows = []
    nd_label = f"Year {maturity_years + 1}"
    for company in companies_dict.keys():
        for p in price_list:
            s = irr_series_for_company(
                company_name=company,
                companies=companies_dict,
                price=p,
                par=par,
                maturity_years=maturity_years,
                per_year=per_year,
                delay_quarters=delay_quarters,
                recovery_lag_years=recovery_lag_years,
                compounding=compounding,
                r_init=r_init
            )
            for year_label, irr_val in s.items():
                year_plot = nd_label if year_label == "No Default" else year_label
                rows.append({
                    "Company": company,
                    "Price": f"{p:.0f}",
                    "Year": year_label,
                    "YearPlot": year_plot,
                    "IRR (%)": irr_val
                })

    df = pd.DataFrame(rows)
    cat_order = [f"Year {i}" for i in range(1, maturity_years + 2)]  # include mapped No Default
    df["YearPlot"] = pd.Categorical(df["YearPlot"], categories=cat_order, ordered=True)
    return df

def expected_irr(
    company_name: str,
    companies: Dict[str, Dict[str, float]],
    price: float,
    *,
    maturity_years: int,
    compounding: str,
    # args needed by irr_series_for_company:
    par: float,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    r_init: float
) -> float:
    """
    Expected IRR (% per year) for one company at a given price.
    Uses yearly_default_probs + survival; matches scenario labels used by irr_series_for_company.
    """
    pd_val = float(companies[company_name]["PD"])
    year_probs = yearly_default_probs(pd_val, years=maturity_years)      # [p1..pN]
    labels = [f"Year {i}" for i in range(1, maturity_years + 1)] + ["No Default"]
    probs = {f"Year {i+1}": p for i, p in enumerate(year_probs)}
    probs["No Default"] = (1 - pd_val) ** maturity_years

    irr_vals = irr_series_for_company(
        company_name=company_name,
        companies=companies,
        price=price,
        par=par,
        maturity_years=maturity_years,
        per_year=per_year,
        delay_quarters=delay_quarters,
        recovery_lag_years=recovery_lag_years,
        compounding=compounding,
        r_init=r_init
    )  # returns % by scenario

    return float(sum(probs[lbl] * irr_vals[lbl] for lbl in labels))


def expected_irr_table(
    companies_dict: Dict[str, Dict[str, float]],
    price_list: List[float],
    *,
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    compounding: str,
    r_init: float
) -> pd.DataFrame:
    rows = []
    for company in companies_dict.keys():
        for p in price_list:
            val = expected_irr(
                company_name=company,
                companies=companies_dict,
                price=p,
                maturity_years=maturity_years,
                compounding=compounding,
                par=par,
                per_year=per_year,
                delay_quarters=delay_quarters,
                recovery_lag_years=recovery_lag_years,
                r_init=r_init
            )
            rows.append({"Company": company, "Price": float(p), "Expected IRR (%)": float(val)})
    df = pd.DataFrame(rows)
    df["Price"] = pd.Categorical(df["Price"], categories=sorted(price_list), ordered=True)
    return df

def _expand_bracket_for_price(
    company: str,
    companies_dict: Dict[str, Dict[str, float]],
    target_pct: float,
    compounding: str,
    p_lo_init: float,
    p_hi_init: float,
    *,
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    r_init: float,
    max_expansions: int = 12
) -> Tuple[float, float]:
    """
    Expand [p_lo, p_hi] until f(p_lo)*f(p_hi) <= 0 or expansions exhausted.
    f(P) = E[IRR](P) - target_pct
    """
    p_lo, p_hi = p_lo_init, p_hi_init
    f_lo = expected_irr(company, companies_dict, p_lo,
                        maturity_years=maturity_years, compounding=compounding,
                        par=par, per_year=per_year,
                        delay_quarters=delay_quarters,
                        recovery_lag_years=recovery_lag_years,
                        r_init=r_init) - target_pct

    f_hi = expected_irr(company, companies_dict, p_hi,
                        maturity_years=maturity_years, compounding=compounding,
                        par=par, per_year=per_year,
                        delay_quarters=delay_quarters,
                        recovery_lag_years=recovery_lag_years,
                        r_init=r_init) - target_pct

    expansions = 0
    while not (np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0) and expansions < max_expansions:
        p_lo = max(1.0, p_lo * 0.8)   # sanity guard: price ≥ $1
        p_hi = p_hi * 1.25

        f_lo = expected_irr(company, companies_dict, p_lo,
                            maturity_years=maturity_years, compounding=compounding,
                            par=par, per_year=per_year,
                            delay_quarters=delay_quarters,
                            recovery_lag_years=recovery_lag_years,
                            r_init=r_init) - target_pct

        f_hi = expected_irr(company, companies_dict, p_hi,
                            maturity_years=maturity_years, compounding=compounding,
                            par=par, per_year=per_year,
                            delay_quarters=delay_quarters,
                            recovery_lag_years=recovery_lag_years,
                            r_init=r_init) - target_pct

        expansions += 1

    return p_lo, p_hi


def price_for_target_expected_irr(
    company: str,
    companies_dict: Dict[str, Dict[str, float]],
    *,
    target_pct: float = 10.0,
    compounding: str = "simple",
    p_lo: float = 60.0,
    p_hi: float = 120.0,
    tol: float = 1e-4,
    max_iter: int = 100,
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    r_init: float
) -> float:
    """
    Solve for the price (OID, % of par) that gives expected IRR == target_pct.
    Uses bisection with automatic bracketing expansion.
    """
    # Expand bracket first
    p_lo, p_hi = _expand_bracket_for_price(
        company, companies_dict, target_pct, compounding,
        p_lo, p_hi,
        par=par, maturity_years=maturity_years,
        per_year=per_year, delay_quarters=delay_quarters,
        recovery_lag_years=recovery_lag_years, r_init=r_init
    )

    f_lo = expected_irr(company, companies_dict, p_lo,
                        maturity_years=maturity_years, compounding=compounding,
                        par=par, per_year=per_year,
                        delay_quarters=delay_quarters,
                        recovery_lag_years=recovery_lag_years,
                        r_init=r_init) - target_pct

    f_hi = expected_irr(company, companies_dict, p_hi,
                        maturity_years=maturity_years, compounding=compounding,
                        par=par, per_year=per_year,
                        delay_quarters=delay_quarters,
                        recovery_lag_years=recovery_lag_years,
                        r_init=r_init) - target_pct

    # If we still don't have a valid bracket, fallback to closest endpoint
    if not (np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0):
        v_lo = expected_irr(company, companies_dict, p_lo,
                            maturity_years=maturity_years, compounding=compounding,
                            par=par, per_year=per_year,
                            delay_quarters=delay_quarters,
                            recovery_lag_years=recovery_lag_years,
                            r_init=r_init)
        v_hi = expected_irr(company, companies_dict, p_hi,
                            maturity_years=maturity_years, compounding=compounding,
                            par=par, per_year=per_year,
                            delay_quarters=delay_quarters,
                            recovery_lag_years=recovery_lag_years,
                            r_init=r_init)
        return p_lo if abs(v_lo - target_pct) <= abs(v_hi - target_pct) else p_hi

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (p_lo + p_hi)
        f_mid = expected_irr(company, companies_dict, mid,
                             maturity_years=maturity_years, compounding=compounding,
                             par=par, per_year=per_year,
                             delay_quarters=delay_quarters,
                             recovery_lag_years=recovery_lag_years,
                             r_init=r_init) - target_pct

        if not np.isfinite(f_mid):
            mid = np.nextafter(mid, p_hi)

        if abs(f_mid) < tol or (p_hi - p_lo) < tol:
            return mid

        if f_lo * f_mid <= 0:
            p_hi, f_hi = mid, f_mid
        else:
            p_lo, f_lo = mid, f_mid

    return 0.5 * (p_lo + p_hi)


def prices_to_hit_target(
    companies_dict: Dict[str, Dict[str, float]],
    *,
    target_pct: float = 10.0,
    compounding: str = "simple",
    p_lo: float = 60.0,
    p_hi: float = 120.0,
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    r_init: float,
    measure_from: str = "global",   # <-- add this
) -> pd.DataFrame:
    rows = []
    for company in companies_dict.keys():
        p_star = price_for_target_expected_irr(
            company_name=company,              # <-- was company=
            companies_dict=companies_dict,
            target_pct=target_pct,
            delay_quarters=delay_quarters,
            compounding=compounding,
            measure_from=measure_from,         # <-- pass through
            p_lo=p_lo,
            p_hi=p_hi,
            par=par,
            maturity_years=maturity_years,
            per_year=per_year,
            recovery_lag_years=recovery_lag_years,
            r_init=r_init,
            tol=1e-4,
            max_iter=100,
        )
        rows.append({
            "Company": company,
            "Target IRR (%)": target_pct,
            "Price* (OID % of par)": p_star
        })
    return pd.DataFrame(rows)



# ---------- Core: scenario IRRs with optional time-rebasing (no globals) ----------
def irr_series_with_delay(
    company_name: str,
    companies_dict: Dict[str, Dict[str, float]],
    price: float,
    *,
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    compounding: str,
    r_init: float,
    measure_from: str = "global"  # "global" (fund clock) or "deployment"
) -> pd.Series:
    """
    Returns a Series of scenario IRRs (in %) for Year 1..maturity_years and No Default.
    - Uses parameterized build_cashflows_for_company (outlay at delay_quarters on fund clock).
    - If measure_from='deployment', rebase times so outlay is at t=0 (shorter horizon when buying late).
    """
    scen = build_cashflows_for_company(
        company_name=company_name,
        companies=companies_dict,
        price=price,
        par=par,
        maturity_years=maturity_years,
        per_year=per_year,
        delay_quarters=delay_quarters,
        recovery_lag_years=recovery_lag_years,
    )

    order = [f"Year {i}" for i in range(1, maturity_years + 1)] + ["No Default"]
    out = {}

    if measure_from not in ("global", "deployment"):
        raise ValueError("measure_from must be 'global' or 'deployment'")

    dt = 1.0 / per_year
    shift = delay_quarters * dt

    for key in order:
        cf, t = scen[key]
        if measure_from == "deployment":
            # Shift times so the outlay quarter is t=0. Keep t >= 0.
            t2 = t - shift
            mask = t2 >= 0
            cf, t2 = cf[mask], t2[mask]
            r = safe_irr(cf, t2, compounding=compounding, r_init=r_init)
        else:
            r = safe_irr(cf, t, compounding=compounding, r_init=r_init)

        out[key] = 100.0 * r if np.isfinite(r) else np.nan

    return pd.Series(out)


# ---------- Expected IRR (strict; no fallbacks; no renormalization) ----------
def expected_irr_with_delay(
    company_name: str,
    companies_dict: Dict[str, Dict[str, float]],
    price: float,
    *,
    par: float,
    maturity_years: int,
    per_year: int,
    delay_quarters: int,
    recovery_lag_years: float,
    compounding: str,
    r_init: float,
    measure_from: str = "global"
) -> float:
    """
    Probability-weighted expected IRR (%) using annual PD and survival to maturity_years.
    Strict behavior: if any scenario IRR is NaN, returns NaN (matches your original style).
    """
    pd_val = float(companies_dict[company_name]["PD"])

    probs = {f"Year {k}": (1 - pd_val) ** (k - 1) * pd_val for k in range(1, maturity_years + 1)}
    probs["No Default"] = (1 - pd_val) ** maturity_years

    irr_vals = irr_series_with_delay(
        company_name=company_name,
        companies_dict=companies_dict,
        price=price,
        par=par,
        maturity_years=maturity_years,
        per_year=per_year,
        delay_quarters=delay_quarters,
        recovery_lag_years=recovery_lag_years,
        compounding=compounding,
        r_init=r_init,
        measure_from=measure_from
    )

    # strict: if any scenario is NaN, propagate
    if any(not np.isfinite(irr_vals[k]) for k in probs.keys()):
        return np.nan

    return float(sum(probs[k] * irr_vals[k] for k in probs.keys()))


# ---------- Single price solver for target expected IRR (no globals) ----------
def price_for_target_expected_irr(
    company_name: str,
    companies_dict: Dict[str, Dict[str, float]],
    *,
    target_pct: float = 10.0,
    delay_quarters: int,
    compounding: str,
    measure_from: str,
    p_lo: float,
    p_hi: float,
    tol: float,
    max_iter: int,
    par: float,
    maturity_years: int,
    per_year: int,
    recovery_lag_years: float,
    r_init: float
) -> float:
    """
    Solve E[IRR](price) = target_pct using strict expected IRR.
    Works for any delay and either clock convention (global/deployment).
    """
    def f(price):
        val = expected_irr_with_delay(
            company_name, companies_dict, price,
            par=par, maturity_years=maturity_years, per_year=per_year,
            delay_quarters=delay_quarters, recovery_lag_years=recovery_lag_years,
            compounding=compounding, r_init=r_init, measure_from=measure_from
        )
        return val - target_pct

    v_lo, v_hi = f(p_lo), f(p_hi)
    expand = 0
    while (not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_lo * v_hi > 0) and expand < 10:
        p_lo = max(1.0, p_lo * 0.9)
        p_hi = p_hi * 1.2
        v_lo, v_hi = f(p_lo), f(p_hi)
        expand += 1

    if not (np.isfinite(v_lo) and np.isfinite(v_hi) and v_lo * v_hi <= 0):
        candidates = [(p_lo, v_lo), (p_hi, v_hi)]
        candidates = [(p, v) for p, v in candidates if np.isfinite(v)]
        return min(candidates, key=lambda x: abs(x[1]))[0] if candidates else np.nan

    for _ in range(max_iter):
        mid = 0.5 * (p_lo + p_hi)
        v_mid = f(mid)
        if not np.isfinite(v_mid):
            mid = np.nextafter(mid, p_hi)
            v_mid = f(mid)
        if abs(v_mid) < tol or abs(p_hi - p_lo) < tol:
            return mid
        if v_lo * v_mid <= 0:
            p_hi, v_hi = mid, v_mid
        else:
            p_lo, v_lo = mid, v_mid
    return 0.5 * (p_lo + p_hi)


# ---------- Wrapper(s) ----------
def target_prices_10pct_for_delay(
    companies_dict: Dict[str, Dict[str, float]],
    *,
    delay_quarters: int,
    compounding: str,
    measure_from: str,
    par: float,
    maturity_years: int,
    per_year: int,
    recovery_lag_years: float,
    r_init: float,
    p_lo: float = 60.0,
    p_hi: float = 120.0,
    tol: float = 1e-4,
    max_iter: int = 80
) -> pd.DataFrame:
    rows = []
    for company in companies_dict.keys():
        p_star = price_for_target_expected_irr(
            company_name=company,              # <-- was company=
            companies_dict=companies_dict,
            target_pct=10.0,
            delay_quarters=delay_quarters,
            compounding=compounding,
            measure_from=measure_from,
            p_lo=p_lo, p_hi=p_hi, tol=tol, max_iter=max_iter,
            par=par, maturity_years=maturity_years, per_year=per_year,
            recovery_lag_years=recovery_lag_years, r_init=r_init
        )
        rows.append({"Company": company, f"Price* @10% (q={delay_quarters})": p_star})
    return pd.DataFrame(rows).sort_values("Company")    


def target_prices_10pct_all_delays(
    companies_dict: Dict[str, Dict[str, float]],
    *,
    delays: List[int],
    compounding: str,
    measure_from: str,
    par: float,
    maturity_years: int,
    per_year: int,
    recovery_lag_years: float,
    r_init: float,
    p_lo: float = 60.0,
    p_hi: float = 120.0,
    tol: float = 1e-4,
    max_iter: int = 80
) -> pd.DataFrame:
    """
    Returns one wide table with columns: Company, q=0, q=1, q=2, ...
    """
    base = pd.DataFrame({"Company": list(companies_dict.keys())}).sort_values("Company")
    for dq in delays:
        df_dq = target_prices_10pct_for_delay(
            companies_dict,
            delay_quarters=dq,
            compounding=compounding,
            measure_from=measure_from,
            par=par,
            maturity_years=maturity_years,
            per_year=per_year,
            recovery_lag_years=recovery_lag_years,
            r_init=r_init,
            p_lo=p_lo,
            p_hi=p_hi,
            tol=tol,
            max_iter=max_iter
        )
        base = base.merge(df_dq, on="Company", how="left")
    return base
