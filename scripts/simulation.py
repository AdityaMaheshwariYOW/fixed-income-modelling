import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from scripts.financial import irr_simulated_batch

# -------------------------
# Default & Simulation Utilities
# -------------------------

def prob_no_default(pd: float, n: int) -> float:
    """
    Compute the probability that a company does NOT default over n periods,
    assuming independent default probability per period.

    Parameters
    ----------
    pd : float
        Per-period default probability.
    n : int
        Number of time periods.

    Returns
    -------
    float
        Probability of survival across all periods.
    """
    return (1 - pd) ** n

def simulate_defaults(n_periods: int, n_simulations: int, pd: float) -> jnp.ndarray:
    """
    Generate binary default paths for multiple simulations.

    Parameters
    ----------
    n_periods : int
        Number of time steps per simulation.
    n_simulations : int
        Number of independent simulations.
    pd : float
        Probability of default at each time step.

    Returns
    -------
    jnp.ndarray
        (n_simulations x n_periods) matrix with 1 if alive, 0 if defaulted.
    """
    rand_matrix = jax.random.uniform(jax.random.PRNGKey(0), (n_simulations, n_periods))
    default_matrix = (rand_matrix > pd).astype(jnp.int32)
    default_path = jnp.cumprod(default_matrix, axis=1)  # once defaulted, always defaulted
    return default_path

def build_cashflow_matrix(
    coupon: float,
    delta_notional: float,
    price: float,
    default_path: jnp.ndarray
) -> jnp.ndarray:
    """
    Construct a matrix of cashflows under simulated default paths.

    Parameters
    ----------
    coupon : float
        Periodic coupon payment.
    delta_notional : float
        Change in notional exposure at entry.
    price : float
        Entry price of the bond.
    default_path : jnp.ndarray
        Matrix indicating survival status (1=alive, 0=defaulted) for each period.

    Returns
    -------
    jnp.ndarray
        Cashflow matrix of shape (n_simulations, n_periods).
    """
    n_sim, n_t = default_path.shape
    cashflows = jnp.ones_like(default_path, dtype=jnp.float32) * coupon
    cashflows = cashflows.at[:, 0].add(-delta_notional * price)
    cashflows = cashflows * default_path
    return cashflows

def run_simulations_for_company(
    company: str,
    coupon: float,
    delta_notional: float,
    price: float,
    n_periods: int,
    n_simulations: int,
    pd: float
) -> Dict:
    """
    Run a default-adjusted cashflow simulation for one company.

    Parameters
    ----------
    company : str
        Company name identifier.
    coupon : float
        Periodic cash coupon.
    delta_notional : float
        Change in notional at initial time.
    price : float
        Entry bond price.
    n_periods : int
        Number of cashflow periods.
    n_simulations : int
        Number of Monte Carlo simulations.
    pd : float
        Per-period default probability.

    Returns
    -------
    dict
        Dictionary with simulated paths and cashflows.
    """
    path = simulate_defaults(n_periods, n_simulations, pd)
    cf_mat = build_cashflow_matrix(coupon, delta_notional, price, path)
    return {
        "company": company,
        "cf_mat": cf_mat,
        "alive_mat": path
    }

def summarize_simulation_results(
    sim_results: Dict[str, Dict],
    price_at_t0: float
) -> Tuple[float, float]:
    """
    Compute summary statistics (mean and std) of IRRs across simulation paths.

    Parameters
    ----------
    sim_results : dict
        Dictionary of results from multiple company simulations.
    price_at_t0 : float
        Time-zero investment amount applied to all simulations.

    Returns
    -------
    tuple of float
        Mean and standard deviation of simulated IRRs.
    """
    cf_list = []
    for result in sim_results.values():
        cf = result["cf_mat"].at[:, 0].set(-price_at_t0)
        cf_list.append(cf)

    total_cf = jnp.sum(jnp.stack(cf_list), axis=0)  # shape (n_sims, n_periods)
    irr_array = irr_simulated_batch(total_cf)
    return float(jnp.mean(irr_array)), float(jnp.std(irr_array))
