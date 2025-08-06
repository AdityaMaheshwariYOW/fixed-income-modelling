import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List
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


def build_cashflow_matrix(coupon, delta_notional, price, alive_mat, default_price_factor=0.7):
    """
    Build cashflow matrix per simulation with proper sign.

    Parameters
    ----------
    coupon : float
        Periodic coupon.
    delta_notional : float
        Notional change at t=0 (should be negative).
    price : float
        Entry price (per unit of notional).
    alive_mat : jnp.array
        Simulation default paths (n_simulations x n_periods)

    Returns
    -------
    jnp.array
        Cashflow matrix (n_simulations x n_periods)
    """
    n_simulations, n_periods = alive_mat.shape

    # Initialize cashflows
    cf_mat = jnp.zeros((n_simulations, n_periods))

    # Time 0: initial investment (outlay)
    cf_mat = cf_mat.at[:, 0].set(delta_notional * price)  # Should be negative

    # Coupons: only if alive
    for t in range(1, n_periods):
        cf_mat = cf_mat.at[:, t].add(coupon * alive_mat[:, t])

    # Final period: add notional repayment if survived to last
    # Check where default happens — repay notional only at default
    default_occurred = (alive_mat[:, 1:] - alive_mat[:, :-1]) == -1
    first_default_idx = jnp.argmax(default_occurred, axis=1) + 1
    default_happens = jnp.any(default_occurred, axis=1)

    # Repay at default (lower price), else at end (full price)
    default_prices = price * default_price_factor  # e.g., 70% recovery on default
    for i in range(n_simulations):
        if default_happens[i]:
            cf_mat = cf_mat.at[i, first_default_idx[i]].add(-delta_notional * default_prices)
        else:
            cf_mat = cf_mat.at[i, -1].add(-delta_notional * price)  # final repayment

    return cf_mat

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


def aggregate_total_cashflows(sim_results: dict) -> jnp.ndarray:
    """
    Aggregate total cashflows from simulation results across all companies.

    Parameters
    ----------
    sim_results : dict
        Dictionary containing simulation results per company. Each value must have a 'cf_mat' key
        representing a (n_simulations x n_periods) cashflow matrix.

    Returns
    -------
    jnp.ndarray
        Array of shape (n_simulations,) representing total cashflow per simulation across all companies.
    """
    # Stack and sum across companies
    cf_matrices = [result["cf_mat"] for result in sim_results.values()]
    
    # Ensure all matrices have the same shape
    shapes = [cf.shape for cf in cf_matrices]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("Cashflow matrices have inconsistent shapes.")

    # Shape: (n_simulations, n_periods)
    portfolio_cf = jnp.sum(jnp.stack(cf_matrices), axis=0)

    # Sum across periods for each simulation
    total_cf_all = jnp.sum(portfolio_cf, axis=1)
    return total_cf_all