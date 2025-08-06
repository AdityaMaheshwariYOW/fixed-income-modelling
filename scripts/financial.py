import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import lax
from typing import Union, List

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
