import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from scipy.optimize import brentq

def prob_no_default(pd: float, n: int) -> float:
    """
    Probability of surviving n independent periods with default probability pd each.
    """
    return (1 - pd) ** n

def simulate_defaults(df, pd_default, recovery_price=0.7, n_sims=10_000, seed=0):
    """
    Simulate default-adjusted cashflows. Supports:
      - Scalar recovery_price
      - Array of recovery_price over time
      - Function recovery_price(t)
    """
    rng = np.random.default_rng(seed)
    results = {}

    for comp, sub in df.groupby('company'):
        sub = sub.sort_values('time')
        c = sub['coupon'].to_numpy()
        d = sub['delta_notional'].to_numpy()
        p = sub['price'].to_numpy()
        times = sub['time'].to_numpy()
        n = len(times)

        base_cf = c[:, None] + d[:, None] * p[:, None]
        cf_mat = np.tile(base_cf, (1, n_sims))

        uni_later = rng.random((n - 1, n_sims))
        default_mask = np.vstack([
            np.zeros((1, n_sims), dtype=bool),       # time 0 → never default
            uni_later < pd_default                   # time 1+
        ])

        any_default = default_mask.any(axis=0)
        first_def = np.where(any_default, default_mask.argmax(axis=0), -1)

        # Handle recovery price resolution
        if callable(recovery_price):
            recovery_at = lambda i: recovery_price(times[i])
        elif hasattr(recovery_price, '__len__'):
            recovery_vec = np.array(recovery_price)
            recovery_at = lambda i: recovery_vec[i]
        else:
            recovery_at = lambda i: recovery_price  # scalar

        # Apply default logic per simulation
        for i in range(n_sims):
            fd = first_def[i]
            if fd == -1:
                continue
            cf_mat[fd, i] = c[fd] + d[fd] * recovery_at(fd)
            cf_mat[(fd+1):, i] = 0.0

        results[comp] = {
            'times': times,
            'cf_mat': cf_mat,
            'total_cf': cf_mat.sum(axis=0),
            'prop_no_default': np.mean(first_def == -1),
            'n_periods': n
        }

    return results


def aggregate_total_cashflows(sim_results):
    """
    Combine all simulated cashflows into total portfolio cashflows.
    """
    total_cf_all = np.sum([v['total_cf'] for v in sim_results.values()], axis=0)
    return total_cf_all

def simulate_irrs(sim_results):
    cf_stack = np.vstack([v['cf_mat'] for v in sim_results.values()])
    time_stack = np.concatenate([v['times'] for v in sim_results.values()])
    cf_stack_jax = jnp.array(cf_stack.T)
    time_stack_jax = jnp.array(time_stack)

    @vmap
    def compute_irr_safe(cf_row):
        is_zero = jnp.all(cf_row == 0.0)
        return lax.cond(
            is_zero,
            lambda _: jnp.nan,
            lambda _: IRR(cf_row, time_stack_jax),
            operand=None
        )

    irrs = np.array(compute_irr_safe(cf_stack_jax))
    return irrs[~np.isnan(irrs)]

def solve_fair_price(df, PD, recovery_price, target_IRR, lower=0.5, upper=1.5):
    """
    Solve for entrance price P0 such that expected IRR under default simulations equals IRR under no default.
    """
    def set_price_and_simulate(p0):
        df_mod = df.copy()
        t0 = df_mod['time'].min()
        df_mod.loc[df_mod['time'] == t0, 'price'] = p0
        df_mod['total_cf'] = df_mod['coupon'] + df_mod['delta_notional'] * df_mod['price']

        sim_results = simulate_defaults(df_mod, PD, recovery_price)
        irrs = simulate_irrs(sim_results)
        return irrs.mean()

    def objective(p0):
        return set_price_and_simulate(p0) - target_IRR

    return brentq(objective, lower, upper)

