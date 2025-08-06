import pytest
import jax.numpy as jnp
from scripts.simulation import (
    prob_no_default,
    simulate_defaults,
    build_cashflow_matrix,
    run_simulations_for_company,
    summarize_simulation_results
)
from typing import Dict, Tuple, List


def test_prob_no_default():
    assert prob_no_default(0.0, 5) == 1.0
    assert prob_no_default(0.5, 1) == 0.5
    assert round(prob_no_default(0.1, 3), 4) == round((0.9 ** 3), 4)


def test_simulate_defaults_returns_correct_shape():
    sim_matrix = simulate_defaults(n_periods=5, n_simulations=100, pd=0.0)
    assert sim_matrix.shape == (100, 5)
    assert jnp.all(sim_matrix == 1)

def test_build_cashflow_matrix_values():
    # Simulate 10 simulations with 5 periods, all survive (no defaults)
    path = jnp.ones((10, 5), dtype=jnp.int32)

    # Run cashflow matrix builder with known inputs
    cashflows = build_cashflow_matrix(
        coupon=5.0,
        delta_notional=-1.0,
        price=1.0,
        alive_mat=path,
        default_price_factor=0.7  # should not matter here since no default
    )

    assert cashflows.shape == (10, 5)

    # Check first column (initial outlay)
    expected_t0 = -1.0  # -1.0 * 1.0
    assert jnp.allclose(cashflows[:, 0], expected_t0), f"Initial cashflow incorrect: {cashflows[:, 0]}"

    # Check intermediate coupons (periods 1 to 3)
    expected_coupon = 5.0
    assert jnp.allclose(cashflows[:, 1:-1], expected_coupon), f"Coupons incorrect: {cashflows[:, 1:-1]}"

    # Final period: coupon + notional repayment
    expected_final = 5.0 + 1.0  # coupon + full repayment
    assert jnp.allclose(cashflows[:, -1], expected_final), f"Final cashflow incorrect: {cashflows[:, -1]}"



def test_run_simulations_for_company_output():
    out = run_simulations_for_company(
        company="ABC",
        coupon=4.0,
        delta_notional=-1.0,
        price=1.0,
        n_periods=6,
        n_simulations=20,
        pd=0.1
    )
    assert isinstance(out, dict)
    assert out["cf_mat"].shape == (20, 6)
    assert out["alive_mat"].shape == (20, 6)
    assert out["company"] == "ABC"


def test_summarize_simulation_results_runs():
    sims = {
        "X": run_simulations_for_company("X", 4.0, -1.0, 1.0, 5, 50, 0.05),
        "Y": run_simulations_for_company("Y", 5.0, -1.0, 1.0, 5, 50, 0.05)
    }
    mean_irr, std_irr = summarize_simulation_results(sims, price_at_t0=1.0)
    assert isinstance(mean_irr, float)
    assert isinstance(std_irr, float)
