import pytest
import jax.numpy as jnp
from scripts.financial import NPV, irr_newton, irr_simulated_batch, compute_no_default_irr, expected_irr_given_price
from typing import Dict, Tuple


def test_npv_basic():
    cf = jnp.array([100, 100, 100])
    t = jnp.array([1, 2, 3])
    r = 0.05
    npv = NPV(r, cf, t)
    expected = sum([100 / jnp.exp(r * ti) for ti in t])
    assert jnp.isclose(npv, expected, atol=2e-4)


def test_irr_newton_positive():
    cf = jnp.array([-1000, 400, 400, 400])
    t = jnp.array([0, 1, 2, 3])
    irr = irr_newton(cf, t)
    assert jnp.isfinite(irr)
    npv = NPV(irr, cf, t)
    assert jnp.abs(npv) < 2e-4

def test_irr_simulated_batch_simple():
    cashflows = jnp.array([
        [-100, 60, 60],
        [-100, 0, 120]
    ])
    irr_array = irr_simulated_batch(cashflows)
    assert irr_array.shape == (2,)
    assert jnp.all(jnp.isfinite(irr_array))


def test_compute_no_default_irr():
    cf = jnp.array([0.0, 30.0, 30.0, 30.0])
    t = jnp.array([0, 1, 2, 3])
    p0 = 90.0
    irr = compute_no_default_irr(cf, t, p0)
    assert jnp.isfinite(irr)
    full_cf = cf.at[0].set(-p0)
    assert jnp.abs(NPV(irr, full_cf, t)) < 2e-4


def test_expected_irr_given_price():
    sim_results = {
        'CompanyA': {'cf_mat': jnp.array([[10.0, 12.0, 14.0], [11.0, 13.0, 15.0]])},
        'CompanyB': {'cf_mat': jnp.array([[5.0, 6.0, 7.0], [6.0, 7.0, 8.0]])}
    }
    p0 = 20.0
    irr = expected_irr_given_price(sim_results, p0)
    assert isinstance(irr, float)
    assert irr > -1.0 and irr < 1.0