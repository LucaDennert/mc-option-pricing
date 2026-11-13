"""Monte Carlo vs closed forms. Tolerances are a few standard errors (+ known bias)."""
import numpy as np
import pytest

from mcpricer import (Asian, Autocallable, Barrier, BlackScholes, ControlVariate, Digital, European,
                      Heston, Lookback, MertonJD, MonteCarloEngine)
from mcpricer import analytic as A
from mcpricer.random import RandomSource, brownian_bridge_increments

S, R, Q, SIG = 100.0, 0.03, 0.01, 0.2
BS = BlackScholes(S, R, Q, SIG)


def close(res, target, n_se=4.0, extra=0.0):
    assert abs(res.price - target) <= n_se * res.std_error + extra, (res, target)


@pytest.mark.parametrize("method,anti", [("pseudo", False), ("pseudo", True), ("sobol", False)])
@pytest.mark.parametrize("cp", ["call", "put"])
def test_european_bs(method, anti, cp):
    eng = MonteCarloEngine(n_paths=2**15, n_steps=1, method=method, antithetic=anti, seed=1)
    close(eng.price(BS, European(105, 1.0, cp)), float(A.bs_price(S, 105, 1.0, R, Q, SIG, cp)))


def test_digital():
    eng = MonteCarloEngine(n_paths=2**15, n_steps=1, seed=2)
    close(eng.price(BS, Digital(100, 1.0)), float(A.bs_digital(S, 100, 1.0, R, Q, SIG)))


def test_geometric_asian_and_control_variate():
    geo = Asian(100, 1.0, 12, average="geometric")
    exact = A.geometric_asian_price(S, 100, 1.0, R, Q, SIG, geo.fixing_times)
    eng = MonteCarloEngine(n_paths=2**15, n_steps=12, seed=3)
    close(eng.price(BS, geo), exact)
    # CV must shrink the error by > 10x on the arithmetic Asian
    ari = Asian(100, 1.0, 12)
    plain = eng.price(BS, ari)
    cv = eng.price(BS, ari, ControlVariate(geo, exact))
    assert cv.std_error < plain.std_error / 10
    close(cv, plain.price, n_se=0, extra=4 * plain.std_error)


@pytest.mark.parametrize("kind,H", [("down-and-out", 85.0), ("down-and-in", 85.0),
                                    ("up-and-out", 125.0), ("up-and-in", 125.0)])
@pytest.mark.parametrize("cp", ["call", "put"])
def test_barrier_continuous_brownian_bridge(kind, H, cp):
    exact = A.barrier_price(S, 100, H, 1.0, R, Q, SIG, cp, kind)
    eng = MonteCarloEngine(n_paths=2**15, n_steps=50, seed=4)
    close(eng.price(BS, Barrier(100, H, 1.0, cp, kind, "continuous", SIG)), exact, extra=0.01)


def test_barrier_bgk_beats_raw_discrete():
    exact = A.barrier_price(S, 100, 90, 1.0, R, Q, 0.25, "call", "down-and-out")
    m = BlackScholes(S, R, Q, 0.25)
    eng = MonteCarloEngine(n_paths=2**15, n_steps=52, seed=5)
    raw = eng.price(m, Barrier(100, 90, 1.0, "call", "down-and-out", "discrete"))
    bgk = eng.price(m, Barrier(100, 90, 1.0, "call", "down-and-out", "bgk", 0.25))
    assert abs(bgk.price - exact) < 0.3 * abs(raw.price - exact)


def test_floating_lookback_with_bgk_correction():
    m = BlackScholes(S, 0.05, 0.0, 0.3)
    exact = A.floating_lookback_call(S, 1.0, 0.05, 0.0, 0.3)
    eng = MonteCarloEngine(n_paths=2**14, n_steps=250, seed=6)
    close(eng.price(m, Lookback(1.0, sigma=0.3)), exact, extra=0.05)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_heston_qe(K):
    h = Heston(S, R, Q, v0=0.04, kappa=1.5, theta=0.06, xi=0.8, rho=-0.7)  # Feller violated
    exact = A.heston_price(S, K, 1.0, R, Q, 0.04, 1.5, 0.06, 0.8, -0.7)[0]
    eng = MonteCarloEngine(n_paths=2**15, n_steps=24, antithetic=True, seed=7)
    close(eng.price(h, European(K, 1.0)), exact, extra=0.02)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_merton(K):
    m = MertonJD(S, R, Q, 0.15, lam=0.5, mu_j=-0.1, sigma_j=0.15)
    exact = A.merton_price(S, K, 1.0, R, Q, 0.15, 0.5, -0.1, 0.15)
    eng = MonteCarloEngine(n_paths=2**15, n_steps=4, seed=8)
    close(eng.price(m, European(K, 1.0)), exact)


def test_martingale_property_all_models():
    models = [BS, Heston(S, R, Q, 0.04, 1.5, 0.06, 0.8, -0.7),
              MertonJD(S, R, Q, 0.15, 0.5, -0.1, 0.15)]
    eng = MonteCarloEngine(n_paths=2**15, n_steps=12, seed=9)
    for m in models:
        close(eng.price(m, European(0.0, 1.0)), S * np.exp(-Q * 1.0), extra=0.02)


def test_autocallable_degenerate_is_zero_coupon_bond():
    ac = Autocallable(2.0, n_obs=8, autocall_barrier=1e9, coupon=0.0, ki_barrier=0.0)
    res = MonteCarloEngine(n_paths=2**12, seed=10).price(BS, ac)
    assert res.price == pytest.approx(np.exp(-R * 2.0), rel=1e-12)


def test_autocallable_certain_first_call():
    ac = Autocallable(2.0, n_obs=8, autocall_barrier=0.0, coupon=0.01, coupon_barrier=0.0)
    res = MonteCarloEngine(n_paths=2**12, seed=11).price(BS, ac)
    assert res.price == pytest.approx(1.01 * np.exp(-R * 0.25), rel=1e-12)


def test_brownian_bridge_covariance():
    times = np.linspace(0, 1, 9)
    z = RandomSource("pseudo", 0).normals(200_000, 8)
    w = np.cumsum(brownian_bridge_increments(z, times), axis=1)
    cov = np.cov(w.T)
    exact = np.minimum.outer(times[1:], times[1:])
    np.testing.assert_allclose(cov, exact, atol=0.01)


def test_rqmc_beats_mc():
    p = European(100, 1.0)
    mc = MonteCarloEngine(n_paths=2**14, n_steps=1, seed=0).price(BS, p)
    qmc = MonteCarloEngine(n_paths=2**14, n_steps=1, method="sobol", seed=0).price(BS, p)
    assert qmc.std_error < mc.std_error / 10
