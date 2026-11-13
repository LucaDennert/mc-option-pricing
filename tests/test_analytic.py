import numpy as np
import pytest

from mcpricer import analytic as A

S, K, T, R, Q, SIG = 100.0, 105.0, 0.75, 0.03, 0.01, 0.22


def test_put_call_parity():
    c = A.bs_price(S, K, T, R, Q, SIG, "call")
    p = A.bs_price(S, K, T, R, Q, SIG, "put")
    assert c - p == pytest.approx(S * np.exp(-Q * T) - K * np.exp(-R * T), abs=1e-12)


@pytest.mark.parametrize("cp", ["call", "put"])
def test_implied_vol_roundtrip(cp):
    for sig in [0.05, 0.2, 0.8]:
        price = float(A.bs_price(S, K, T, R, Q, sig, cp))
        assert A.implied_vol(price, S, K, T, R, Q, cp) == pytest.approx(sig, abs=1e-8)


@pytest.mark.parametrize("cp", ["call", "put"])
@pytest.mark.parametrize("direction,H", [("down", 85.0), ("up", 125.0)])
@pytest.mark.parametrize("strike", [90.0, 110.0])
def test_barrier_in_out_parity(cp, direction, H, strike):
    out = A.barrier_price(S, strike, H, T, R, Q, SIG, cp, f"{direction}-and-out")
    inn = A.barrier_price(S, strike, H, T, R, Q, SIG, cp, f"{direction}-and-in")
    van = float(A.bs_price(S, strike, T, R, Q, SIG, cp))
    assert out + inn == pytest.approx(van, abs=1e-10)
    assert out >= -1e-12 and inn >= -1e-12


def test_barrier_far_away_is_vanilla():
    van = float(A.bs_price(S, K, T, R, Q, SIG))
    assert A.barrier_price(S, K, 1e-3, T, R, Q, SIG, "call", "down-and-out") == pytest.approx(van, rel=1e-8)


def test_geometric_asian_single_fixing_is_european():
    g = A.geometric_asian_price(S, K, T, R, Q, SIG, [T])
    assert g == pytest.approx(float(A.bs_price(S, K, T, R, Q, SIG)), rel=1e-12)


def test_heston_reduces_to_bs_when_vol_of_vol_vanishes():
    Ks = np.array([70.0, 100.0, 130.0])
    h = A.heston_price(S, Ks, T, R, Q, SIG**2, 2.0, SIG**2, 1e-4, -0.5)
    np.testing.assert_allclose(h, A.bs_price(S, Ks, T, R, Q, SIG), atol=2e-3)


def test_heston_put_call_parity():
    args = (S, K, T, R, Q, 0.04, 1.5, 0.05, 0.7, -0.7)
    c = A.heston_price(*args, cp="call")[0]
    p = A.heston_price(*args, cp="put")[0]
    assert c - p == pytest.approx(S * np.exp(-Q * T) - K * np.exp(-R * T), abs=1e-10)


def test_merton_without_jumps_is_bs():
    m = A.merton_price(S, K, T, R, Q, SIG, lam=0.0, mu_j=-0.1, sigma_j=0.1)
    assert m == pytest.approx(float(A.bs_price(S, K, T, R, Q, SIG)), rel=1e-12)
