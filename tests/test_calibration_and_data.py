import numpy as np
import pandas as pd
import pytest

from mcpricer.analytic import heston_price, implied_vol
from mcpricer.calibration import calibrate_heston
from mcpricer.data import optionmetrics as om

TRUE = dict(v0=0.025, kappa=2.0, theta=0.05, xi=0.8, rho=-0.7)
S0, R, Q = 100.0, 0.04, 0.015


def _surface():
    rows = []
    for T in [0.1, 0.25, 0.5, 1.0, 2.0]:
        F = S0 * np.exp((R - Q) * T)
        Ks = F * np.exp(np.linspace(-0.4, 0.2, 13) * np.sqrt(T))
        call = heston_price(S0, Ks, T, R, Q, **TRUE)
        for k, c in zip(Ks, call):
            cp = "call" if k >= F else "put"
            px = c if cp == "call" else c - S0 * np.exp(-Q * T) + k * np.exp(-R * T)
            rows.append(dict(T=T, K=k, cp=cp, mid=px, r=R, q=Q,
                             iv=implied_vol(px, S0, k, T, R, Q, cp)))
    return pd.DataFrame(rows).dropna()


def test_calibration_recovers_parameters():
    res = calibrate_heston(_surface(), S0)
    assert res.rmse_iv < 1e-4
    for k, v in TRUE.items():
        assert getattr(res.params, k) == pytest.approx(v, rel=0.05), k


def _fake_ivydb(date="2023-12-29"):
    """Raw tables in IvyDB layout generated from Black-Scholes with a known forward."""
    from mcpricer.analytic import bs_price
    d = pd.Timestamp(date)
    S, r, q, sig = 4769.83, 0.05, 0.015, 0.15
    opts = []
    for days in [30, 58, 93, 184, 3]:  # 3 days -> filtered out
        T = days / 365
        for k in np.arange(3800, 5800, 50.0):
            for cp in "CP":
                px = float(bs_price(S, k, T, r, q, sig, "call" if cp == "C" else "put"))
                opts.append(dict(secid=om.SPX_SECID, date=d, exdate=d + pd.Timedelta(days=days),
                                 cp_flag=cp, strike_price=int(k * 1000),
                                 best_bid=px - 0.1, best_offer=px + 0.1, volume=10,
                                 open_interest=100, impl_volatility=sig, delta=0.5,
                                 optionid=1, am_settlement=1, ss_flag="0"))
    zero = pd.DataFrame({"date": d, "days": [7, 30, 90, 180, 365], "rate": 100 * r})
    spot = pd.DataFrame({"secid": [om.SPX_SECID], "date": [d], "close": [S]})
    return {"options": pd.DataFrame(opts), "spot": spot, "zero": zero}, (S, r, q, sig)


def test_optionmetrics_cleaning_and_forwards():
    raw, (S, r, q, sig) = _fake_ivydb()
    surf, s0 = om.build_surface(raw)
    assert s0 == pytest.approx(S)
    assert surf["days"].min() >= 7
    assert set(surf["cp"]) == {"call", "put"}
    # OTM only
    assert ((surf["cp"] == "call") == (surf["K"] >= surf["F"])).all()
    # implied dividend yield recovered from put-call parity
    np.testing.assert_allclose(surf["q"], q, atol=2e-4)
    np.testing.assert_allclose(surf["iv"], sig, atol=2e-3)
