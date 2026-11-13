"""Monte Carlo Greeks.

* :func:`bump_greeks` - central finite differences with common random numbers
  (same seed for up/down revaluations), works for any model / payoff.
* :func:`pathwise_delta_call` and :func:`likelihood_ratio_delta_digital` -
  unbiased single-simulation estimators under GBM, showing why the pathwise
  method fails for discontinuous payoffs and the LR method does not.
"""
from __future__ import annotations

import numpy as np

from .engine import MonteCarloEngine
from .models import BlackScholes, Model
from .payoffs import Payoff
from .random import RandomSource


def bump_greeks(engine: MonteCarloEngine, model: Model, payoff: Payoff,
                rel_bump_spot: float = 0.01, abs_bump_vol: float = 0.01,
                abs_bump_rate: float = 1e-4) -> dict:
    """Delta, gamma, vega (per vol point x100), rho (per 1bp x1e4) by CRN bumping.

    Vega is taken w.r.t. ``sigma`` for BS / Merton and w.r.t. sqrt(v0) for Heston.
    """
    p0 = engine.price(model, payoff).price
    h = model.S0 * rel_bump_spot
    up = engine.price(model.with_params(S0=model.S0 + h), payoff).price
    dn = engine.price(model.with_params(S0=model.S0 - h), payoff).price
    out = {"price": p0, "delta": (up - dn) / (2 * h), "gamma": (up - 2 * p0 + dn) / h**2}

    if hasattr(model, "sigma"):
        s = model.sigma
        vu = engine.price(model.with_params(sigma=s + abs_bump_vol), payoff).price
        vd = engine.price(model.with_params(sigma=s - abs_bump_vol), payoff).price
        out["vega"] = (vu - vd) / (2 * abs_bump_vol)
    elif hasattr(model, "v0"):
        s = np.sqrt(model.v0)
        vu = engine.price(model.with_params(v0=(s + abs_bump_vol) ** 2), payoff).price
        vd = engine.price(model.with_params(v0=(s - abs_bump_vol) ** 2), payoff).price
        out["vega"] = (vu - vd) / (2 * abs_bump_vol)

    ru = engine.price(model.with_params(r=model.r + abs_bump_rate), payoff).price
    rd = engine.price(model.with_params(r=model.r - abs_bump_rate), payoff).price
    out["rho"] = (ru - rd) / (2 * abs_bump_rate)
    return out


def _terminal_gbm(model: BlackScholes, T: float, n: int, seed: int):
    z = RandomSource("pseudo", seed).normals(n, 1)[:, 0]
    s_T = model.S0 * np.exp((model.r - model.q - 0.5 * model.sigma**2) * T
                            + model.sigma * np.sqrt(T) * z)
    return s_T, z


def pathwise_delta_call(model: BlackScholes, K: float, T: float, n: int = 200_000,
                        seed: int = 7) -> tuple[float, float]:
    """Pathwise delta of a call: e^{-rT} 1{S_T > K} S_T / S0. Returns (estimate, se)."""
    s_T, _ = _terminal_gbm(model, T, n, seed)
    est = np.exp(-model.r * T) * (s_T > K) * s_T / model.S0
    return est.mean(), est.std(ddof=1) / np.sqrt(n)


def likelihood_ratio_delta_digital(model: BlackScholes, K: float, T: float,
                                   n: int = 200_000, seed: int = 7) -> tuple[float, float]:
    """LR delta of a cash-or-nothing call: e^{-rT} 1{S_T > K} Z / (S0 sigma sqrt(T))."""
    s_T, z = _terminal_gbm(model, T, n, seed)
    est = np.exp(-model.r * T) * (s_T > K) * z / (model.S0 * model.sigma * np.sqrt(T))
    return est.mean(), est.std(ddof=1) / np.sqrt(n)
