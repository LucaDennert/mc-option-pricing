"""Calibration of the Heston model to an implied-volatility surface.

The objective is the vector of vega-weighted price errors
``(P_model - P_mkt) / Vega_BS(iv_mkt)``, a first-order approximation of the
implied-vol error that avoids inverting Black-Scholes inside the optimiser.
Heston prices come from the semi-analytic formula, vectorised across strikes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .analytic import bs_greeks, heston_price, implied_vol

PARAM_NAMES = ("v0", "kappa", "theta", "xi", "rho")
LOWER = np.array([1e-4, 0.05, 1e-4, 0.05, -0.999])
UPPER = np.array([1.0, 15.0, 1.0, 4.0, 0.5])


@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def to_json(self, path, **extra):
        with open(path, "w") as f:
            json.dump({**asdict(self), **extra}, f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: d[k] for k in PARAM_NAMES})


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_iv: float           # root-mean-square implied-vol error (vol points, e.g. 0.005)
    fitted: pd.DataFrame     # input quotes + model price + model IV
    success: bool
    n_quotes: int


def _model_prices(x, surface: pd.DataFrame, S0: float) -> np.ndarray:
    v0, kappa, theta, xi, rho = x
    out = np.empty(len(surface))
    for T, g in surface.groupby("T", sort=False):
        r, q = g["r"].iloc[0], g["q"].iloc[0]
        call = heston_price(S0, g["K"].values, T, r, q, v0, kappa, theta, xi, rho, "call")
        put = call - S0 * np.exp(-q * T) + g["K"].values * np.exp(-r * T)
        out[surface.index.get_indexer(g.index)] = np.where(g["cp"].values == "call", call, put)
    return out


def calibrate_heston(surface: pd.DataFrame, S0: float, x0=None,
                     max_nfev: int = 300, verbose: int = 0) -> CalibrationResult:
    """Fit Heston to quotes.

    ``surface`` needs columns: T, K, cp ("call"/"put"), mid, iv, r, q.
    ``r`` and ``q`` are per-maturity continuous rates (q typically implied
    from the put-call-parity forward).
    """
    s = surface.reset_index(drop=True).copy()
    vega = bs_greeks(S0, s["K"].values, s["T"].values, s["r"].values, s["q"].values,
                     s["iv"].values, "call")["vega"]
    vega = np.maximum(vega, 1e-4 * S0 * np.sqrt(s["T"].values))  # guard deep-OTM quotes
    mkt = s["mid"].values

    if x0 is None:
        atm = s.loc[(s["K"] / (S0 * np.exp((s["r"] - s["q"]) * s["T"])) - 1).abs().idxmin(), "iv"]
        x0 = [atm**2, 2.0, atm**2, 0.7, -0.6]
    x0 = np.clip(np.asarray(x0, dtype=float), LOWER + 1e-6, UPPER - 1e-6)

    def resid(x):
        return (_model_prices(x, s, S0) - mkt) / vega

    sol = least_squares(resid, x0, bounds=(LOWER, UPPER), method="trf",
                        x_scale="jac", max_nfev=max_nfev, verbose=verbose)
    params = HestonParams(*sol.x)

    s["model_price"] = _model_prices(sol.x, s, S0)
    s["model_iv"] = [implied_vol(p, S0, k, t, r, q, cp)
                     for p, k, t, r, q, cp in s[["model_price", "K", "T", "r", "q", "cp"]].values]
    err = (s["model_iv"] - s["iv"]).dropna()
    return CalibrationResult(params, float(np.sqrt((err**2).mean())), s, bool(sol.success), len(s))
