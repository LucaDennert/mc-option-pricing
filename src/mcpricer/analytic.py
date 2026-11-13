"""Closed-form and semi-analytic benchmarks.

These are used (i) to validate the Monte Carlo engine in the test-suite,
(ii) as control variates, and (iii) to calibrate Heston to market quotes.
All prices are for continuous dividend yield ``q`` and continuous rate ``r``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr as N

_SQRT2PI = math.sqrt(2 * math.pi)


def _phi(cp: str) -> int:
    cp = cp.lower()
    if cp in ("c", "call"):
        return 1
    if cp in ("p", "put"):
        return -1
    raise ValueError("cp must be 'call' or 'put'")


# ------------------------------------------------------------------ Black-Scholes
def bs_price(S, K, T, r, q, sigma, cp="call"):
    """Black-Scholes-Merton price of a European call/put (vectorised)."""
    S, K, T, sigma = map(lambda x: np.asarray(x, dtype=float), (S, K, T, sigma))
    phi = _phi(cp)
    sqT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sqT
    d2 = d1 - sqT
    return phi * (S * np.exp(-q * T) * N(phi * d1) - K * np.exp(-r * T) * N(phi * d2))


def bs_greeks(S, K, T, r, q, sigma, cp="call") -> dict:
    """Delta, gamma, vega (per 1.00 vol), theta (per year), rho."""
    phi = _phi(cp)
    sqT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sqT
    d2 = d1 - sqT
    pdf = np.exp(-0.5 * d1**2) / _SQRT2PI
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    return {
        "delta": phi * dq * N(phi * d1),
        "gamma": dq * pdf / (S * sqT),
        "vega": S * dq * pdf * np.sqrt(T),
        "theta": (-S * dq * pdf * sigma / (2 * np.sqrt(T))
                  + phi * q * S * dq * N(phi * d1) - phi * r * K * dr * N(phi * d2)),
        "rho": phi * K * T * dr * N(phi * d2),
    }


def bs_digital(S, K, T, r, q, sigma, cp="call", cash=1.0):
    """Cash-or-nothing digital paying ``cash`` if S_T > K (call) / S_T < K (put)."""
    phi = _phi(cp)
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return cash * np.exp(-r * T) * N(phi * d2)


def implied_vol(price, S, K, T, r, q, cp="call", lo=1e-4, hi=5.0):
    """Black-Scholes implied volatility by Brent's method (scalar). NaN if no root."""
    intrinsic = max(_phi(cp) * (S * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-12:
        return np.nan
    f = lambda s: float(bs_price(S, K, T, r, q, s, cp)) - price  # noqa: E731
    try:
        if f(lo) * f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-10, maxiter=200)
    except ValueError:
        return np.nan


# ------------------------------------------------------------------ Asian
def geometric_asian_price(S, K, T, r, q, sigma, fixing_times, cp="call"):
    """Discretely monitored geometric-average Asian (fixed strike), exact.

    ln G = mean_i ln S(t_i) is Gaussian with
    mean  ln S + (r - q - sigma^2/2) * mean(t_i)
    var   sigma^2 / n^2 * sum_{i,j} min(t_i, t_j).
    """
    t = np.asarray(fixing_times, dtype=float)
    n = len(t)
    mu = math.log(S) + (r - q - 0.5 * sigma**2) * t.mean()
    var = sigma**2 * np.minimum.outer(t, t).sum() / n**2
    sd = math.sqrt(var)
    phi = _phi(cp)
    d1 = (mu - math.log(K) + var) / sd
    d2 = d1 - sd
    fwd_g = math.exp(mu + 0.5 * var)
    return math.exp(-r * T) * phi * (fwd_g * N(phi * d1) - K * N(phi * d2))


# ------------------------------------------------------------------ Barriers
def barrier_price(S, K, H, T, r, q, sigma, cp="call", kind="down-and-out"):
    """Continuously monitored single barrier, no rebate (Reiner-Rubinstein / Haug).

    ``kind`` in {"down-and-out", "down-and-in", "up-and-out", "up-and-in"}.
    """
    phi = _phi(cp)
    direction, inout = kind.split("-and-")
    eta = 1 if direction == "down" else -1
    if (direction == "down" and S <= H) or (direction == "up" and S >= H):
        # already breached
        vanilla = float(bs_price(S, K, T, r, q, sigma, cp))
        return vanilla if inout == "in" else 0.0

    b = r - q
    sqT = sigma * math.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    x1 = math.log(S / K) / sqT + (1 + mu) * sqT
    x2 = math.log(S / H) / sqT + (1 + mu) * sqT
    y1 = math.log(H**2 / (S * K)) / sqT + (1 + mu) * sqT
    y2 = math.log(H / S) / sqT + (1 + mu) * sqT
    dq, dr = math.exp((b - r) * T), math.exp(-r * T)
    hs = H / S

    A = phi * S * dq * N(phi * x1) - phi * K * dr * N(phi * x1 - phi * sqT)
    B = phi * S * dq * N(phi * x2) - phi * K * dr * N(phi * x2 - phi * sqT)
    C = (phi * S * dq * hs ** (2 * (mu + 1)) * N(eta * y1)
         - phi * K * dr * hs ** (2 * mu) * N(eta * y1 - eta * sqT))
    D = (phi * S * dq * hs ** (2 * (mu + 1)) * N(eta * y2)
         - phi * K * dr * hs ** (2 * mu) * N(eta * y2 - eta * sqT))

    high = K > H
    table = {
        ("down", "in", 1): C if high else A - B + D,
        ("up", "in", 1): A if high else B - C + D,
        ("down", "in", -1): B - C + D if high else A,
        ("up", "in", -1): A - B + D if high else C,
        ("down", "out", 1): A - C if high else B - D,
        ("up", "out", 1): 0.0 if high else A - B + C - D,
        ("down", "out", -1): A - B + C - D if high else 0.0,
        ("up", "out", -1): B - D if high else A - C,
    }
    return float(table[(direction, inout, phi)])


# ------------------------------------------------------------------ Lookback
def floating_lookback_call(S, T, r, q, sigma):
    """Continuously monitored floating-strike lookback call, payoff S_T - min S
    (Goldman-Sosin-Gatto 1979), at inception (running min = S). Requires r != q."""
    b = r - q
    sqT = sigma * math.sqrt(T)
    a1 = (b + 0.5 * sigma**2) * T / sqT
    a2 = a1 - sqT
    a3 = (-b + 0.5 * sigma**2) * T / sqT
    k = sigma**2 / (2 * b)
    return (S * math.exp(-q * T) * N(a1) - S * math.exp(-q * T) * k * N(-a1)
            - S * math.exp(-r * T) * (N(a2) - k * N(-a3)))


# ------------------------------------------------------------------ Merton
def merton_price(S, K, T, r, q, sigma, lam, mu_j, sigma_j, cp="call", n_terms=60):
    """Merton (1976) jump-diffusion: Poisson mixture of Black-Scholes prices."""
    if lam <= 0:
        return float(bs_price(S, K, T, r, q, sigma, cp))
    kj = math.exp(mu_j + 0.5 * sigma_j**2) - 1
    lam_p = lam * (1 + kj)
    price = 0.0
    log_w = -lam_p * T
    for n in range(n_terms):
        if n > 0:
            log_w += math.log(lam_p * T) - math.log(n)
        sig_n = math.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - lam * kj + n * (mu_j + 0.5 * sigma_j**2) / T
        price += math.exp(log_w) * float(bs_price(S, K, T, r_n, q, sig_n, cp))
    return price


# ------------------------------------------------------------------ Heston
_GL_X, _GL_W = np.polynomial.legendre.leggauss(256)


def heston_cf(u, T, S, r, q, v0, kappa, theta, xi, rho):
    """Characteristic function E[exp(i u ln S_T)] - 'little trap' form
    (Albrecher, Mayer, Schoutens & Tistaert 2007), numerically stable."""
    iu = 1j * u
    a = kappa - rho * xi * iu
    d = np.sqrt(a**2 + xi**2 * (iu + u**2))
    g = (a - d) / (a + d)
    edt = np.exp(-d * T)
    C = (r - q) * iu * T + kappa * theta / xi**2 * (
        (a - d) * T - 2 * np.log((1 - g * edt) / (1 - g)))
    D = (a - d) / xi**2 * (1 - edt) / (1 - g * edt)
    return np.exp(C + D * v0 + iu * math.log(S))


def heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, cp="call", u_max=None):
    """European option under Heston (Gil-Pelaez inversion, Gauss-Legendre).

    Vectorised over strikes ``K`` for a single maturity ``T``.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if u_max is None:
        u_max = max(100.0, 60.0 / math.sqrt(max(v0, theta) * T))
    u = 0.5 * u_max * (_GL_X + 1)  # nodes on (0, u_max)
    w = 0.5 * u_max * _GL_W
    args = (T, S, r, q, v0, kappa, theta, xi, rho)
    cf = heston_cf(u, *args)
    cf_shift = heston_cf(u - 1j, *args)
    fwd = S * math.exp((r - q) * T)
    lnK = np.log(K)[:, None]
    e = np.exp(-1j * u[None, :] * lnK)
    iu = 1j * u[None, :]
    P1 = 0.5 + (w * np.real(e * cf_shift[None, :] / (iu * fwd))).sum(axis=1) / math.pi
    P2 = 0.5 + (w * np.real(e * cf[None, :] / iu)).sum(axis=1) / math.pi
    call = S * math.exp(-q * T) * P1 - K * math.exp(-r * T) * P2
    if _phi(cp) == 1:
        return call
    return call - S * math.exp(-q * T) + K * math.exp(-r * T)
