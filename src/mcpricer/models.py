"""Risk-neutral dynamics for a single underlying.

Each model exposes ``simulate(times, n_paths, source) -> paths`` returning an
array of shape ``(n_paths, len(times))`` with ``paths[:, 0] == S0``.

* :class:`BlackScholes` - GBM, exact log-normal stepping (no discretisation bias).
* :class:`Heston`       - stochastic volatility; Andersen (2008) QE scheme or
  full-truncation Euler.
* :class:`MertonJD`     - GBM + log-normal jumps (Merton 1976), exact stepping.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtri

from .random import RandomSource, brownian_bridge_increments


@dataclass(frozen=True)
class Model:
    S0: float
    r: float
    q: float

    def with_params(self, **kwargs) -> "Model":
        """Return a copy with some parameters changed (used for bump Greeks)."""
        return replace(self, **kwargs)

    def discount(self, t):
        return np.exp(-self.r * np.asarray(t, dtype=float))

    def forward(self, t):
        return self.S0 * np.exp((self.r - self.q) * np.asarray(t, dtype=float))

    def simulate(self, times: np.ndarray, n_paths: int, source: RandomSource,
                 brownian_bridge: bool = False) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class BlackScholes(Model):
    sigma: float

    def simulate(self, times, n_paths, source, brownian_bridge=False):
        times = np.asarray(times, dtype=float)
        dt = np.diff(times)
        z = source.normals(n_paths, len(dt))
        if brownian_bridge and len(dt) > 1:
            dw = brownian_bridge_increments(z, times)
        else:
            dw = z * np.sqrt(dt)
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        log_s = np.cumsum(drift + self.sigma * dw, axis=1)
        paths = np.empty((n_paths, len(times)))
        paths[:, 0] = self.S0
        paths[:, 1:] = self.S0 * np.exp(log_s)
        return paths


# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Heston(Model):
    """dS/S = (r-q) dt + sqrt(v) dW1,  dv = kappa(theta - v) dt + xi sqrt(v) dW2,
    d<W1, W2> = rho dt."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    scheme: str = "qe"  # "qe" (Andersen 2008) or "euler" (full truncation)

    @property
    def feller(self) -> float:
        """Feller ratio 2*kappa*theta / xi^2 (>1 means v never hits zero)."""
        return 2 * self.kappa * self.theta / self.xi**2

    def simulate(self, times, n_paths, source, brownian_bridge=False):
        times = np.asarray(times, dtype=float)
        dts = np.diff(times)
        n = len(dts)
        # one call -> one QMC point of dimension 2n (normals for S, uniforms for v)
        u = source.uniforms(n_paths, 2 * n)
        z_s = ndtri(u[:, :n])
        u_v = u[:, n:]

        log_s = np.full(n_paths, np.log(self.S0))
        v = np.full(n_paths, self.v0)
        out = np.empty((n_paths, n + 1))
        out[:, 0] = self.S0

        k, th, xi, rho = self.kappa, self.theta, self.xi, self.rho
        for i, dt in enumerate(dts):
            if self.scheme == "qe":
                v_new = self._qe_step(v, dt, u_v[:, i])
                # Andersen's central discretisation of the integrated variance
                g1 = g2 = 0.5
                K0 = -rho * k * th * dt / xi
                K1 = g1 * dt * (k * rho / xi - 0.5) - rho / xi
                K2 = g2 * dt * (k * rho / xi - 0.5) + rho / xi
                K3 = g1 * dt * (1 - rho**2)
                K4 = g2 * dt * (1 - rho**2)
                log_s = (log_s + (self.r - self.q) * dt + K0 + K1 * v + K2 * v_new
                         + np.sqrt(np.maximum(K3 * v + K4 * v_new, 0.0)) * z_s[:, i])
                v = v_new
            elif self.scheme == "euler":
                z_v = ndtri(u_v[:, i])
                z1 = z_s[:, i]
                z2 = rho * z1 + np.sqrt(1 - rho**2) * z_v
                vp = np.maximum(v, 0.0)
                log_s = log_s + (self.r - self.q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
                v = v + k * (th - vp) * dt + xi * np.sqrt(vp * dt) * z2
            else:
                raise ValueError(f"unknown scheme {self.scheme!r}")
            out[:, i + 1] = np.exp(log_s)
        return out

    def _qe_step(self, v, dt, u):
        """Quadratic-Exponential step for the CIR variance (Andersen 2008)."""
        k, th, xi = self.kappa, self.theta, self.xi
        e = np.exp(-k * dt)
        m = th + (v - th) * e
        s2 = v * xi**2 * e / k * (1 - e) + th * xi**2 / (2 * k) * (1 - e) ** 2
        psi = s2 / np.maximum(m, 1e-300) ** 2
        v_new = np.empty_like(v)

        quad = psi <= 1.5
        if quad.any():
            p = psi[quad]
            b2 = 2 / p - 1 + np.sqrt(2 / p) * np.sqrt(2 / p - 1)
            a = m[quad] / (1 + b2)
            zv = ndtri(u[quad])
            v_new[quad] = a * (np.sqrt(b2) + zv) ** 2
        expo = ~quad
        if expo.any():
            p = (psi[expo] - 1) / (psi[expo] + 1)
            beta = (1 - p) / m[expo]
            uu = u[expo]
            v_new[expo] = np.where(uu <= p, 0.0,
                                   np.log((1 - p) / np.maximum(1 - uu, 1e-300)) / beta)
        return v_new


# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MertonJD(Model):
    """GBM with compound-Poisson log-normal jumps: J = exp(N(mu_j, sigma_j^2))."""

    sigma: float
    lam: float
    mu_j: float
    sigma_j: float

    @property
    def kappa_j(self) -> float:
        """Mean relative jump size E[J - 1]."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

    def simulate(self, times, n_paths, source, brownian_bridge=False):
        times = np.asarray(times, dtype=float)
        dts = np.diff(times)
        n = len(dts)
        u = source.uniforms(n_paths, 3 * n)
        z = ndtri(u[:, :n])
        if brownian_bridge and n > 1:
            dw = brownian_bridge_increments(z, times)
        else:
            dw = z * np.sqrt(dts)
        # Poisson counts by inverse CDF (keeps the QMC / antithetic structure)
        n_jumps = _poisson_inverse_cdf(u[:, n:2 * n], self.lam * dts)
        z_j = ndtri(u[:, 2 * n:])
        jump_log = n_jumps * self.mu_j + np.sqrt(n_jumps) * self.sigma_j * z_j

        drift = (self.r - self.q - self.lam * self.kappa_j - 0.5 * self.sigma**2) * dts
        log_s = np.cumsum(drift + self.sigma * dw + jump_log, axis=1)
        paths = np.empty((n_paths, n + 1))
        paths[:, 0] = self.S0
        paths[:, 1:] = self.S0 * np.exp(log_s)
        return paths


def _poisson_inverse_cdf(u: np.ndarray, mean: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Vectorised Poisson quantile: smallest k with CDF(k) >= u (column means)."""
    mean = np.broadcast_to(np.asarray(mean, dtype=float), u.shape[1:])
    pmf = np.exp(-mean)
    cdf = pmf.copy()
    counts = np.zeros_like(u)
    k = 0
    while True:
        counts += u > cdf
        if np.all(cdf >= 1.0 - tol) or k > 200:
            break
        k += 1
        pmf = pmf * mean / k
        cdf = cdf + pmf
    return counts
