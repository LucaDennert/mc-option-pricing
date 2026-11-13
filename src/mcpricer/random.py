"""Random number generation: pseudo-random and randomized quasi-Monte Carlo.

Everything is built on a single primitive, :meth:`RandomSource.uniforms`, so the
same model code works with pseudo-random numbers, scrambled Sobol points and
antithetic sampling. Normals are obtained by inverse-CDF (``ndtri``), which keeps
the mapping monotone - a requirement for QMC and for antithetic pairs
``(U, 1-U) -> (Z, -Z)``.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

_EPS = 1e-12


class RandomSource:
    """Stateful generator of uniform / normal draws.

    Parameters
    ----------
    method : {"pseudo", "sobol"}
        ``"pseudo"`` uses NumPy's PCG64; ``"sobol"`` uses an Owen-scrambled
        Sobol sequence (randomized QMC). Successive calls continue the sequence.
    seed : int or None
        Seed for reproducibility (also seeds the Sobol scrambling).
    antithetic : bool
        If True, each call returns ``n`` rows where the second half is the
        antithetic image ``1 - U`` of the first half.
    """

    def __init__(self, method: str = "pseudo", seed: int | None = None, antithetic: bool = False):
        if method not in ("pseudo", "sobol"):
            raise ValueError("method must be 'pseudo' or 'sobol'")
        self.method = method
        self.seed = seed
        self.antithetic = antithetic
        self._rng = np.random.default_rng(seed)
        self._sobol: qmc.Sobol | None = None
        self._sobol_dim: int | None = None

    # ------------------------------------------------------------------ core
    def _raw_uniforms(self, n: int, dim: int) -> np.ndarray:
        if self.method == "pseudo":
            return self._rng.random((n, dim))
        if self._sobol is None or self._sobol_dim != dim:
            self._sobol = qmc.Sobol(d=dim, scramble=True, seed=self._rng)
            self._sobol_dim = dim
        with warnings.catch_warnings():
            # balance properties are best for powers of two; the engine sizes
            # batches accordingly, but we do not want to spam warnings otherwise
            warnings.simplefilter("ignore", category=UserWarning)
            return self._sobol.random(n)

    def uniforms(self, n: int, dim: int) -> np.ndarray:
        """Return an ``(n, dim)`` array of U(0,1) draws, clipped away from {0, 1}."""
        if self.antithetic:
            if n % 2:
                raise ValueError("antithetic sampling needs an even number of paths")
            u = self._raw_uniforms(n // 2, dim)
            u = np.vstack([u, 1.0 - u])
        else:
            u = self._raw_uniforms(n, dim)
        return np.clip(u, _EPS, 1.0 - _EPS)

    def normals(self, n: int, dim: int) -> np.ndarray:
        """Return an ``(n, dim)`` array of N(0,1) draws via inverse CDF."""
        return ndtri(self.uniforms(n, dim))

    def spawn(self, seed: int) -> "RandomSource":
        """Independent copy with a new seed (used for RQMC replications)."""
        return RandomSource(self.method, seed, self.antithetic)


def brownian_bridge_increments(z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Map i.i.d. normals to Brownian increments using a Brownian bridge.

    The first column of ``z`` fixes the terminal value ``W(T)``, the second the
    midpoint, and so on (bisection order). With Sobol points this concentrates
    most of the variance in the first - best equidistributed - dimensions.

    Parameters
    ----------
    z : (n_paths, n_steps) standard normals
    times : (n_steps + 1,) time grid starting at 0

    Returns
    -------
    dW : (n_paths, n_steps) Brownian increments on ``times``
    """
    n_paths, n_steps = z.shape
    if len(times) != n_steps + 1:
        raise ValueError("times must have n_steps + 1 points")
    w = np.zeros((n_paths, n_steps + 1))
    w[:, -1] = np.sqrt(times[-1] - times[0]) * z[:, 0]

    # bisection order: list of (left, right) index intervals to fill
    filled = 1
    queue = [(0, n_steps)]
    while queue:
        nxt = []
        for left, right in queue:
            if right - left < 2:
                continue
            mid = (left + right) // 2
            tl, tm, tr = times[left], times[mid], times[right]
            mean = ((tr - tm) * w[:, left] + (tm - tl) * w[:, right]) / (tr - tl)
            std = np.sqrt((tm - tl) * (tr - tm) / (tr - tl))
            w[:, mid] = mean + std * z[:, filled]
            filled += 1
            nxt += [(left, mid), (mid, right)]
        queue = nxt
    return np.diff(w, axis=1)
