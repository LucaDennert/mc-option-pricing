"""Monte Carlo pricing engine with variance reduction.

Supported techniques (combinable):

* antithetic variates                      (``antithetic=True``)
* control variates with optimal beta       (``control_variates=[...]``)
* randomized quasi-Monte Carlo (scrambled Sobol) + Brownian bridge
                                           (``method="sobol"``)

Standard errors are always honest:
* with antithetic sampling the i.i.d. unit is the *pair* average;
* with RQMC the error is estimated from independent scramblings (replications).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .models import Model
from .payoffs import Payoff
from .random import RandomSource


@dataclass
class ControlVariate:
    """A payoff whose discounted expectation ``expected`` is known in closed form."""
    payoff: Payoff
    expected: float


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int
    elapsed: float
    details: dict = field(default_factory=dict)

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error)

    def __repr__(self) -> str:
        return (f"MCResult(price={self.price:.6f}, se={self.std_error:.2e}, "
                f"n={self.n_paths}, t={self.elapsed:.2f}s)")


class MonteCarloEngine:
    """Price any :class:`~mcpricer.payoffs.Payoff` under any :class:`~mcpricer.models.Model`.

    Parameters
    ----------
    n_paths : total number of simulated paths
    n_steps : number of uniform time steps on [0, T]; payoff dates are merged in
    method : "pseudo" or "sobol" (randomized QMC)
    antithetic : use antithetic variates
    brownian_bridge : with Sobol, build Brownian paths by bisection (BS / Merton)
    seed : base seed (common random numbers for bump-and-revalue Greeks)
    batch_size : paths simulated at once (memory control)
    n_replications : independent scramblings for RQMC error estimation
    """

    def __init__(self, n_paths: int = 100_000, n_steps: int = 100, method: str = "pseudo",
                 antithetic: bool = False, brownian_bridge: bool = True, seed: int = 42,
                 batch_size: int = 2**15, n_replications: int = 16):
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.method = method
        self.antithetic = antithetic
        self.brownian_bridge = brownian_bridge
        self.seed = seed
        self.batch_size = int(batch_size)
        self.n_replications = int(n_replications)

    # ------------------------------------------------------------------ grid
    def time_grid(self, payoff: Payoff, extra=()) -> np.ndarray:
        req = np.concatenate([np.atleast_1d(payoff.required_times()), np.atleast_1d(extra)])
        T = float(req.max())
        grid = np.concatenate([np.linspace(0.0, T, self.n_steps + 1), req])
        grid = np.unique(np.round(grid, 12))
        keep = np.concatenate([[True], np.diff(grid) > 1e-9])
        return grid[keep]

    # ------------------------------------------------------------------ core
    def _samples(self, model, payoff, cvs, source, n, times):
        """Simulate ``n`` paths in batches; return per-unit samples Y and X."""
        ys, xs = [], []
        done = 0
        while done < n:
            m = min(self.batch_size, n - done)
            if self.antithetic and m % 2:
                m += 1
            paths = model.simulate(times, m, source, self.brownian_bridge)
            y = payoff.discounted(paths, times, model.discount)
            x = (np.column_stack([cv.payoff.discounted(paths, times, model.discount)
                                  for cv in cvs]) if cvs else np.empty((m, 0)))
            if self.antithetic:  # i.i.d. unit = average over the antithetic pair
                h = m // 2
                y = 0.5 * (y[:h] + y[h:])
                x = 0.5 * (x[:h] + x[h:])
            ys.append(y)
            xs.append(x)
            done += m
        return np.concatenate(ys), np.vstack(xs)

    @staticmethod
    def _beta(y, x):
        if x.shape[1] == 0:
            return np.zeros(0)
        xc = x - x.mean(axis=0)
        yc = y - y.mean()
        cov_xx = xc.T @ xc
        cov_xy = xc.T @ yc
        return np.linalg.lstsq(cov_xx, cov_xy, rcond=None)[0]

    def price(self, model: Model, payoff: Payoff, control_variates=None) -> MCResult:
        if isinstance(control_variates, ControlVariate):
            control_variates = [control_variates]
        cvs = list(control_variates or [])
        ex = np.array([cv.expected for cv in cvs])
        extra = [cv.payoff.required_times() for cv in cvs]
        times = self.time_grid(payoff, np.concatenate(extra) if extra else ())
        t0 = time.perf_counter()

        if self.method == "pseudo":
            src = RandomSource("pseudo", self.seed, self.antithetic)
            y, x = self._samples(model, payoff, cvs, src, self.n_paths, times)
            beta = self._beta(y, x)
            adj = y - (x - ex) @ beta if cvs else y
            price = adj.mean()
            se = adj.std(ddof=1) / np.sqrt(len(adj))
            details = {"beta": beta, "raw_price": y.mean(),
                       "raw_se": y.std(ddof=1) / np.sqrt(len(y))}
        elif self.method == "sobol":
            R = self.n_replications
            per = max(self.n_paths // R, 2)
            reps_y, reps_x = [], []
            for k in range(R):
                src = RandomSource("sobol", self.seed + 1000 * k, self.antithetic)
                y, x = self._samples(model, payoff, cvs, src, per, times)
                reps_y.append(y)
                reps_x.append(x)
            beta = self._beta(np.concatenate(reps_y), np.vstack(reps_x))
            ests = np.array([(y - (x - ex) @ beta).mean() if cvs else y.mean()
                             for y, x in zip(reps_y, reps_x)])
            price = ests.mean()
            se = ests.std(ddof=1) / np.sqrt(R) if R > 1 else np.nan
            details = {"beta": beta, "replications": ests}
        else:
            raise ValueError("method must be 'pseudo' or 'sobol'")

        details["n_grid_points"] = len(times)
        return MCResult(float(price), float(se), self.n_paths,
                        time.perf_counter() - t0, details)
