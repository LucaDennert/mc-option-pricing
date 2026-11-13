"""Payoffs: vanilla and exotic, single underlying.

A payoff receives simulated ``paths`` (n_paths x len(times)), the time grid and
a discount function, and returns the *discounted* value per path. This lets
products with intermediate cash-flows (autocallables) live in the same
framework as European payoffs.

Every payoff declares the dates it needs (``required_times``); the engine merges
them into the simulation grid so that fixings / observations are exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BGK_BETA = 0.5825971579390106  # -zeta(1/2)/sqrt(2*pi), Broadie-Glasserman-Kou (1997)


def _idx(times: np.ndarray, targets) -> np.ndarray:
    """Indices of ``targets`` inside the grid ``times`` (must be on the grid)."""
    targets = np.atleast_1d(np.asarray(targets, dtype=float))
    idx = np.searchsorted(times, targets - 1e-10)
    if np.any(idx >= len(times)) or not np.allclose(times[idx], targets, atol=1e-8):
        raise ValueError("payoff dates are not on the simulation grid")
    return idx


class Payoff:
    maturity: float

    def required_times(self) -> np.ndarray:
        return np.array([self.maturity])

    def payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def discounted(self, paths, times, discount) -> np.ndarray:
        return discount(self.maturity) * self.payoff(paths, times)

    def _terminal(self, paths, times):
        return paths[:, _idx(times, self.maturity)[0]]


# ------------------------------------------------------------------ vanilla
@dataclass
class European(Payoff):
    strike: float
    maturity: float
    cp: str = "call"

    def payoff(self, paths, times):
        s = self._terminal(paths, times)
        if self.cp == "call":
            return np.maximum(s - self.strike, 0.0)
        return np.maximum(self.strike - s, 0.0)


@dataclass
class Digital(Payoff):
    """Cash-or-nothing digital."""
    strike: float
    maturity: float
    cp: str = "call"
    cash: float = 1.0

    def payoff(self, paths, times):
        s = self._terminal(paths, times)
        hit = s > self.strike if self.cp == "call" else s < self.strike
        return self.cash * hit.astype(float)


# ------------------------------------------------------------------ Asian
@dataclass
class Asian(Payoff):
    """Fixed-strike Asian on ``n_fixings`` equally spaced dates in (0, T]."""
    strike: float
    maturity: float
    n_fixings: int = 12
    cp: str = "call"
    average: str = "arithmetic"  # or "geometric"

    @property
    def fixing_times(self) -> np.ndarray:
        return self.maturity * np.arange(1, self.n_fixings + 1) / self.n_fixings

    def required_times(self):
        return self.fixing_times

    def average_price(self, paths, times):
        fx = paths[:, _idx(times, self.fixing_times)]
        if self.average == "geometric":
            return np.exp(np.log(fx).mean(axis=1))
        return fx.mean(axis=1)

    def payoff(self, paths, times):
        a = self.average_price(paths, times)
        if self.cp == "call":
            return np.maximum(a - self.strike, 0.0)
        return np.maximum(self.strike - a, 0.0)


# ------------------------------------------------------------------ Barrier
@dataclass
class Barrier(Payoff):
    """Single knock-in / knock-out barrier option (no rebate).

    monitoring:
      * ``"discrete"``   - barrier checked on the simulation grid only;
      * ``"bgk"``        - discrete check with the Broadie-Glasserman-Kou shift
                           H -> H exp(+-beta sigma sqrt(dt)) to approximate
                           continuous monitoring;
      * ``"continuous"`` - Brownian-bridge crossing probability between grid
                           points (exact for GBM, needs ``sigma``).
    """
    strike: float
    barrier: float
    maturity: float
    cp: str = "call"
    kind: str = "down-and-out"
    monitoring: str = "discrete"
    sigma: float | None = None  # needed for "bgk" / "continuous"

    def _vanilla(self, s_T):
        if self.cp == "call":
            return np.maximum(s_T - self.strike, 0.0)
        return np.maximum(self.strike - s_T, 0.0)

    def survival(self, paths, times) -> np.ndarray:
        """Per-path probability of NOT touching the barrier (0/1 if discrete)."""
        down = self.kind.startswith("down")
        H = self.barrier
        if self.monitoring == "bgk":
            dt = np.diff(times).mean()
            # continuous barrier H  ~  discrete barrier moved towards the spot
            shift = np.exp(BGK_BETA * self.sigma * np.sqrt(dt))
            H = H * shift if down else H / shift
        breached = (paths <= H).any(axis=1) if down else (paths >= H).any(axis=1)
        alive = (~breached).astype(float)
        if self.monitoring == "continuous":
            if self.sigma is None:
                raise ValueError("continuous monitoring needs sigma")
            dt = np.diff(times)[None, :]
            a = np.log(paths[:, :-1] / H)
            b = np.log(paths[:, 1:] / H)
            # P(bridge crosses H | endpoints) = exp(-2 a b / (sigma^2 dt)) when same side
            p_cross = np.exp(-2.0 * a * b / (self.sigma**2 * dt))
            p_cross = np.where(a * b > 0, p_cross, 1.0)
            alive = alive * np.prod(1.0 - p_cross, axis=1)
        return alive

    def payoff(self, paths, times):
        alive = self.survival(paths, times)
        van = self._vanilla(self._terminal(paths, times))
        return van * alive if self.kind.endswith("out") else van * (1.0 - alive)


# ------------------------------------------------------------------ Lookback
@dataclass
class Lookback(Payoff):
    """Lookback on the simulation grid.

    kind="floating": call pays S_T - min S, put pays max S - S_T.
    kind="fixed":    call pays max(max S - K, 0), put pays max(K - min S, 0).
    If ``sigma`` is given, the discrete extremum is corrected with the
    Broadie-Glasserman-Kou factor exp(+-beta sigma sqrt(dt)).
    """
    maturity: float
    cp: str = "call"
    kind: str = "floating"
    strike: float | None = None
    sigma: float | None = None

    def payoff(self, paths, times):
        s_T = self._terminal(paths, times)
        mx, mn = paths.max(axis=1), paths.min(axis=1)
        if self.sigma is not None:
            c = np.exp(BGK_BETA * self.sigma * np.sqrt(np.diff(times).mean()))
            mx, mn = mx * c, mn / c
        if self.kind == "floating":
            return s_T - mn if self.cp == "call" else mx - s_T
        if self.cp == "call":
            return np.maximum(mx - self.strike, 0.0)
        return np.maximum(self.strike - mn, 0.0)


# ------------------------------------------------------------------ Autocallable
@dataclass
class Autocallable(Payoff):
    """Phoenix autocallable certificate (the workhorse of the Italian
    certificates market), expressed per unit notional.

    On each observation date t_i (levels relative to S0):
      * if S(t_i)/S0 >= coupon_barrier: pay coupon (+ missed coupons if memory);
      * if S(t_i)/S0 >= autocall_barrier (not at maturity): redeem 1 and stop.
    At maturity, if not called: redeem 1 if S_T/S0 >= ki_barrier, else S_T/S0
    (European knock-in put, i.e. capital at risk below the barrier).
    """
    maturity: float
    n_obs: int = 4
    autocall_barrier: float = 1.0
    coupon_barrier: float = 0.7
    coupon: float = 0.02
    ki_barrier: float = 0.6
    memory: bool = True
    obs_times: np.ndarray | None = field(default=None, repr=False)

    def _obs(self):
        if self.obs_times is not None:
            return np.asarray(self.obs_times, dtype=float)
        return self.maturity * np.arange(1, self.n_obs + 1) / self.n_obs

    def required_times(self):
        return self._obs()

    def payoff(self, paths, times):  # undiscounted is ill-defined: use discounted()
        raise NotImplementedError("use discounted(): cash-flows occur at different dates")

    def discounted(self, paths, times, discount):
        obs = self._obs()
        perf = paths[:, _idx(times, obs)] / paths[:, [0]]
        n_paths = perf.shape[0]
        pv = np.zeros(n_paths)
        alive = np.ones(n_paths, dtype=bool)
        missed = np.zeros(n_paths)
        last = len(obs) - 1
        for i, t in enumerate(obs):
            df = discount(t)
            p = perf[:, i]
            cpn_hit = alive & (p >= self.coupon_barrier)
            n_cpn = 1.0 + (missed if self.memory else 0.0)
            pv += np.where(cpn_hit, self.coupon * n_cpn * df, 0.0)
            missed = np.where(cpn_hit, 0.0, missed + alive)
            if i < last:
                called = alive & (p >= self.autocall_barrier)
                pv += np.where(called, df, 0.0)
                alive &= ~called
            else:
                redemption = np.where(p >= self.ki_barrier, 1.0, p)
                pv += np.where(alive, redemption * df, 0.0)
        return pv
