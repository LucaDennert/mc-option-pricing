"""Generate a *synthetic* SPX-like option surface (Heston + bid/ask noise).

It has exactly the schema produced by ``fetch_optionmetrics.py`` so that the
whole pipeline (calibration -> exotic pricing) runs without WRDS access, e.g. in CI.
The data are NOT market data.

    python scripts/make_synthetic_surface.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from mcpricer.analytic import heston_price, implied_vol

ROOT = Path(__file__).resolve().parents[1]
TRUE = dict(v0=0.020, kappa=1.8, theta=0.045, xi=0.95, rho=-0.75)
S0, R, Q = 4800.0, 0.045, 0.014
DAYS = [21, 49, 77, 112, 168, 259, 357, 539, 721]


def main(out=ROOT / "data" / "sample" / "synthetic_spx_surface.csv", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in DAYS:
        T = d / 365
        F = S0 * np.exp((R - Q) * T)
        sd = np.sqrt(TRUE["theta"] * T)
        k_lo, k_hi = max(0.6, np.exp(-3.5 * sd)), min(1.3, np.exp(1.8 * sd))
        K = np.unique(np.round(np.linspace(k_lo, k_hi, 25) * F / 25) * 25)
        call = heston_price(S0, K, T, R, Q, **TRUE)
        put = call - S0 * np.exp(-Q * T) + K * np.exp(-R * T)
        for k, c, p in zip(K, call, put):
            cp, px = ("call", c) if k >= F else ("put", p)
            if px < 0.5:
                continue
            half = max(0.05, 0.01 * px) * (1 + rng.random())
            mid = px + rng.normal(0, 0.25 * half)
            rows.append(dict(date="2024-01-02", exdate=str(pd.Timestamp("2024-01-02")
                                                           + pd.Timedelta(days=d))[:10],
                             days=d, T=T, K=k, cp=cp, bid=mid - half, ask=mid + half, mid=mid,
                             iv_om=np.nan, F=F, r=R, q=Q, volume=np.nan, open_interest=np.nan))
    s = pd.DataFrame(rows)
    s["iv"] = [implied_vol(m, S0, k, t, R, Q, cp) for m, k, t, cp in s[["mid", "K", "T", "cp"]].values]
    s["S0"] = S0
    s = s.dropna(subset=["iv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False)
    print(f"wrote {len(s)} synthetic quotes -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
