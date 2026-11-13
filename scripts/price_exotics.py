"""Model risk: price a book of exotics under flat-vol Black-Scholes vs calibrated Heston.

Both models reproduce the ATM-spot implied vol of each product's maturity, so any
price difference comes from the smile / skew (and its dynamics) - which is the
point: vanillas near the money agree, path-dependent and digital payoffs do not.

    python scripts/price_exotics.py                                       # synthetic
    python scripts/price_exotics.py --params results/heston_params_SPX_2023-12-29.json \
                                    --input data/raw/SPX_2023-12-29.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mcpricer import (Asian, Autocallable, Barrier, BlackScholes, Digital, European, Heston,
                      Lookback, MonteCarloEngine)
from mcpricer.analytic import heston_price, implied_vol
from mcpricer.calibration import HestonParams

ROOT = Path(__file__).resolve().parents[1]
S0 = 100.0  # products are quoted in % of spot


def rates_at(surface, T):
    """r and q of the quoted maturity closest to T."""
    m = surface.groupby("T")[["r", "q"]].first()
    row = m.iloc[np.argmin(np.abs(m.index.values - T))]
    return float(row["r"]), float(row["q"])


def book():
    return [
        ("ATM call 1Y", European(100, 1.0), 252),
        ("Digital call 105% 1Y", Digital(105, 1.0, cash=100.0), 252),
        ("Down-and-out call K100 H80 1Y", Barrier(100, 80, 1.0, "call", "down-and-out"), 252),
        ("Down-and-in put K100 H75 1Y", Barrier(100, 75, 1.0, "put", "down-and-in"), 252),
        ("Up-and-out call K100 H120 1Y", Barrier(100, 120, 1.0, "call", "up-and-out"), 252),
        ("Asian call (12 fixings) 1Y", Asian(100, 1.0, 12), 252),
        ("Floating lookback call 1Y", Lookback(1.0, "call", "floating"), 252),
        ("Phoenix autocall 3Y (quarterly, 2% cpn, KI 60%)",
         Autocallable(3.0, n_obs=12, coupon=2.0 / 100, coupon_barrier=0.7, ki_barrier=0.6), 156),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path,
                    default=ROOT / "results" / "heston_params_synthetic_spx_surface.json")
    ap.add_argument("--input", type=Path,
                    default=ROOT / "data" / "sample" / "synthetic_spx_surface.csv")
    ap.add_argument("--n-paths", type=int, default=2**17)
    args = ap.parse_args()

    p = HestonParams.from_json(args.params)
    surface = pd.read_csv(args.input)
    rows = []
    for name, payoff, steps in book():
        T = payoff.maturity
        r, q = rates_at(surface, T)
        # BS vol = Heston-implied vol of the ATM-spot (K = S0) option of the same maturity
        atm_call = heston_price(S0, S0, T, r, q, p.v0, p.kappa, p.theta, p.xi, p.rho)[0]
        atm_vol = implied_vol(atm_call, S0, S0, T, r, q, "call")

        heston = Heston(S0, r, q, p.v0, p.kappa, p.theta, p.xi, p.rho)
        bs = BlackScholes(S0, r, q, atm_vol)
        eng = MonteCarloEngine(n_paths=args.n_paths, n_steps=steps, antithetic=True,
                               batch_size=2**13, seed=2024)
        scale = 100.0 if isinstance(payoff, Autocallable) else 1.0  # per-unit notional -> %
        rb, rh = eng.price(bs, payoff), eng.price(heston, payoff)
        rows.append({
            "product": name,
            "BS (flat ATM vol)": scale * rb.price,
            "Heston": scale * rh.price,
            "diff": scale * (rh.price - rb.price),
            "Heston s.e.": scale * rh.std_error,
            "ATM vol used": f"{100 * atm_vol:.1f}%",
        })
        print(f"{name:52s} BS {scale * rb.price:8.3f}   Heston {scale * rh.price:8.3f}")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / f"exotics_bs_vs_heston_{args.params.stem.replace('heston_params_', '')}.md"
    floatfmt = (".3f", ".3f", ".3f", ".3f", ".3f")
    out.write_text("Prices in % of spot (S0 = 100); daily monitoring for path-dependent "
                   "payoffs; antithetic MC, 131k paths.\n\n"
                   + df.to_markdown(index=False, floatfmt=floatfmt) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
