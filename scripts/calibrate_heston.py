"""Calibrate Heston to an implied-vol surface and plot the fit.

    python scripts/calibrate_heston.py                               # synthetic sample
    python scripts/calibrate_heston.py --input data/raw/SPX_2023-12-29.csv

Outputs ``results/heston_params_<name>.json`` and ``results/heston_fit_<name>.png``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _plotstyle as ps
from mcpricer.calibration import calibrate_heston

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "sample" / "synthetic_spx_surface.csv"


def load_surface(path):
    s = pd.read_csv(path)
    return s.drop(columns="S0"), float(s["S0"].iloc[0])


def plot_fit(res, S0, out, title):
    ps.apply()
    import matplotlib.pyplot as plt
    f = res.fitted
    mats = sorted(f["T"].unique())
    ncol = 3
    nrow = int(np.ceil(len(mats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.9 * nrow), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, T in zip(axes, mats):
        g = f[f["T"] == T].sort_values("K")
        k = np.log(g["K"] / g["F"])
        ax.scatter(k, 100 * g["iv"], s=16, color=ps.SERIES[0], label="market", zorder=3,
                   edgecolor=ps.SURFACE, linewidth=0.8)
        ax.plot(k, 100 * g["model_iv"], color=ps.SERIES[1], label="Heston")
        ax.set_title(f"T = {T:.2f}y", fontsize=10)
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.set_ylabel("implied vol (%)")
    for ax in axes[len(mats):]:
        ax.set_visible(False)
    axes[0].legend(loc="upper right")
    p = res.params
    fig.suptitle(f"{title} - Heston fit  |  v0={p.v0:.4f}  kappa={p.kappa:.2f}  "
                 f"theta={p.theta:.4f}  xi={p.xi:.2f}  rho={p.rho:.2f}  |  "
                 f"IV RMSE = {100 * res.rmse_iv:.2f} vol pts",
                 x=0.01, ha="left", fontsize=11, color=ps.INK)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT)
    args = ap.parse_args()
    surface, S0 = load_surface(args.input)
    name = args.input.stem
    print(f"{name}: S0={S0:.2f}, {len(surface)} quotes, {surface['T'].nunique()} maturities")

    res = calibrate_heston(surface, S0)
    p = res.params
    print(f"v0={p.v0:.5f} kappa={p.kappa:.4f} theta={p.theta:.5f} xi={p.xi:.4f} rho={p.rho:.4f}")
    print(f"IV RMSE = {100 * res.rmse_iv:.3f} vol points | Feller 2*kappa*theta/xi^2 = "
          f"{2 * p.kappa * p.theta / p.xi**2:.2f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    p.to_json(out_dir / f"heston_params_{name}.json", S0=S0, rmse_iv=res.rmse_iv,
              source=name, n_quotes=res.n_quotes)
    title = "Synthetic SPX-like surface" if "synthetic" in name else f"OptionMetrics {name}"
    plot_fit(res, S0, out_dir / f"heston_fit_{name}.png", title)
    print(f"wrote results/heston_params_{name}.json and results/heston_fit_{name}.png")


if __name__ == "__main__":
    main()
