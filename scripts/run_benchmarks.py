"""Reproduce every benchmark figure / table in the README (pure simulation, no data needed).

    python scripts/run_benchmarks.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import _plotstyle as ps
from mcpricer import (Asian, Barrier, BlackScholes, ControlVariate, Digital, European, Heston,
                      MonteCarloEngine)
from mcpricer import analytic as A
from mcpricer.greeks import bump_greeks, likelihood_ratio_delta_digital, pathwise_delta_call

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
S0, K, T, R, Q, SIG = 100.0, 100.0, 1.0, 0.03, 0.01, 0.20
BS = BlackScholes(S0, R, Q, SIG)


# ----------------------------------------------------------------------------- 1
def convergence():
    """RMSE vs number of paths for a European call, over independent seeds."""
    true = float(A.bs_price(S0, K, T, R, Q, SIG))
    payoff = European(K, T)
    cv = ControlVariate(European(0.0, T), S0 * np.exp(-Q * T))  # discounted S_T
    ns = 2 ** np.arange(8, 18)
    seeds = range(30)
    configs = {
        "plain MC": dict(method="pseudo"),
        "antithetic": dict(method="pseudo", antithetic=True),
        "control variate (S_T)": dict(method="pseudo", cv=True),
        "Sobol (RQMC)": dict(method="sobol", n_replications=1),
    }
    res = {}
    for label, cfg in configs.items():
        rmse = []
        for n in ns:
            errs = []
            for s in seeds:
                eng = MonteCarloEngine(n_paths=int(n), n_steps=1, seed=s,
                                       method=cfg["method"], antithetic=cfg.get("antithetic", False),
                                       n_replications=cfg.get("n_replications", 16))
                p = eng.price(BS, payoff, cv if cfg.get("cv") else None).price
                errs.append(p - true)
            rmse.append(np.sqrt(np.mean(np.square(errs))))
        res[label] = np.array(rmse)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, (label, rmse) in enumerate(res.items()):
        ax.loglog(ns, rmse, marker="o", color=ps.SERIES[i], label=label)
    ref = res["plain MC"][0] * np.sqrt(ns[0] / ns)
    ax.loglog(ns, ref, ls="--", lw=1, color=ps.INK_2, label=r"$\propto N^{-1/2}$")
    ref1 = res["Sobol (RQMC)"][0] * (ns[0] / ns)
    ax.loglog(ns, ref1, ls=":", lw=1, color=ps.INK_2, label=r"$\propto N^{-1}$")
    ax.set_xlabel("number of paths N")
    ax.set_ylabel("RMSE vs Black-Scholes (30 seeds)")
    ax.set_title("European call: convergence of plain MC vs variance reduction")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "convergence_european.png")
    plt.close(fig)
    return res, ns


# ----------------------------------------------------------------------------- 2
def asian_table():
    asian = Asian(K, T, n_fixings=12)
    geo = Asian(K, T, n_fixings=12, average="geometric")
    cv = ControlVariate(geo, A.geometric_asian_price(S0, K, T, R, Q, SIG, geo.fixing_times))
    n = 2**16
    cases = [
        ("plain MC", dict(), None),
        ("antithetic", dict(antithetic=True), None),
        ("geometric-Asian control variate", dict(), cv),
        ("antithetic + control variate", dict(antithetic=True), cv),
        ("Sobol + Brownian bridge (RQMC)", dict(method="sobol"), None),
        ("Sobol + Brownian bridge + CV", dict(method="sobol"), cv),
    ]
    rows = []
    for label, kw, c in cases:
        eng = MonteCarloEngine(n_paths=n, n_steps=12, seed=1, **kw)
        r = eng.price(BS, asian, c)
        rows.append([label, r.price, r.std_error, r.elapsed])
    df = pd.DataFrame(rows, columns=["method", "price", "std error", "time (s)"])
    base = df.loc[0, "std error"] ** 2 * df.loc[0, "time (s)"]
    df["variance reduction"] = df.loc[0, "std error"] ** 2 / df["std error"] ** 2
    df["efficiency gain (var x time)"] = base / (df["std error"] ** 2 * df["time (s)"])
    (OUT / "asian_variance_reduction.md").write_text(
        f"Arithmetic Asian call, S0=K=100, T=1, 12 monthly fixings, sigma=20%, "
        f"N = {n:,} paths.\n\n"
        + df.to_markdown(index=False, floatfmt=(".4f", ".4f", ".2e", ".3f", ".0f", ".0f")) + "\n")
    return df


# ----------------------------------------------------------------------------- 3
def barrier_monitoring():
    sig, H = 0.25, 90.0
    model = BlackScholes(S0, R, Q, sig)
    exact = A.barrier_price(S0, K, H, T, R, Q, sig, "call", "down-and-out")
    steps = [4, 12, 26, 52, 126, 252, 504]
    out = {"discrete (no correction)": [], "BGK shifted barrier": [], "Brownian-bridge survival": []}
    se = []
    for m in steps:
        eng = MonteCarloEngine(n_paths=2**16, n_steps=m, seed=3, batch_size=2**13)
        for label, mon in zip(out, ["discrete", "bgk", "continuous"]):
            r = eng.price(model, Barrier(K, H, T, "call", "down-and-out", mon, sig))
            out[label].append(r.price)
        se.append(r.std_error)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(exact, color=ps.INK, lw=1, ls="--")
    ax.annotate(f"continuous-monitoring closed form = {exact:.3f}", (steps[-1], exact),
                xytext=(0, -14), textcoords="offset points", ha="right", fontsize=9,
                color=ps.INK_2)
    for i, (label, vals) in enumerate(out.items()):
        ax.semilogx(steps, vals, marker="o", color=ps.SERIES[i], label=label)
    ax.fill_between(steps, np.array(out["Brownian-bridge survival"]) - 1.96 * np.array(se),
                    np.array(out["Brownian-bridge survival"]) + 1.96 * np.array(se),
                    color=ps.SERIES[2], alpha=0.12, lw=0)
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_xlabel("monitoring dates / time steps per year")
    ax.set_ylabel("price")
    ax.set_title("Down-and-out call (K=100, H=90, vol 25%): discrete vs continuous monitoring")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "barrier_monitoring.png")
    plt.close(fig)
    return exact, steps, out


# ----------------------------------------------------------------------------- 4
def heston_schemes():
    # parameters with the Feller condition violated (realistic for equity index)
    h = Heston(S0, R, Q, v0=0.02, kappa=1.8, theta=0.045, xi=0.95, rho=-0.75)
    Kh = 90.0
    exact = A.heston_price(S0, Kh, T, R, Q, h.v0, h.kappa, h.theta, h.xi, h.rho, "put")[0]
    steps = [1, 2, 4, 8, 16, 32, 64, 128]
    res = {}
    for scheme in ["qe", "euler"]:
        bias, band = [], []
        for m in steps:
            eng = MonteCarloEngine(n_paths=2**18, n_steps=m, seed=11, antithetic=True,
                                   batch_size=2**14)
            r = eng.price(h.with_params(scheme=scheme), European(Kh, T, "put"))
            bias.append(r.price - exact)
            band.append(1.96 * r.std_error)
        res[scheme] = (np.array(bias), np.array(band))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0, color=ps.INK, lw=1)
    for i, (scheme, label) in enumerate([("qe", "Andersen QE"), ("euler", "full-truncation Euler")]):
        b, w = res[scheme]
        ax.semilogx(steps, b, marker="o", color=ps.SERIES[i], label=label, base=2)
        ax.fill_between(steps, b - w, b + w, color=ps.SERIES[i], alpha=0.12, lw=0)
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_xlabel("time steps per year")
    ax.set_ylabel("MC price - semi-analytic price")
    ax.set_title(f"Heston 1Y 90% put (Feller ratio {h.feller:.2f}): discretisation bias")
    ax.legend(loc="lower right")
    fig.savefig(OUT / "heston_schemes.png")
    plt.close(fig)
    return exact, steps, res


# ----------------------------------------------------------------------------- 5
def greeks_table():
    an = A.bs_greeks(S0, K, T, R, Q, SIG)
    eng = MonteCarloEngine(n_paths=2**17, n_steps=1, seed=5, antithetic=True)
    bump = bump_greeks(eng, BS, European(K, T))
    pw, pw_se = pathwise_delta_call(BS, K, T)
    # digital delta: analytic derivative of e^{-rT} N(d2)
    from scipy.stats import norm
    d2 = (np.log(S0 / K) + (R - Q - 0.5 * SIG**2) * T) / (SIG * np.sqrt(T))
    dig_true = np.exp(-R * T) * norm.pdf(d2) / (S0 * SIG * np.sqrt(T))
    lr, lr_se = likelihood_ratio_delta_digital(BS, K, T)
    dig_bump = bump_greeks(eng, BS, Digital(K, T))
    rows = [
        ["call delta", an["delta"], bump["delta"], f"pathwise {pw:.4f} ± {1.96 * pw_se:.4f}"],
        ["call gamma", an["gamma"], bump["gamma"], "-"],
        ["call vega", an["vega"], bump["vega"], "-"],
        ["call rho", an["rho"], bump["rho"], "-"],
        ["digital delta", dig_true, dig_bump["delta"],
         f"likelihood ratio {lr:.4f} ± {1.96 * lr_se:.4f} (pathwise = 0, biased)"],
    ]
    df = pd.DataFrame(rows, columns=["Greek", "closed form", "MC bump & revalue (CRN)",
                                     "MC single-run estimator"])
    (OUT / "greeks.md").write_text(
        "ATM 1Y call and digital, S0=K=100, sigma=20%, 131k antithetic paths; "
        "bumps use common random numbers.\n\n"
        + df.to_markdown(index=False, floatfmt=".4f") + "\n")
    return df


if __name__ == "__main__":
    print("1/5 convergence ...")
    res, ns = convergence()
    for k, v in res.items():
        print(f"   {k:28s} RMSE @N={ns[-1]}: {v[-1]:.2e}")
    print("2/5 Asian variance reduction ...")
    print(asian_table().to_string(index=False))
    print("3/5 barrier monitoring ...")
    exact, steps, out = barrier_monitoring()
    print("   exact", round(exact, 4), {k: np.round(v, 4).tolist() for k, v in out.items()})
    print("4/5 Heston schemes ...")
    exact, steps, res = heston_schemes()
    print("   exact", round(exact, 4), {k: np.round(v[0], 4).tolist() for k, v in res.items()})
    print("5/5 Greeks ...")
    print(greeks_table().to_string(index=False))
    print(f"done -> {OUT}")
