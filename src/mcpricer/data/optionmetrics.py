"""OptionMetrics IvyDB US (via WRDS) -> clean implied-volatility surface.

Pipeline
--------
1. ``fetch_raw``      - one trading day of option quotes (``optionm.opprcdYYYY``),
                        the underlying close (``optionm.secprdYYYY``) and the
                        zero-coupon curve (``optionm.zerocd``).
2. ``build_surface``  - cleaning filters, put-call-parity implied forwards
                        (hence implied dividend yield per expiry), OTM selection
                        and our own implied vols, consistent with r and q.

Requires a WRDS account with an OptionMetrics subscription and ``pip install wrds``.

.. warning::
   OptionMetrics data are licensed. Do **not** commit raw or processed WRDS data
   to a public repository - ``data/raw/`` is git-ignored for this reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytic import implied_vol

SPX_SECID = 108105  # S&P 500 index in IvyDB US

OPTION_COLS = ("secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, "
               "volume, open_interest, impl_volatility, delta, optionid, "
               "am_settlement, ss_flag")


# ----------------------------------------------------------------------------- WRDS
def connect(username: str | None = None):
    """Open a WRDS connection (prompts for password / Duo on first use)."""
    try:
        import wrds
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install wrds  (needed only for the OptionMetrics pipeline)") from e
    return wrds.Connection(wrds_username=username)


def _date(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()  # validates input -> safe to inline in SQL


def lookup_secid(db, ticker: str) -> pd.DataFrame:
    """Candidate secids for a ticker from ``optionm.secnmd`` (name history)."""
    t = "".join(ch for ch in ticker.upper() if ch.isalnum() or ch in ".-")
    return db.raw_sql(
        f"select secid, ticker, issuer, effect_date from optionm.secnmd "
        f"where ticker = '{t}' order by effect_date desc")


def _yearly(db, base: str, year: int, sql_tail: str, cols: str) -> pd.DataFrame:
    """Query ``optionm.{base}{year}``, falling back to the non-partitioned table."""
    try:
        return db.raw_sql(f"select {cols} from optionm.{base}{year} {sql_tail}")
    except Exception:
        return db.raw_sql(f"select {cols} from optionm.{base} {sql_tail}")


def fetch_raw(db, date, secid: int = SPX_SECID) -> dict:
    """Download quotes, spot and zero curve for one date. Returns a dict of DataFrames."""
    d = _date(date)
    secid = int(secid)
    where = f"where secid = {secid} and date = '{d:%Y-%m-%d}'"
    options = _yearly(db, "opprcd", d.year, where, OPTION_COLS)
    spot = _yearly(db, "secprd", d.year, where, "secid, date, close")
    zero = db.raw_sql(f"select date, days, rate from optionm.zerocd "
                      f"where date = '{d:%Y-%m-%d}' order by days")
    if options.empty or spot.empty or zero.empty:
        raise ValueError(f"no OptionMetrics data for secid={secid} on {d.date()} "
                         "(holiday, or outside your subscription's coverage?)")
    return {"options": options, "spot": spot, "zero": zero}


# ----------------------------------------------------------------------------- cleaning
def zero_rate(zero: pd.DataFrame, days) -> np.ndarray:
    """Continuously-compounded zero rate (decimal) interpolated at ``days``."""
    z = zero.sort_values("days")
    return np.interp(np.asarray(days, dtype=float), z["days"].values, z["rate"].values / 100.0)


def implied_forward(chain: pd.DataFrame, df: float, n_strikes: int = 5) -> float:
    """Put-call-parity forward: F = K + (C - P) / DF, median over the strikes
    where |C - P| is smallest (closest to ATM, tightest quotes)."""
    c = chain[chain["cp"] == "call"].set_index("K")["mid"]
    p = chain[chain["cp"] == "put"].set_index("K")["mid"]
    both = pd.concat([c, p], axis=1, keys=["C", "P"]).dropna()
    if both.empty:
        return np.nan
    both["diff"] = both["C"] - both["P"]
    near = both.reindex(both["diff"].abs().sort_values().index[:n_strikes])
    return float(np.median(near.index.values + near["diff"].values / df))


def build_surface(raw: dict, min_days: int = 7, max_days: int = 730,
                  moneyness=(0.7, 1.3), min_mid: float = 0.10, max_rel_spread: float = 0.5,
                  monthly_only: bool = True, max_expiries: int | None = 10) -> tuple[pd.DataFrame, float]:
    """Clean raw OptionMetrics quotes into an OTM implied-vol surface.

    Returns ``(surface, S0)`` where ``surface`` has columns
    date, exdate, days, T, K, cp, bid, ask, mid, iv_om, iv, F, r, q, volume, open_interest.
    """
    o = raw["options"].copy()
    S0 = float(abs(raw["spot"]["close"].iloc[0]))
    o["date"] = pd.to_datetime(o["date"])
    o["exdate"] = pd.to_datetime(o["exdate"])
    o["days"] = (o["exdate"] - o["date"]).dt.days
    o["T"] = o["days"] / 365.0
    o["K"] = o["strike_price"] / 1000.0           # IvyDB stores strike x 1000
    o["cp"] = np.where(o["cp_flag"].str.upper() == "C", "call", "put")
    o["bid"], o["ask"] = o["best_bid"], o["best_offer"]
    o["mid"] = 0.5 * (o["bid"] + o["ask"])
    o["iv_om"] = o["impl_volatility"]

    # --- basic quote filters
    m = (o["ss_flag"].astype(str) == "0")                    # standard settlement
    m &= (o["bid"] > 0) & (o["ask"] > o["bid"]) & (o["mid"] >= min_mid)
    m &= ((o["ask"] - o["bid"]) / o["mid"]) <= max_rel_spread
    m &= o["days"].between(min_days, max_days)
    if monthly_only and "am_settlement" in o:
        m &= o["am_settlement"] == 1                         # SPX AM-settled monthlies
    o = o[m].copy()

    # --- choose expiries (spread across the term structure)
    exps = np.sort(o["exdate"].unique())
    if max_expiries and len(exps) > max_expiries:
        exps = exps[np.unique(np.linspace(0, len(exps) - 1, max_expiries).round().astype(int))]
    o = o[o["exdate"].isin(exps)]

    # --- rates, implied forwards and dividend yields per expiry
    zero = raw["zero"]
    rows = []
    for exd, g in o.groupby("exdate"):
        days = g["days"].iloc[0]
        T = days / 365.0
        r = float(zero_rate(zero, days))
        df = np.exp(-r * T)
        F = implied_forward(g, df)
        if not np.isfinite(F) or F <= 0:
            continue
        q = r - np.log(F / S0) / T
        g = g.assign(r=r, q=q, F=F)
        lo, hi = moneyness
        km = g["K"] / F
        otm = ((g["cp"] == "call") & (g["K"] >= F)) | ((g["cp"] == "put") & (g["K"] < F))
        rows.append(g[otm & km.between(lo, hi)])
    if not rows:
        raise ValueError("no expiry survived the filters")
    s = pd.concat(rows, ignore_index=True)

    s["iv"] = [implied_vol(p, S0, k, t, r, q, cp)
               for p, k, t, r, q, cp in s[["mid", "K", "T", "r", "q", "cp"]].values]
    s = s.dropna(subset=["iv"])
    cols = ["date", "exdate", "days", "T", "K", "cp", "bid", "ask", "mid", "iv_om", "iv",
            "F", "r", "q", "volume", "open_interest"]
    return s[cols].sort_values(["T", "K"]).reset_index(drop=True), S0
