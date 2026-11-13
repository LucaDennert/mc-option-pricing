"""Download one day of OptionMetrics quotes from WRDS and build a clean surface.

    pip install wrds
    python scripts/fetch_optionmetrics.py --date 2023-12-29 --wrds-user <your_username>

Writes ``data/raw/<TICKER>_<date>.csv`` (git-ignored: OptionMetrics data are
licensed and must not be published).
"""
import argparse
from pathlib import Path

from mcpricer.data import optionmetrics as om

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2023-12-29", help="trading date YYYY-MM-DD")
    ap.add_argument("--secid", type=int, default=om.SPX_SECID, help="IvyDB secid (SPX = 108105)")
    ap.add_argument("--ticker", default="SPX", help="label used in the output file name")
    ap.add_argument("--wrds-user", default=None)
    ap.add_argument("--all-expiries", action="store_true", help="keep weeklies / PM-settled too")
    ap.add_argument("--max-expiries", type=int, default=10)
    args = ap.parse_args()

    db = om.connect(args.wrds_user)
    try:
        raw = om.fetch_raw(db, args.date, args.secid)
    finally:
        db.close()
    surface, S0 = om.build_surface(raw, monthly_only=not args.all_expiries,
                                   max_expiries=args.max_expiries)
    surface["S0"] = S0
    out = ROOT / "data" / "raw" / f"{args.ticker}_{args.date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    surface.to_csv(out, index=False)
    print(f"S0 = {S0:.2f}; {len(surface)} quotes on {surface['exdate'].nunique()} expiries")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
