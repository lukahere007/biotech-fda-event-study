"""Turn the raw openFDA + listing snapshots into data/events_mapped.csv.

Offline (no network), so it can be re-run locally after editing the hand-curated
data/sponsor_ticker_map.csv. Also writes data/interim/sponsor_review.csv (one row per
sponsor name with the automatic suggestion) which the hand review works from, and
data/interim/novel_counts_by_year.csv (our count vs CDER's published annual count).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_event.config import CDER_PUBLISHED_COUNTS, YEARS  # noqa: E402
from fda_event.events import (auto_match, define_novel_approvals, listing_lookup,  # noqa: E402
                              map_tickers, normalize_company)

RAW = ROOT / "data" / "raw"


def main() -> None:
    apps = pd.read_csv(RAW / "openfda_applications.csv.gz", dtype=str)
    prods = pd.read_csv(RAW / "openfda_products.csv.gz", dtype=str)
    orig = pd.read_csv(RAW / "openfda_orig_approvals.csv.gz", dtype=str)
    listings = pd.read_csv(RAW / "listed_symbols.csv", dtype=str, keep_default_na=False)
    overrides = pd.read_csv(ROOT / "data" / "sponsor_ticker_map.csv", dtype=str)

    ev = define_novel_approvals(apps, prods, orig, YEARS)
    ev["sponsor_key"] = ev["sponsor"].map(normalize_company)
    ev = map_tickers(ev, listings, overrides)
    cols = ["year", "approval_date", "appl_no", "appl_type", "novel_basis", "class_code", "review_priority",
            "brand_name", "active_ingredients", "sponsor", "sponsor_key", "ticker", "map_method", "map_status"]
    ev = ev[cols].sort_values(["approval_date", "appl_no"])
    ev.to_csv(ROOT / "data" / "events_mapped.csv", index=False)

    interim = ROOT / "data" / "interim"
    interim.mkdir(exist_ok=True)
    counts = ev.groupby("year").size().rename("this_study").to_frame()
    counts["cder_published"] = pd.Series(CDER_PUBLISHED_COUNTS)
    counts.to_csv(interim / "novel_counts_by_year.csv")

    lk = listing_lookup(listings)
    rev = (ev.groupby("sponsor_key")
             .agg(n_events=("appl_no", "size"), sponsor=("sponsor", "first"),
                  example_drug=("brand_name", "first"), first_date=("approval_date", "min"),
                  last_date=("approval_date", "max"), ticker=("ticker", "first"),
                  map_method=("map_method", "first"))
             .reset_index())
    rev["auto_ticker"] = [auto_match(k, lk)[0] for k in rev["sponsor_key"]]
    rev["auto_issuer"] = rev["auto_ticker"].map(lk.set_index("symbol")["issuer"])
    rev.sort_values("n_events", ascending=False).to_csv(interim / "sponsor_review.csv", index=False)
    print(counts.to_string())
    print(ev["map_method"].value_counts(dropna=False).to_string())
    print("events:", len(ev), "with ticker:", ev["ticker"].notna().sum())


if __name__ == "__main__":
    main()
