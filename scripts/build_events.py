"""Turn the raw FDA + SEC snapshots into data/events_mapped.csv.

Runs offline (no network) on the files written by fetch_sources.py, so it can be
re-run locally after editing the hand-reviewed data/sponsor_ticker_map.csv.
Also writes data/interim/sponsor_review.csv: one row per applicant name with the
automatic suggestion, which is what the hand review works from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_event.events import (extract_applicant, map_tickers, match_novel_to_drugsatfda,  # noqa: E402
                              normalize_company, sec_lookup, auto_match)

RAW = ROOT / "data" / "raw"


def main() -> None:
    novel = pd.read_csv(RAW / "fda_novel_approvals.csv", parse_dates=["approval_date"])
    apps = pd.read_csv(RAW / "drugsatfda_applications.csv.gz", dtype=str)
    prods = pd.read_csv(RAW / "drugsatfda_products.csv.gz", dtype=str)
    orig = pd.read_csv(RAW / "drugsatfda_orig_approvals.csv.gz", dtype=str)
    letters = pd.read_csv(RAW / "approval_letter_headers.csv.gz", dtype=str)
    sec = pd.read_csv(RAW / "sec_company_tickers.csv", dtype={"cik": str})
    overrides = pd.read_csv(ROOT / "data" / "sponsor_ticker_map.csv", dtype=str)

    ev = match_novel_to_drugsatfda(novel, prods, orig, apps)
    ev = ev.merge(letters[["appl_no", "letter_url", "header"]], on="appl_no", how="left")
    ev["letter_applicant"] = ev["header"].map(extract_applicant)
    ev["applicant"] = ev["letter_applicant"].where(ev["letter_applicant"].str.len() > 0,
                                                   ev["drugsatfda_sponsor"])
    ev["applicant_source"] = (ev["letter_applicant"].str.len() > 0).map(
        {True: "approval_letter", False: "drugsatfda_current_sponsor"})
    ev["sponsor_key"] = ev["applicant"].map(normalize_company)
    ev = map_tickers(ev, sec, overrides)

    cols = ["year", "drug_name", "active_ingredient", "approval_date", "appl_type", "appl_no",
            "match_method", "drugsatfda_sponsor", "letter_applicant", "applicant", "applicant_source",
            "sponsor_key", "ticker", "map_method", "map_status", "approved_use"]
    ev[cols].sort_values("approval_date").to_csv(ROOT / "data" / "events_mapped.csv", index=False)

    sec_l = sec_lookup(sec)
    rev = (ev.groupby("sponsor_key", dropna=False)
             .agg(n_events=("drug_name", "size"), example_drug=("drug_name", "first"),
                  letter_applicant=("letter_applicant", "first"),
                  drugsatfda_sponsor=("drugsatfda_sponsor", "first"),
                  first_date=("approval_date", "min"), last_date=("approval_date", "max"),
                  mapped_ticker=("ticker", "first"), map_method=("map_method", "first"))
             .reset_index())
    rev["auto_ticker"] = [auto_match(k, sec_l)[0] for k in rev["sponsor_key"]]
    names = sec_l.set_index("ticker")["sec_name"]
    rev["auto_sec_name"] = rev["auto_ticker"].map(names)
    (ROOT / "data" / "interim").mkdir(exist_ok=True)
    rev.sort_values("n_events", ascending=False).to_csv(ROOT / "data" / "interim" / "sponsor_review.csv",
                                                        index=False)
    print(ev["map_method"].value_counts(dropna=False))
    print("events:", len(ev), "with ticker:", ev["ticker"].notna().sum())


if __name__ == "__main__":
    main()
