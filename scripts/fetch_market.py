"""Fetch daily prices, splits and shares outstanding for the mapped event tickers.

Run in GitHub Actions after build_events.py. Writes:
  data/prices.parquet              date, ticker, close (split-adjusted), adj_close, volume
  data/splits.csv                  yfinance split history per ticker
  data/shares_outstanding.csv.gz   SEC XBRL dei:EntityCommonStockSharesOutstanding by filing
  data/market_fetch_log.json       tickers requested / returned / empty
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_event.config import (BENCHMARK, OTHER_INDICES, PRICE_END, PRICE_START,  # noqa: E402
                              SEC_USER_AGENT)

DATA = ROOT / "data"


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        raw = yf.download(chunk, start=PRICE_START, end=PRICE_END, auto_adjust=False,
                          group_by="ticker", threads=True, progress=False)
        for t in chunk:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            sub = sub.dropna(subset=["Adj Close"])
            if sub.empty:
                continue
            frames.append(pd.DataFrame({"date": sub.index.tz_localize(None) if sub.index.tz else sub.index,
                                        "ticker": t, "close": sub["Close"].values,
                                        "adj_close": sub["Adj Close"].values,
                                        "volume": sub["Volume"].values}))
        time.sleep(2)
    return pd.concat(frames, ignore_index=True)


def fetch_splits(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            s = yf.Ticker(t).splits
        except Exception as e:  # noqa: BLE001
            print("splits failed", t, e)
            continue
        for d, ratio in s.items():
            rows.append({"ticker": t, "date": pd.Timestamp(d).tz_localize(None).normalize(), "ratio": ratio})
    return pd.DataFrame(rows, columns=["ticker", "date", "ratio"])


def fetch_shares(ticker_cik: dict[str, str]) -> pd.DataFrame:
    rows = []
    for t, cik in ticker_cik.items():
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"
        try:
            r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
            r.raise_for_status()
            facts = r.json()["facts"].get("dei", {}).get("EntityCommonStockSharesOutstanding", {})
            for item in facts.get("units", {}).get("shares", []):
                rows.append({"ticker": t, "cik": cik, "end": item["end"], "filed": item["filed"],
                             "accn": item["accn"], "form": item.get("form"), "shares": item["val"]})
        except Exception as e:  # noqa: BLE001
            print("companyfacts failed", t, e)
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    # Several share classes in one filing are reported as separate facts with the same date: sum them.
    return (df.drop_duplicates(["ticker", "accn", "end", "shares"])
              .groupby(["ticker", "cik", "accn", "end", "filed"], as_index=False)["shares"].sum())


def main() -> None:
    ev = pd.read_csv(DATA / "events_mapped.csv")
    tickers = sorted(ev["ticker"].dropna().unique().tolist())
    all_t = tickers + [BENCHMARK] + OTHER_INDICES
    prices = fetch_prices(all_t)
    prices.to_parquet(DATA / "prices.parquet", index=False)
    got = set(prices["ticker"])
    fetch_splits(tickers).to_csv(DATA / "splits.csv", index=False)

    sec = pd.read_csv(DATA / "raw" / "sec_company_tickers.csv", dtype={"cik": str})
    cik = sec.drop_duplicates("ticker").set_index("ticker")["cik"]
    shares = fetch_shares({t: cik[t] for t in tickers if t in cik.index})
    shares.to_csv(DATA / "shares_outstanding.csv.gz", index=False)

    log = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "yfinance": yf.__version__,
           "requested": len(all_t), "returned": len(got),
           "empty": sorted(set(all_t) - got), "shares_tickers": int(shares["ticker"].nunique())}
    (DATA / "market_fetch_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
