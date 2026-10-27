"""Fetch daily prices, splits and shares outstanding for the mapped event tickers.

Run in GitHub Actions after build_events.py. Writes:
  data/prices.parquet              date, ticker, close (split-adjusted), adj_close, volume
  data/splits.csv                  yfinance split history per ticker
  data/shares_outstanding.csv.gz   yfinance shares-outstanding history (Yahoo; SEC XBRL is
                                   unreachable from GitHub runners)
  data/market_fetch_log.json       tickers requested / returned / empty
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_event.config import BENCHMARK, OTHER_INDICES, PRICE_END, PRICE_START  # noqa: E402

DATA = ROOT / "data"


def fetch_prices(tickers: list[str], batch: int = 40) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "ticker", "close", "adj_close", "volume"])


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


def fetch_shares(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        try:
            sh = yf.Ticker(t).get_shares_full(start=PRICE_START, end=PRICE_END)
        except Exception as e:  # noqa: BLE001
            print("shares failed", t, e)
            continue
        if sh is None or len(sh) == 0:
            continue
        sh = sh[~sh.index.duplicated(keep="last")]
        idx = sh.index.tz_localize(None) if sh.index.tz is not None else sh.index
        df = pd.DataFrame({"ticker": t, "date": idx.normalize(), "shares": sh.values})
        # Daily rows are mostly repeats: keep the last value per month to stay compact.
        frames.append(df.groupby(df["date"].dt.to_period("M")).tail(1))
        time.sleep(0.3)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "date", "shares"])


def main() -> None:
    ev = pd.read_csv(DATA / "events_mapped.csv")
    tickers = sorted(ev["ticker"].dropna().unique().tolist())
    all_t = tickers + [BENCHMARK] + OTHER_INDICES
    prices = fetch_prices(all_t)
    # Second pass, one ticker at a time, for symbols the batch call returned empty
    # (batch downloads occasionally drop a symbol; delisted symbols stay empty).
    missing = [t for t in all_t if t not in set(prices["ticker"])]
    if missing:
        time.sleep(5)
        prices = pd.concat([prices, fetch_prices(missing, batch=1)], ignore_index=True)
    prices.to_parquet(DATA / "prices.parquet", index=False)
    got = set(prices["ticker"])
    fetch_splits(tickers).to_csv(DATA / "splits.csv", index=False)

    shares = fetch_shares(tickers)
    shares.to_csv(DATA / "shares_outstanding.csv.gz", index=False)

    log = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "yfinance": yf.__version__,
           "requested": len(all_t), "returned": len(got),
           "empty": sorted(set(all_t) - got), "shares_tickers": int(shares["ticker"].nunique())}
    (DATA / "market_fetch_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
