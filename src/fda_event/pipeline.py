"""End-to-end analysis: events + prices -> abnormal returns -> tables.

Kept separate from scripts/run_analysis.py so it can be imported in a notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .events import flag_confounded, map_tickers, normalize_company
from .model import abnormal_returns, day_zero, to_returns
from .stats import car, car_sd, summarize


@dataclass
class Inputs:
    events: pd.DataFrame
    prices: pd.DataFrame
    splits: pd.DataFrame
    shares: pd.DataFrame
    applications: pd.DataFrame
    orig_approvals: pd.DataFrame
    sec: pd.DataFrame
    overrides: pd.DataFrame


def load(root: Path) -> Inputs:
    d = root / "data"
    return Inputs(
        events=pd.read_csv(d / "events_mapped.csv", parse_dates=["approval_date"]),
        prices=pd.read_parquet(d / "prices.parquet"),
        splits=pd.read_csv(d / "splits.csv", parse_dates=["date"]),
        shares=pd.read_csv(d / "shares_outstanding.csv.gz", parse_dates=["end", "filed"]),
        applications=pd.read_csv(d / "raw" / "drugsatfda_applications.csv.gz", dtype=str),
        orig_approvals=pd.read_csv(d / "raw" / "drugsatfda_orig_approvals.csv.gz", dtype=str),
        sec=pd.read_csv(d / "raw" / "sec_company_tickers.csv", dtype={"cik": str}),
        overrides=pd.read_csv(d / "sponsor_ticker_map.csv", dtype=str),
    )


# --------------------------------------------------------------------------- characteristics
def market_cap(ticker: str, asof: pd.Timestamp, prices: pd.DataFrame, splits: pd.DataFrame,
               shares: pd.DataFrame) -> float:
    """Point-in-time market cap: shares from the latest SEC filing *filed* by `asof`
    times the close on/before `asof`, with splits after the share count's date undone.

    yfinance 'close' is split-adjusted to today, so mcap = shares_e * close_t * prod(splits after e).
    """
    sh = shares[(shares["ticker"] == ticker) & (shares["filed"] <= asof)]
    px = prices[(prices["ticker"] == ticker) & (prices["date"] <= asof)]
    if sh.empty or px.empty:
        return np.nan
    s = sh.sort_values(["filed", "end"]).iloc[-1]
    if (asof - s["end"]).days > 400:          # stale share count: do not guess
        return np.nan
    sp = splits[(splits["ticker"] == ticker) & (splits["date"] > s["end"])]["ratio"].prod()
    return float(s["shares"] * px.sort_values("date")["close"].iloc[-1] * sp)


def prior_approvals(ev: pd.DataFrame, inp: Inputs) -> pd.Series:
    """Number of earlier original NDA/BLA approvals (any year in Drugs@FDA) for the same ticker.

    Every Drugs@FDA sponsor name is mapped with the same overrides + exact-name rule used for
    the events (no fuzzy matching here, to avoid false links). Approvals before the event date
    (excluding the event's own application) are counted.
    """
    apps = inp.applications[inp.applications["ApplType"].isin(["NDA", "BLA"])]
    o = inp.orig_approvals.merge(apps[["ApplNo", "SponsorName"]], on="ApplNo")
    o["date"] = pd.to_datetime(o["SubmissionStatusDate"], errors="coerce")
    names = pd.DataFrame({"sponsor_key": o["SponsorName"].map(normalize_company).unique()})
    mapped = map_tickers(names, inp.sec, inp.overrides)
    mapped = mapped[mapped["map_method"].isin(["override", "auto_exact"])].dropna(subset=["ticker"])
    o["sponsor_key"] = o["SponsorName"].map(normalize_company)
    o = o.merge(mapped[["sponsor_key", "ticker"]], on="sponsor_key")
    # Also count earlier sample events (their applicant-at-approval may differ from current sponsor).
    hist = pd.concat([o[["ticker", "date", "ApplNo"]],
                      ev[["ticker", "approval_date", "appl_no"]].rename(
                          columns={"approval_date": "date", "appl_no": "ApplNo"})])
    hist = hist.drop_duplicates(["ticker", "ApplNo"])
    out = []
    for _, e in ev.iterrows():
        h = hist[(hist["ticker"] == e["ticker"]) & (hist["date"] < e["approval_date"])
                 & (hist["ApplNo"] != e["appl_no"])]
        out.append(len(h))
    return pd.Series(out, index=ev.index)


# --------------------------------------------------------------------------- main study
@dataclass
class Study:
    events: pd.DataFrame                  # final analysed events with characteristics
    attrition: pd.DataFrame
    ar: pd.DataFrame                      # tau x event_id
    rm: pd.DataFrame
    fits: pd.DataFrame
    cars: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_study(inp: Inputs, benchmark: str = C.BENCHMARK) -> Study:
    ev = inp.events.copy()
    steps = [("FDA CDER novel approvals 2015-2024", len(ev))]
    ev = ev[ev["appl_no"].notna()]
    steps.append(("matched to a Drugs@FDA application", len(ev)))
    ev = ev[ev["ticker"].notna()]
    steps.append(("applicant mapped to a US exchange-listed ticker", len(ev)))
    have_px = set(inp.prices["ticker"])
    ev = ev[ev["ticker"].isin(have_px)]
    steps.append(("price history available (yfinance)", len(ev)))

    returns = to_returns(inp.prices)
    cal = returns[benchmark].dropna().index
    ev["event_date"] = day_zero(ev["approval_date"], cal).values
    ev = ev[ev["event_date"].notna()]
    # Same ticker, same day 0 (e.g. two drugs approved together): one combined event.
    ev = (ev.sort_values(["approval_date", "drug_name"])
            .groupby(["ticker", "event_date"], as_index=False)
            .agg({**{c: "first" for c in ev.columns if c not in ("ticker", "event_date", "drug_name")},
                  "drug_name": lambda s: " + ".join(s)}))
    steps.append(("after merging same-ticker same-day approvals", len(ev)))
    ev["confounded"] = flag_confounded(ev, cal, C.CONFOUND_GAP).values
    ev = ev[~ev["confounded"]]
    steps.append((f"after dropping approvals <= {C.CONFOUND_GAP} trading days after another for the same ticker",
                  len(ev)))
    ev = ev.sort_values("event_date").reset_index(drop=True)
    ev["event_id"] = np.arange(len(ev))

    ar, rm, fits = abnormal_returns(ev, returns, benchmark, C.ESTIMATION_WINDOW,
                                    (C.PLOT_RANGE[0], C.PLOT_RANGE[1]), C.MIN_ESTIMATION_OBS)
    ok = fits["dropped"].isna()
    for reason, n in fits.loc[~ok, "dropped"].value_counts().items():
        steps.append((f"dropped: {reason.replace('_', ' ')}", -int(n)))
    ev = ev.merge(fits[["event_id", "beta", "sigma", "n_est", "dropped"]], on="event_id")
    ev = ev[ev["dropped"].isna()].drop(columns="dropped")
    steps.append(("final event sample", len(ev)))

    # Characteristics, measured before the run-up (at day -30, the end of the estimation window).
    pos = {d: i for i, d in enumerate(returns.index)}
    ev["size_date"] = [returns.index[pos[d] + C.ESTIMATION_WINDOW[1]] for d in ev["event_date"]]
    ev["mcap_usd"] = [market_cap(t, d, inp.prices, inp.splits, inp.shares)
                      for t, d in zip(ev["ticker"], ev["size_date"])]
    ev["size_group"] = np.select([ev["mcap_usd"] < C.SMALL_CAP_USD, ev["mcap_usd"] >= C.SMALL_CAP_USD],
                                 ["small (<$2B)", "large (>=$2B)"], "unknown")
    ev["prior_approvals"] = prior_approvals(ev, inp).values
    ev["first_approval"] = ev["prior_approvals"] == 0

    cars = pd.DataFrame({"event_id": ev["event_id"]}).set_index("event_id")
    for name, w in C.EVENT_WINDOWS.items():
        c = car(ar, w)
        cars[f"CAR{name}"] = c
        cars[f"SCAR{name}"] = c / car_sd(fits, rm, w)
    ev = ev.merge(cars, left_on="event_id", right_index=True)
    return Study(ev, pd.DataFrame(steps, columns=["step", "events"]), ar[ev["event_id"]],
                 rm[ev["event_id"]], fits[fits["event_id"].isin(ev["event_id"])], cars)


def groups(ev: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the subsamples reported in the README."""
    return {
        "All events": pd.Series(True, index=ev.index),
        "Small cap (<$2B)": ev["size_group"].eq("small (<$2B)"),
        "Large cap (>=$2B)": ev["size_group"].eq("large (>=$2B)"),
        "First approval for company": ev["first_approval"],
        "Company had prior approvals": ~ev["first_approval"],
        "Small cap & first approval": ev["size_group"].eq("small (<$2B)") & ev["first_approval"],
    }


def results_table(ev: pd.DataFrame, windows=C.EVENT_WINDOWS) -> pd.DataFrame:
    rows = []
    for gname, mask in groups(ev).items():
        sub = ev[mask]
        for wname in windows:
            s = summarize(sub[f"CAR{wname}"], sub[f"SCAR{wname}"])
            rows.append({"group": gname, "window": wname, **s})
    return pd.DataFrame(rows)


def abnormal_volume(ev: pd.DataFrame, prices: pd.DataFrame, taus=range(-10, 11)) -> pd.DataFrame:
    """Median across events of volume(tau) / median volume in the estimation window."""
    vol = prices.pivot_table(index="date", columns="ticker", values="volume").sort_index()
    pos = {d: i for i, d in enumerate(vol.index)}
    out = {}
    for _, e in ev.iterrows():
        if e["ticker"] not in vol.columns or e["event_date"] not in pos:
            continue
        v = vol[e["ticker"]]
        p = pos[e["event_date"]]
        base = v.iloc[max(p + C.ESTIMATION_WINDOW[0], 0): p + C.ESTIMATION_WINDOW[1] + 1].median()
        if not base or np.isnan(base) or p + max(taus) >= len(v):
            continue
        out[e["event_id"]] = pd.Series([v.iloc[p + t] / base for t in taus], index=list(taus))
    return pd.DataFrame(out)
