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
    listings: pd.DataFrame
    overrides: pd.DataFrame


def load(root: Path) -> Inputs:
    d = root / "data"
    return Inputs(
        events=pd.read_csv(d / "events_mapped.csv", parse_dates=["approval_date"]),
        prices=pd.read_parquet(d / "prices.parquet"),
        splits=pd.read_csv(d / "splits.csv", parse_dates=["date"]),
        shares=pd.read_csv(d / "shares_outstanding.csv.gz", parse_dates=["date"]),
        applications=pd.read_csv(d / "raw" / "openfda_applications.csv.gz", dtype=str),
        orig_approvals=pd.read_csv(d / "raw" / "openfda_orig_approvals.csv.gz", dtype=str),
        listings=pd.read_csv(d / "raw" / "listed_symbols.csv", dtype=str, keep_default_na=False),
        overrides=pd.read_csv(d / "sponsor_ticker_map.csv", dtype=str),
    )


# --------------------------------------------------------------------------- characteristics
def market_cap(ticker: str, asof: pd.Timestamp, prices: pd.DataFrame, splits: pd.DataFrame,
               shares: pd.DataFrame, max_gap_days: int = 400) -> tuple[float, str]:
    """Approximate market cap on `asof`: shares outstanding x close.

    Shares come from Yahoo's shares-outstanding history (the latest observation on or
    before `asof`, else the first one after it, within `max_gap_days`). yfinance closes are
    split-adjusted to today, so the close is converted back to the share count's basis by
    multiplying by every split ratio dated after the share observation.
    Returns (mcap, basis) where basis is 'before', 'after' or 'missing'.
    """
    sh = shares[shares["ticker"] == ticker]
    px = prices[(prices["ticker"] == ticker) & (prices["date"] <= asof)]
    if sh.empty or px.empty:
        return np.nan, "missing"
    gap = (sh["date"] - asof).dt.days
    before = sh[(gap <= 0) & (gap >= -max_gap_days)]
    after = sh[(gap > 0) & (gap <= max_gap_days)]
    if len(before):
        s, basis = before.sort_values("date").iloc[-1], "before"
    elif len(after):
        s, basis = after.sort_values("date").iloc[0], "after"
    else:
        return np.nan, "missing"
    sp = splits[(splits["ticker"] == ticker) & (splits["date"] > s["date"])]["ratio"].prod()
    return float(s["shares"] * px.sort_values("date")["close"].iloc[-1] * sp), basis


def prior_approvals(ev: pd.DataFrame, inp: Inputs) -> pd.Series:
    """Number of earlier original NDA/BLA approvals (any year in Drugs@FDA) for the same ticker.

    Every Drugs@FDA sponsor name is mapped with the same overrides + exact-name rule used for
    the events (no fuzzy matching here, to avoid false links). Approvals before the event date
    (excluding the event's own application) are counted.
    """
    apps = inp.applications[inp.applications["appl_type"].isin(["NDA", "BLA"])]
    o = inp.orig_approvals.drop_duplicates("appl_no").merge(apps[["appl_no", "sponsor"]], on="appl_no")
    o["approval_date"] = pd.to_datetime(o["approval_date"].astype(str), format="%Y%m%d", errors="coerce")
    o["sponsor_key"] = o["sponsor"].map(normalize_company)
    o = map_tickers(o, inp.listings, inp.overrides, date_col="approval_date")
    o = o[o["map_method"].isin(["override", "auto_exact"])].dropna(subset=["ticker"])
    hist = o[["ticker", "approval_date", "appl_no"]]
    out = []
    for _, e in ev.iterrows():
        h = hist[(hist["ticker"] == e["ticker"]) & (hist["approval_date"] < e["approval_date"])
                 & (hist["appl_no"] != e["appl_no"])]
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
    steps = [("Novel approvals 2015-2024 (Drugs@FDA: NDA type 1 + new-biologic BLAs)", len(ev))]
    ev = ev[ev["ticker"].notna()]
    steps.append(("sponsor mapped to a US exchange-listed ticker", len(ev)))
    have_px = set(inp.prices["ticker"])
    ev = ev[ev["ticker"].isin(have_px)]
    steps.append(("price history available (yfinance)", len(ev)))

    returns = to_returns(inp.prices)
    cal = returns[benchmark].dropna().index
    ev["event_date"] = day_zero(ev["approval_date"], cal).values
    ev = ev[ev["event_date"].notna()]
    # Same ticker, same day 0 (e.g. two drugs approved together): one combined event.
    ev = (ev.sort_values(["approval_date", "brand_name"])
            .groupby(["ticker", "event_date"], as_index=False)
            .agg({**{c: "first" for c in ev.columns if c not in ("ticker", "event_date", "brand_name")},
                  "brand_name": lambda s: " + ".join(map(str, s))}))
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
    mc = [market_cap(t, d, inp.prices, inp.splits, inp.shares) for t, d in zip(ev["ticker"], ev["size_date"])]
    ev["mcap_usd"] = [m for m, _ in mc]
    ev["mcap_basis"] = [b for _, b in mc]
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
