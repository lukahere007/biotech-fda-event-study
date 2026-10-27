"""Event alignment and the market model (MacKinlay 1997).

Conventions
-----------
* Returns are simple daily returns of split- and dividend-adjusted closes.
* Day 0 is the first trading day on or after the FDA approval date. FDA posts
  most approvals late in the day, often after the 4pm close, so the price
  reaction can land on day 0 (intraday news) or day +1 (after-close news).
  The headline window [0,+1] covers both cases; weekend/holiday approvals roll
  forward to the next session.
* The market model R_it = a_i + b_i R_mt + e_it is fitted by OLS on the
  estimation window [-250,-30]; abnormal returns are AR_it = R_it - a_i - b_i R_mt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def to_returns(prices: pd.DataFrame, col: str = "adj_close") -> pd.DataFrame:
    """Long price table (date, ticker, col) -> wide simple-return matrix (date x ticker)."""
    wide = prices.pivot_table(index="date", columns="ticker", values=col).sort_index()
    # Do not fill gaps: a missing price stays missing, so no fake zero returns.
    return wide.pct_change(fill_method=None)


def day_zero(event_dates: pd.Series | pd.DatetimeIndex, trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day on or after each event date (NaT if past the end of the calendar)."""
    trading_days = pd.DatetimeIndex(trading_days).sort_values()
    d = pd.to_datetime(pd.Series(event_dates)).dt.normalize()
    pos = trading_days.searchsorted(d.values, side="left")
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    ok = pos < len(trading_days)
    out[ok] = trading_days[pos[ok]]
    return out


@dataclass
class EventResult:
    alpha: float
    beta: float
    sigma: float          # residual std in the estimation window
    n_est: int
    mkt_mean: float       # mean market return in the estimation window
    mkt_ss: float         # sum of squared demeaned market returns (for prediction-error variance)
    ar: pd.Series         # abnormal returns indexed by event time (tau)
    rm: pd.Series         # market returns over the event window (needed for exact CAR variance)


def market_model(stock: pd.Series, market: pd.Series, t0_pos: int,
                 est: tuple[int, int], evt: tuple[int, int], min_obs: int) -> EventResult | None:
    """Fit the market model for one event and return abnormal returns over `evt`.

    `stock` and `market` are return series on the same trading-day index; t0_pos is the
    integer position of day 0. Returns None if there are too few estimation observations
    or the stock has no return on day 0.
    """
    n = len(stock)
    lo, hi = t0_pos + est[0], t0_pos + est[1]
    if lo < 0 or t0_pos + evt[1] >= n:
        return None
    y, x = stock.iloc[lo:hi + 1], market.iloc[lo:hi + 1]
    ok = y.notna() & x.notna()
    y, x = y[ok].to_numpy(), x[ok].to_numpy()
    if len(y) < min_obs:
        return None
    xm = x.mean()
    ss = ((x - xm) ** 2).sum()
    beta = ((x - xm) * (y - y.mean())).sum() / ss
    alpha = y.mean() - beta * xm
    resid = y - alpha - beta * x
    sigma2 = (resid ** 2).sum() / (len(y) - 2)

    taus = np.arange(evt[0], evt[1] + 1)
    rs = stock.iloc[t0_pos + evt[0]: t0_pos + evt[1] + 1].to_numpy()
    rm = market.iloc[t0_pos + evt[0]: t0_pos + evt[1] + 1].to_numpy()
    if np.isnan(stock.iloc[t0_pos]):
        return None
    ar = pd.Series(rs - alpha - beta * rm, index=taus)
    return EventResult(alpha, beta, float(np.sqrt(sigma2)), len(y), xm, ss, ar, pd.Series(rm, index=taus))


def abnormal_returns(events: pd.DataFrame, returns: pd.DataFrame, benchmark: str,
                     est: tuple[int, int], evt: tuple[int, int], min_obs: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the market model for every event.

    events needs columns event_id, ticker, event_date (day 0, a trading day in `returns`).
    Returns (AR matrix tau x event_id, market-return matrix tau x event_id, per-event fit table
    with alpha, beta, sigma, n_est, mkt_mean, mkt_ss and a drop reason for events that cannot be used).
    """
    idx = returns.index
    mkt = returns[benchmark]
    ars, rms, fits = {}, {}, []
    for _, e in events.iterrows():
        row = {"event_id": e["event_id"], "ticker": e["ticker"], "dropped": None}
        if e["ticker"] not in returns.columns:
            row["dropped"] = "no_price_data"
        elif pd.isna(e["event_date"]) or e["event_date"] not in idx:
            row["dropped"] = "no_trading_day"
        else:
            pos = idx.get_loc(e["event_date"])
            res = market_model(returns[e["ticker"]], mkt, pos, est, evt, min_obs)
            if res is None:
                s = returns[e["ticker"]]
                n_est = s.iloc[max(pos + est[0], 0):max(pos + est[1] + 1, 0)].notna().sum()
                row["dropped"] = ("insufficient_estimation_window" if n_est < min_obs
                                  else "missing_event_window_prices")
            else:
                ars[e["event_id"]], rms[e["event_id"]] = res.ar, res.rm
                row.update(alpha=res.alpha, beta=res.beta, sigma=res.sigma, n_est=res.n_est,
                           mkt_mean=res.mkt_mean, mkt_ss=res.mkt_ss)
        fits.append(row)
    return pd.DataFrame(ars), pd.DataFrame(rms), pd.DataFrame(fits)
