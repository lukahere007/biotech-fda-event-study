"""Cumulative abnormal returns and test statistics.

* Cross-sectional t: mean CAR / (sd(CAR)/sqrt(N)). Robust to event-induced variance
  changes, assumes independence across events.
* BMP (Boehmer, Musumeci & Poulsen 1991): t-test on standardised CARs,
  SCAR_i = CAR_i / sd_i(CAR) where sd_i uses the estimation-window residual
  variance plus market-model parameter uncertainty (MacKinlay 1997 eq. 15-16).
  Down-weights noisy (usually small, volatile) stocks.
* Sign test: share of positive CARs vs 50% (exact binomial).
* Wilcoxon signed-rank test on CARs (non-parametric, uses magnitudes and signs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def car(ar: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    """CAR per event over [a,b]; NaN if any day in the window is missing."""
    w = ar.loc[window[0]:window[1]]
    return w.sum(min_count=len(w)).where(w.notna().all())


def car_sd(fits: pd.DataFrame, rm: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    """Std. dev. of each event's CAR under H0 including estimation error of alpha and beta.

    Var(CAR) = s^2 * [L2 + L2^2/L1 + (sum_t (Rm_t - mean_est))^2 / SS_est]
    """
    f = fits.dropna(subset=["sigma"]).set_index("event_id")
    w = rm.loc[window[0]:window[1], f.index]
    l2 = len(w)
    dev = (w - f["mkt_mean"]).sum()
    var = f["sigma"] ** 2 * (l2 + l2 ** 2 / f["n_est"] + dev ** 2 / f["mkt_ss"])
    return np.sqrt(var)


def summarize(cars: pd.Series, scar: pd.Series | None = None, alpha: float = 0.05) -> dict:
    """All test statistics for one window / group."""
    c = cars.dropna()
    n = len(c)
    out = {"n": n, "caar": np.nan, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan,
           "t_cs": np.nan, "p_cs": np.nan, "t_bmp": np.nan, "p_bmp": np.nan,
           "pct_positive": np.nan, "p_sign": np.nan, "p_wilcoxon": np.nan}
    if n < 3:
        return out
    m, se = c.mean(), c.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    out.update(caar=m, median=c.median(), ci_low=m - tcrit * se, ci_high=m + tcrit * se,
               t_cs=m / se, p_cs=2 * stats.t.sf(abs(m / se), n - 1))
    if scar is not None:
        s = scar.reindex(c.index).dropna()
        if len(s) >= 3:
            t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
            out.update(t_bmp=t, p_bmp=2 * stats.t.sf(abs(t), len(s) - 1))
    k = int((c > 0).sum())
    out.update(pct_positive=k / n, p_sign=stats.binomtest(k, n, 0.5).pvalue,
               p_wilcoxon=stats.wilcoxon(c).pvalue)
    return out


def caar_path(ar: pd.DataFrame, start: int, end: int, alpha: float = 0.05) -> pd.DataFrame:
    """CAAR from `start` through each tau up to `end`, with a cross-sectional t CI.

    Uses only events with complete data over [start, end] so the path is a fixed sample.
    """
    w = ar.loc[start:end]
    w = w.loc[:, w.notna().all()]
    cum = w.cumsum()
    n = cum.shape[1]
    m = cum.mean(axis=1)
    se = cum.std(axis=1, ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    return pd.DataFrame({"tau": cum.index, "caar": m.values, "lo": (m - tcrit * se).values,
                         "hi": (m + tcrit * se).values, "n": n})
