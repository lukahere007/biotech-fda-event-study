"""Tests on synthetic data where the true abnormal return is known."""
import numpy as np
import pandas as pd
import pytest

from fda_event.events import (core_company, extract_applicant, flag_confounded,
                              normalize_brand, normalize_company)
from fda_event.model import abnormal_returns, day_zero, market_model, to_returns
from fda_event.stats import car, car_sd, caar_path, summarize

EST, EVT, MIN_OBS = (-250, -30), (-20, 20), 120


def synthetic_market(n_days=900, n_stocks=60, beta=1.3, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=n_days)
    rm = rng.normal(0.0004, 0.012, n_days)
    rets = {"MKT": rm}
    for i in range(n_stocks):
        rets[f"S{i}"] = 0.0002 + beta * rm + rng.normal(0, sigma, n_days)
    return pd.DataFrame(rets, index=days)


def make_events(rets, n_stocks, jump=0.0, day_offset=0, seed=1):
    """One event per stock on a random day; inject `jump` into the return on day0+day_offset."""
    rng = np.random.default_rng(seed)
    rets = rets.copy()
    positions = rng.integers(300, len(rets) - 40, n_stocks)
    rows = []
    for i, p in enumerate(positions):
        rets.iloc[p + day_offset, rets.columns.get_loc(f"S{i}")] += jump
        rows.append({"event_id": i, "ticker": f"S{i}", "event_date": rets.index[p]})
    return rets, pd.DataFrame(rows)


def test_market_model_recovers_beta():
    r = synthetic_market(n_stocks=1, sigma=0.005)
    res = market_model(r["S0"], r["MKT"], 500, EST, EVT, MIN_OBS)
    assert res is not None
    assert res.beta == pytest.approx(1.3, abs=0.05)
    assert res.n_est == 221


def test_injected_day0_jump_is_recovered():
    r = synthetic_market()
    r, ev = make_events(r, 60, jump=0.05)
    ar, rm, fits = abnormal_returns(ev, r, "MKT", EST, EVT, MIN_OBS)
    assert fits["dropped"].isna().all()
    c01 = car(ar, (0, 1))
    s = summarize(c01, c01 / car_sd(fits, rm, (0, 1)))
    # true CAR = 5%; noise sd of a 2-day CAR is ~2.8%, so the mean over 60 events is ~5% +/- 1%
    assert s["caar"] == pytest.approx(0.05, abs=0.012)
    assert s["ci_low"] < 0.05 < s["ci_high"]
    assert s["p_cs"] < 1e-6 and s["p_bmp"] < 1e-6 and s["p_sign"] < 0.01
    # the pre-event window contains no jump
    pre = summarize(car(ar, (-20, -2)))
    assert abs(pre["caar"]) < 0.02


def test_no_effect_gives_no_false_positive_on_average():
    pvals = []
    for seed in range(20):
        r = synthetic_market(seed=seed)
        r, ev = make_events(r, 60, jump=0.0, seed=seed + 100)
        ar, rm, fits = abnormal_returns(ev, r, "MKT", EST, EVT, MIN_OBS)
        c = car(ar, (0, 1))
        pvals.append(summarize(c, c / car_sd(fits, rm, (0, 1)))["p_bmp"])
    # with 20 null datasets at 5%, more than 4 rejections would be very unlikely (p<0.02)
    assert sum(p < 0.05 for p in pvals) <= 4


def test_jump_on_day_plus1_lands_in_0_1_window_not_day0():
    r = synthetic_market()
    r, ev = make_events(r, 60, jump=0.05, day_offset=1)
    ar, _, _ = abnormal_returns(ev, r, "MKT", EST, EVT, MIN_OBS)
    assert ar.loc[1].mean() == pytest.approx(0.05, abs=0.01)
    assert abs(ar.loc[0].mean()) < 0.01
    path = caar_path(ar, -5, 5).set_index("tau")
    assert path.loc[1, "caar"] - path.loc[0, "caar"] == pytest.approx(0.05, abs=0.01)


def test_day_zero_rolls_weekends_and_holidays_forward():
    # 2023-07-04 (Tue) is a US market holiday; build a calendar without it.
    days = pd.bdate_range("2023-06-26", "2023-07-14").drop(pd.Timestamp("2023-07-04"))
    approvals = pd.Series(pd.to_datetime(["2023-06-30",   # Friday (trading day) -> same day
                                          "2023-07-01",   # Saturday -> Mon 07-03
                                          "2023-07-02",   # Sunday   -> Mon 07-03
                                          "2023-07-04",   # holiday  -> Wed 07-05
                                          "2023-07-20"]))  # after calendar -> NaT
    d0 = day_zero(approvals, days)
    assert list(d0[:4]) == list(pd.to_datetime(["2023-06-30", "2023-07-03", "2023-07-03", "2023-07-05"]))
    assert pd.isna(d0.iloc[4])


def test_short_history_is_dropped_with_reason():
    r = synthetic_market(n_stocks=2)
    r.iloc[:450, r.columns.get_loc("S1")] = np.nan       # listed shortly before the event
    ev = pd.DataFrame({"event_id": [0, 1, 2], "ticker": ["S0", "S1", "NOPE"],
                       "event_date": [r.index[500]] * 3})
    ar, _, fits = abnormal_returns(ev, r, "MKT", EST, EVT, MIN_OBS)
    assert list(ar.columns) == [0]
    reasons = fits.set_index("event_id")["dropped"]
    assert reasons[1] == "insufficient_estimation_window"
    assert reasons[2] == "no_price_data"


def test_car_is_nan_when_window_has_missing_day():
    ar = pd.DataFrame({"a": [0.01, 0.02, np.nan], "b": [0.01, 0.0, 0.03]}, index=[-1, 0, 1])
    c = car(ar, (-1, 1))
    assert np.isnan(c["a"]) and c["b"] == pytest.approx(0.04)


def test_to_returns_does_not_fill_gaps():
    p = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 2),
                      "ticker": ["A"] * 3 + ["B"] * 3,
                      "adj_close": [10, 11, 12.1, 5, np.nan, 5.5]})
    r = to_returns(p)
    assert r.loc["2020-01-02", "A"] == pytest.approx(0.1)
    assert np.isnan(r.loc["2020-01-03", "B"])


def test_confounded_events_flag_later_approvals_only():
    days = pd.bdate_range("2020-01-01", periods=100)
    ev = pd.DataFrame({"ticker": ["X", "X", "X", "Y"],
                       "event_date": [days[10], days[15], days[60], days[12]]})
    flag = flag_confounded(ev, days, gap=10)
    assert list(flag) == [False, True, False, False]


def test_name_normalisation():
    assert normalize_brand("Keytruda (pembrolizumab)*") == "KEYTRUDA"
    assert normalize_company("Novartis Pharms Corp") == normalize_company("NOVARTIS PHARMACEUTICALS CORPORATION")
    assert normalize_company("Merck Sharp & Dohme LLC") == "MERCK SHARP AND DOHME"
    assert core_company(normalize_company("Alnylam Pharmaceuticals, Inc.")) == "ALNYLAM"


def test_extract_applicant_from_letter_header():
    header = ("DEPARTMENT OF HEALTH AND HUMAN SERVICES\nFood and Drug Administration\n"
              "Silver Spring MD 20993\nNDA 212887\nNDA APPROVAL\nAcme Therapeutics, Inc.\n"
              "Attention: Jane Doe\nVice President, Regulatory Affairs\n")
    assert extract_applicant(header) == "Acme Therapeutics, Inc."
    assert extract_applicant("") == ""
