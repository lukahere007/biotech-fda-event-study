"""Run the event study on the committed data snapshot and write results/ and figures/.

    python scripts/run_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fda_event import config as C  # noqa: E402
from fda_event.pipeline import abnormal_volume, build_study, groups, load, results_table  # noqa: E402
from fda_event.stats import car, car_sd, caar_path, summarize  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated palette)

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "lines.linewidth": 2})


def pct(x, d=2):
    return "n/a" if pd.isna(x) else f"{100 * x:+.{d}f}%"


def plot_caar(st, path):
    ev = st.events
    g = groups(ev)
    sets = [("Small cap (<$2B)", SERIES[1]), ("Large cap (>=$2B)", SERIES[0])]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, col in sets:
        ids = ev.loc[g[name], "event_id"]
        p = caar_path(st.ar[ids], C.PLOT_RANGE[0], C.PLOT_RANGE[1])
        ax.fill_between(p["tau"], p["lo"] * 100, p["hi"] * 100, color=col, alpha=0.15, lw=0)
        ax.plot(p["tau"], p["caar"] * 100, color=col, label=f"{name}, n={p['n'].iloc[0]}")
        ax.annotate(name.split(" (")[0], (p["tau"].iloc[-1], p["caar"].iloc[-1] * 100),
                    xytext=(4, 0), textcoords="offset points", color=INK2, va="center", fontsize=9)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Trading days relative to approval (day 0 = first trading day on/after approval date)")
    ax.set_ylabel("Cumulative average abnormal return (%)")
    ax.set_title("CAAR around FDA novel drug approvals, market model vs XBI (95% CI)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(C.PLOT_RANGE[0], C.PLOT_RANGE[1] + 5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_car_dist(ev, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(-0.4, 0.4, 41)
    for name, col in [("Large cap (>=$2B)", SERIES[0]), ("Small cap (<$2B)", SERIES[1])]:
        c = ev.loc[groups(ev)[name], "CAR[0,+1]"].dropna().clip(-0.4, 0.4)
        ax.hist(c, bins=bins, color=col, alpha=0.55, label=f"{name}, n={len(c)}",
                edgecolor=SURF, linewidth=1)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("CAR[0,+1] (clipped at ±40%)")
    ax.set_ylabel("Events")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Distribution of 2-day abnormal returns at approval", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_volume(av, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    med = av.median(axis=1)
    ax.bar(med.index, med.values, color=SERIES[0], width=0.7)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Trading days relative to approval")
    ax.set_ylabel("Median volume / normal volume")
    ax.set_title(f"Abnormal trading volume (median across {av.shape[1]} events)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    inp = load(ROOT)
    st = build_study(inp)
    ev = st.events
    st.attrition.to_csv(RES / "attrition.csv", index=False)
    tab = results_table(ev)
    tab.to_csv(RES / "car_tests.csv", index=False)
    keep = ["event_id", "ticker", "drug_name", "approval_date", "event_date", "applicant",
            "applicant_source", "mcap_usd", "size_group", "prior_approvals", "first_approval",
            "beta"] + [c for c in ev.columns if c.startswith("CAR")]
    ev[keep].to_csv(RES / "event_cars.csv", index=False)

    # Robustness: SPY instead of XBI as the market.
    st_spy = build_study(inp, benchmark="SPY")
    rob = []
    for gname, mask in groups(st_spy.events).items():
        sub = st_spy.events[mask]
        rob.append({"group": gname, "benchmark": "SPY",
                    **summarize(sub["CAR[0,+1]"], sub["SCAR[0,+1]"])})
    rob = pd.DataFrame(rob)
    rob.to_csv(RES / "robustness_spy.csv", index=False)

    # Leakage: day-by-day AAR around day 0.
    days = []
    for t in range(-5, 6):
        a = st.ar.loc[t]
        days.append({"tau": t, **{k: v for k, v in summarize(a).items()
                                  if k in ("n", "caar", "t_cs", "p_cs", "pct_positive", "p_sign")}})
    daily = pd.DataFrame(days).rename(columns={"caar": "aar"})
    daily.to_csv(RES / "daily_aar.csv", index=False)

    av = abnormal_volume(ev, inp.prices)
    av.median(axis=1).rename("median_volume_ratio").to_csv(RES / "abnormal_volume.csv")

    plot_caar(st, FIG / "caar_by_size.png")
    plot_car_dist(ev, FIG / "car_distribution.png")
    plot_volume(av, FIG / "abnormal_volume.png")

    top = ev.sort_values("CAR[0,+1]")
    summary = {
        "n_events": int(len(ev)),
        "n_tickers": int(ev["ticker"].nunique()),
        "size_groups": ev["size_group"].value_counts().to_dict(),
        "first_approval": int(ev["first_approval"].sum()),
        "largest_moves": top[["ticker", "drug_name", "event_date", "CAR[0,+1]"]].iloc[
            list(range(3)) + list(range(-3, 0))].astype(str).to_dict("records"),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(st.attrition.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    show = tab.assign(**{c: tab[c].map(pct) for c in ["caar", "median", "ci_low", "ci_high"]})
    print(show[["group", "window", "n", "caar", "median", "ci_low", "ci_high", "t_cs", "p_cs",
                "t_bmp", "p_bmp", "pct_positive", "p_sign", "p_wilcoxon"]].round(3).to_string(index=False))
    print(rob[["group", "n", "caar", "p_cs", "p_bmp", "p_sign"]].round(4).to_string(index=False))
    print(daily.round(4).to_string(index=False))
    print(av.median(axis=1).round(2).to_string())


if __name__ == "__main__":
    main()
