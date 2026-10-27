"""Fetch the raw FDA and exchange-listing inputs for the event study.

Run in GitHub Actions (the development sandbox cannot reach these hosts).
fda.gov, accessdata.fda.gov and sec.gov block GitHub-hosted runners, so the
sources are the reachable official equivalents:

  openFDA Drugs@FDA bulk file (FDA's open-data copy of Drugs@FDA)
      -> data/raw/openfda_applications.csv.gz   one row per NDA/BLA: number, type, current sponsor
      -> data/raw/openfda_products.csv.gz       brand names and active ingredients
      -> data/raw/openfda_orig_approvals.csv.gz original (ORIG) approved submissions with class code
  Nasdaq Trader symbol directories (all Nasdaq- and NYSE/other-exchange-listed securities)
      -> data/raw/listed_symbols.csv
  data/raw/fetch_log.json  URLs, timestamps, row counts
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

OPENFDA_ZIP = "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip"
NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
UA = "biotech-fda-event-study (github.com/lukahere007/biotech-fda-event-study)"


def get(url: str, tries: int = 4) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # pragma: no cover - network
            last = repr(e)
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def fetch_openfda() -> tuple[dict[str, pd.DataFrame], str]:
    z = zipfile.ZipFile(io.BytesIO(get(OPENFDA_ZIP).content))
    js = json.loads(z.read(z.namelist()[0]))
    apps, prods, subs = [], [], []
    for rec in js["results"]:
        num = rec.get("application_number", "")
        apps.append({"appl_no": num, "appl_type": num[:3], "sponsor": rec.get("sponsor_name")})
        for p in rec.get("products", []) or []:
            ingr = "; ".join(sorted({a.get("name", "") for a in p.get("active_ingredients", []) or []}))
            prods.append({"appl_no": num, "brand_name": p.get("brand_name"), "active_ingredients": ingr})
        for s in rec.get("submissions", []) or []:
            if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP":
                subs.append({"appl_no": num, "approval_date": s.get("submission_status_date"),
                             "class_code": s.get("submission_class_code"),
                             "class_desc": s.get("submission_class_code_description"),
                             "review_priority": s.get("review_priority")})
    prods = pd.DataFrame(prods).drop_duplicates()
    return ({"applications": pd.DataFrame(apps), "products": prods, "orig_approvals": pd.DataFrame(subs)},
            js.get("meta", {}).get("last_updated", ""))


def fetch_listings() -> pd.DataFrame:
    nq = pd.read_csv(io.StringIO(get(NASDAQ_LISTED).text), sep="|", dtype=str)
    nq = nq[nq["Symbol"].notna() & ~nq["Symbol"].str.startswith("File Creation")]
    nq = pd.DataFrame({"symbol": nq["Symbol"], "security_name": nq["Security Name"], "exchange": "Nasdaq",
                       "etf": nq["ETF"], "test_issue": nq["Test Issue"]})
    ot = pd.read_csv(io.StringIO(get(OTHER_LISTED).text), sep="|", dtype=str)
    ot = ot[ot["ACT Symbol"].notna() & ~ot["ACT Symbol"].str.startswith("File Creation")]
    exch = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}
    ot = pd.DataFrame({"symbol": ot["ACT Symbol"], "security_name": ot["Security Name"],
                       "exchange": ot["Exchange"].map(exch).fillna(ot["Exchange"]),
                       "etf": ot["ETF"], "test_issue": ot["Test Issue"]})
    df = pd.concat([nq, ot], ignore_index=True)
    return df[(df["test_issue"] == "N") & (df["etf"] == "N")].drop(columns=["test_issue", "etf"])


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    log = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "sources": {}}
    d, updated = fetch_openfda()
    for k, v in d.items():
        v.to_csv(RAW / f"openfda_{k}.csv.gz", index=False)
    log["sources"]["openfda_drugsfda"] = {"url": OPENFDA_ZIP, "last_updated": updated,
                                         **{k: len(v) for k, v in d.items()}}
    lst = fetch_listings()
    lst.to_csv(RAW / "listed_symbols.csv", index=False)
    log["sources"]["nasdaqtrader_symbol_directory"] = {"urls": [NASDAQ_LISTED, OTHER_LISTED], "rows": len(lst)}
    (RAW / "fetch_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    o = d["orig_approvals"]
    print(o.assign(t=o["appl_no"].str[:3]).groupby(["t", "class_code"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
