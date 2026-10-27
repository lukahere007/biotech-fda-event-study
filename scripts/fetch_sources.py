"""Fetch the raw FDA and SEC inputs for the event study.

Run in GitHub Actions (the development sandbox cannot reach fda.gov / sec.gov).
Writes compact snapshots under data/raw/:

  fda_novel_approvals.csv        CDER "Novel Drug Approvals for YYYY" tables (2015-2024)
  drugsatfda_applications.csv.gz Drugs@FDA Applications table (ApplNo, type, current sponsor)
  drugsatfda_products.csv.gz     Drugs@FDA Products table (brand / active ingredient)
  drugsatfda_orig_approvals.csv.gz  Drugs@FDA original (ORIG) approved submissions
  drugsatfda_letters.csv.gz      Drugs@FDA document index rows for ORIG approval letters
  approval_letter_headers.csv.gz First ~1,200 characters of each matched approval letter
                                 (the letter is addressed to the applicant *at approval time*)
  sec_company_tickers.csv        SEC company_tickers_exchange.json (CIK, name, ticker, exchange)
  fetch_log.json                 URLs, timestamps and row counts
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))

from fda_event.config import SEC_USER_AGENT, YEARS  # noqa: E402
from fda_event.events import match_novel_to_drugsatfda  # noqa: E402

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
FDA_INDEX = "https://www.fda.gov/drugs/development-approval-process-drugs/novel-drug-approvals-fda"
FDA_YEAR_TEMPLATES = [
    "https://www.fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products/novel-drug-approvals-{y}",
    "https://www.fda.gov/drugs/novel-drug-approvals-fda/novel-drug-approvals-{y}",
    "https://www.fda.gov/drugs/development-approval-process-drugs/novel-drug-approvals-{y}",
]
DRUGSATFDA_ZIP = "https://www.fda.gov/media/89850/download"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers_exchange.json"

LOG: dict = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "sources": {}}


def get(url: str, ua: str = BROWSER_UA, tries: int = 4, **kw) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": ua}, timeout=60, **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # pragma: no cover - network
            last = repr(e)
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


# --------------------------------------------------------------------------- FDA novel approvals
def year_urls() -> dict[int, list[str]]:
    urls = {y: [t.format(y=y) for t in FDA_YEAR_TEMPLATES] for y in YEARS}
    try:
        html = get(FDA_INDEX).text
        for href in re.findall(r'href="([^"]*novel-drug-approvals-(\d{4})[^"]*)"', html):
            link, y = href[0], int(href[1])
            if y in urls:
                full = link if link.startswith("http") else "https://www.fda.gov" + link
                urls[y].insert(0, full)
    except RuntimeError as e:
        print("index page failed:", e)
    return urls


def fetch_novel_approvals() -> pd.DataFrame:
    frames = []
    for y, cands in year_urls().items():
        for url in dict.fromkeys(cands):
            try:
                html = get(url, tries=2).text
                tables = pd.read_html(io.StringIO(html), match="Approval Date")
            except Exception as e:  # noqa: BLE001
                print(f"{y}: {url} -> {e}")
                continue
            t = max(tables, key=len)
            t.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in t.columns]
            col = {c: c for c in t.columns}
            for c in t.columns:
                lc = c.lower()
                if lc.startswith("drug name"):
                    col[c] = "drug_name"
                elif lc.startswith("active ingredient"):
                    col[c] = "active_ingredient"
                elif lc.startswith("approval date"):
                    col[c] = "approval_date"
                elif "approved use" in lc:
                    col[c] = "approved_use"
                elif lc.startswith("no"):
                    col[c] = "no"
            t = t.rename(columns=col)
            t["year"] = y
            t["source_url"] = url
            frames.append(t[[c for c in ["no", "drug_name", "active_ingredient", "approval_date",
                                         "approved_use", "year", "source_url"] if c in t.columns]])
            print(f"{y}: {len(t)} rows from {url}")
            break
        else:
            print(f"{y}: NO TABLE FOUND")
    df = pd.concat(frames, ignore_index=True)
    df["approval_date"] = pd.to_datetime(df["approval_date"].astype(str).str.strip(), format="mixed")
    df["approved_use"] = df["approved_use"].astype(str).str.slice(0, 200)
    return df


# --------------------------------------------------------------------------- Drugs@FDA
def read_txt(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    member = next(n for n in z.namelist() if n.lower() == name.lower())
    raw = z.read(member)
    return pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str, encoding="latin-1",
                       on_bad_lines="warn", quoting=3)


def fetch_drugsatfda() -> dict[str, pd.DataFrame]:
    r = get(DRUGSATFDA_ZIP)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("drugs@fda members:", z.namelist())
    apps = read_txt(z, "Applications.txt")
    prods = read_txt(z, "Products.txt")[["ApplNo", "ProductNo", "DrugName", "ActiveIngredient"]]
    subs = read_txt(z, "Submissions.txt")
    subs = subs[(subs["SubmissionType"] == "ORIG") & (subs["SubmissionStatus"] == "AP")]
    docs = read_txt(z, "ApplicationDocs.txt")
    dtypes = read_txt(z, "ApplicationsDocsType_Lookup.txt")
    letter_ids = dtypes.loc[dtypes.iloc[:, 1].str.contains("Letter", case=False, na=False),
                            dtypes.columns[0]].tolist()
    docs = docs[docs["ApplicationDocsTypeID"].isin(letter_ids) & (docs["SubmissionType"] == "ORIG")]
    return {"applications": apps, "products": prods, "orig_approvals": subs, "letters": docs}


def letter_header(url: str) -> str:
    from pypdf import PdfReader

    r = get(url, tries=2)
    reader = PdfReader(io.BytesIO(r.content))
    text = reader.pages[0].extract_text() or ""
    return re.sub(r"[ \t]+", " ", text)[:1200]


# --------------------------------------------------------------------------- SEC
def fetch_sec_tickers() -> pd.DataFrame:
    js = get(SEC_TICKERS, ua=SEC_USER_AGENT).json()
    df = pd.DataFrame(js["data"], columns=js["fields"])
    return df.rename(columns={"cik": "cik", "name": "sec_name", "ticker": "ticker", "exchange": "exchange"})


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    novel = fetch_novel_approvals()
    novel.to_csv(RAW / "fda_novel_approvals.csv", index=False)
    LOG["sources"]["fda_novel_approvals"] = {"rows": len(novel),
                                             "urls": sorted(novel["source_url"].unique().tolist())}

    d = fetch_drugsatfda()
    d["applications"].to_csv(RAW / "drugsatfda_applications.csv.gz", index=False)
    d["products"].to_csv(RAW / "drugsatfda_products.csv.gz", index=False)
    d["orig_approvals"].to_csv(RAW / "drugsatfda_orig_approvals.csv.gz", index=False)
    d["letters"].to_csv(RAW / "drugsatfda_letters.csv.gz", index=False)
    LOG["sources"]["drugsatfda"] = {"url": DRUGSATFDA_ZIP,
                                    **{k: len(v) for k, v in d.items()}}

    matched = match_novel_to_drugsatfda(novel, d["products"], d["orig_approvals"], d["applications"])
    letters = d["letters"].copy()
    letters = letters[letters["ApplNo"].isin(matched["appl_no"].dropna())]
    rows = []
    for appl_no, g in letters.groupby("ApplNo"):
        g = g.sort_values("ApplicationDocsDate")
        url = g["ApplicationDocsURL"].iloc[0]
        try:
            head = letter_header(url)
            err = ""
        except Exception as e:  # noqa: BLE001
            head, err = "", repr(e)[:200]
        rows.append({"appl_no": appl_no, "letter_url": url, "header": head, "error": err})
        time.sleep(0.3)
    hdr = pd.DataFrame(rows)
    hdr.to_csv(RAW / "approval_letter_headers.csv.gz", index=False)
    LOG["sources"]["approval_letters"] = {"requested": len(rows),
                                         "with_text": int((hdr["header"].str.len() > 50).sum()) if len(hdr) else 0}

    sec = fetch_sec_tickers()
    sec.to_csv(RAW / "sec_company_tickers.csv", index=False)
    LOG["sources"]["sec_company_tickers"] = {"url": SEC_TICKERS, "rows": len(sec)}

    (RAW / "fetch_log.json").write_text(json.dumps(LOG, indent=2))
    print(json.dumps(LOG, indent=2))


if __name__ == "__main__":
    main()
