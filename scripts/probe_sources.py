"""Diagnostics: which FDA/SEC/openFDA endpoints are reachable from the runner."""
import re
import sys
import requests

sys.path.insert(0, "src")
from fda_event.config import SEC_USER_AGENT  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TESTS = [
    ("https://web.archive.org/cdx/search/cdx?url=fda.gov/drugs/*novel-drug-approvals-20*&output=json&fl=timestamp,original,statuscode&filter=statuscode:200&collapse=urlkey&limit=200", UA),
    ("https://www.accessdata.fda.gov/drugsatfda_docs/appletter/2019/212887Orig1s000ltr.pdf", UA),
    ("https://www.sec.gov/files/company_tickers_exchange.json", SEC_USER_AGENT),
    ("https://www.sec.gov/files/company_tickers_exchange.json", "Luke Wamalwa research lukahere007@users.noreply.github.com"),
    ("https://data.sec.gov/api/xbrl/companyfacts/CIK0001178670.json", SEC_USER_AGENT),
]
for u, ua in TESTS:
    try:
        r = requests.get(u, headers={"User-Agent": ua}, timeout=90)
        print(f"{r.status_code} {len(r.content):>10} {u[:120]} ua={ua[:30]}")
        if "cdx" in u and r.status_code == 200:
            for row in r.json()[1:]:
                print("   cdx:", row)
    except Exception as e:
        print("ERR", u, e)
try:
    import yfinance as yf
    d = yf.download(["XBI", "ALNY"], start="2019-01-01", end="2019-02-01", auto_adjust=False, progress=False)
    print("yfinance rows", len(d), list(d.columns)[:4])
except Exception as e:
    print("yfinance ERR", e)
