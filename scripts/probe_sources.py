"""Diagnostics: which data endpoints are reachable from the GitHub runner."""
import json
import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
B = "https://www.fda.gov/drugs/"
TESTS = [
    "https://web.archive.org/cdx/search/cdx?url=www.fda.gov/drugs/&matchType=prefix&filter=original:.*novel-drug-approvals-20[12][0-9]$&collapse=urlkey&fl=timestamp,original&limit=100",
    "https://web.archive.org/cdx/search/cdx?url=fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products/&matchType=prefix&collapse=urlkey&fl=timestamp,original&limit=100",
    "https://web.archive.org/web/2025id_/https://www.fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products/novel-drug-approvals-2024",
    "https://archive.org/wayback/available?url=accessdata.fda.gov/drugsatfda_docs/appletter/2019/212887Orig1s000ltr.pdf",
    "https://archive.org/wayback/available?url=www.sec.gov/files/company_tickers_exchange.json&timestamp=20250101",
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    "https://api.fda.gov/drug/drugsfda.json?search=application_number:NDA212887",
]
for u in TESTS:
    try:
        r = requests.get(u, headers={"User-Agent": UA}, timeout=120)
        print(f"== {r.status_code} {len(r.content)} {u}")
        print(r.text[:3000] if "drugsfda" in u or "cdx" in u or "available" in u else r.text[:300])
    except Exception as e:
        print("ERR", u, e)
import yfinance as yf
s = yf.Ticker("ALNY").get_shares_full(start="2016-01-01", end="2016-12-31")
print("shares_full", None if s is None else (len(s), s.head(3).to_dict()))
