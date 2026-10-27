"""Diagnostics: which FDA/SEC/openFDA endpoints are reachable from the runner (prints status + links)."""
import re
import requests

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
URLS = [
    "https://www.fda.gov/",
    "https://www.fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products",
    "https://www.fda.gov/drugs/development-approval-process-drugs/novel-drug-approvals-fda",
    "https://www.fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products/novel-drug-approvals-2023",
    "https://www.fda.gov/drugs/novel-drug-approvals-fda/novel-drug-approvals-2023",
    "https://www.fda.gov/media/89850/download",
    "https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files",
    "https://www.fda.gov/drugs/drug-approvals-and-databases/compilation-cder-new-molecular-entity-nme-drug-and-new-biologic-approvals",
    "https://api.fda.gov/drug/drugsfda.json?limit=1",
    "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip",
    "https://www.sec.gov/files/company_tickers_exchange.json",
]
for u in URLS:
    for ua in (UA, "python-requests"):
        try:
            r = requests.get(u, headers={"User-Agent": ua}, timeout=60, allow_redirects=True)
            print(f"{r.status_code} {len(r.content):>10} ua={ua[:10]} {u} -> {r.url}")
            if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
                links = sorted(set(re.findall(r'href="([^"]*(?:novel|nme|drugsfda|89850|compilation)[^"]*)"', r.text, re.I)))
                for link in links[:60]:
                    print("    link:", link)
        except Exception as e:
            print("ERR", u, e)
