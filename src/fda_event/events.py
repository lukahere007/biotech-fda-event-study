"""Build the event table: FDA novel approvals -> applicant -> US-listed ticker.

Every step records why rows drop, so the attrition table in the README is
produced by code rather than written by hand.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- name helpers
_SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LLC", "LTD", "LIMITED",
    "PLC", "AG", "SA", "NV", "BV", "GMBH", "LP", "KK", "AB", "AS", "SPA", "SE", "HOLDINGS",
    "HOLDING", "THE", "USA", "US", "GROUP", "INTERNATIONAL", "INTL", "DBA", "ADR",
}
_ABBREV = {"PHARMS": "PHARMACEUTICALS", "PHARMA": "PHARMACEUTICALS", "PHARM": "PHARMACEUTICALS",
           "PHARMACEUTICAL": "PHARMACEUTICALS", "THERAP": "THERAPEUTICS", "LABS": "LABORATORIES",
           "BIOSCIENCE": "BIOSCIENCES", "TECHNOLOGY": "TECHNOLOGIES"}
_GENERIC = {"PHARMACEUTICALS", "THERAPEUTICS", "BIOSCIENCES", "BIOTHERAPEUTICS", "BIOPHARMA",
            "BIOPHARMACEUTICALS", "BIOTECH", "BIOTECHNOLOGY", "BIOLOGICS", "LABORATORIES",
            "MEDICAL", "SCIENCES", "HEALTHCARE", "ONCOLOGY", "AND", "OF"}


def normalize_brand(name: object) -> str:
    """'Keytruda (pembrolizumab)*' -> 'KEYTRUDA'."""
    s = str(name).upper()
    s = re.sub(r"\(.*?\)", " ", s)
    return re.sub(r"[^A-Z0-9]", "", s)


def normalize_company(name: object) -> str:
    """Canonical company key: upper case, no punctuation, no legal-form suffixes."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).upper().replace("&", " AND ")
    s = re.sub(r"\b(D/B/A|DBA)\b.*", " ", s)  # 'X Inc dba Y' -> X
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [_ABBREV.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in _SUFFIXES]
    return " ".join(toks)


def core_company(key: str) -> str:
    """Company key without generic industry words ('ALNYLAM PHARMACEUTICALS' -> 'ALNYLAM')."""
    toks = [t for t in key.split() if t not in _GENERIC]
    return " ".join(toks) if toks else key


# --------------------------------------------------------------------------- FDA joins
def match_novel_to_drugsatfda(novel: pd.DataFrame, products: pd.DataFrame,
                              orig: pd.DataFrame, apps: pd.DataFrame,
                              max_days: int = 7) -> pd.DataFrame:
    """Attach the Drugs@FDA application (NDA/BLA number, current sponsor) to each novel approval.

    Match on normalised brand name with an ORIG approval within `max_days`; fall back
    to the first word of the active ingredient.
    """
    p = products.copy()
    p["brand"] = p["DrugName"].map(normalize_brand)
    p["ingr"] = p["ActiveIngredient"].astype(str).str.upper().str.extract(r"([A-Z0-9-]+)")[0]
    o = orig[["ApplNo", "SubmissionStatusDate"]].copy()
    o["ap_date"] = pd.to_datetime(o["SubmissionStatusDate"], errors="coerce")
    cand = p[["ApplNo", "brand", "ingr"]].drop_duplicates().merge(o[["ApplNo", "ap_date"]], on="ApplNo")

    out = []
    for _, r in novel.iterrows():
        brand = normalize_brand(r["drug_name"])
        ingr = str(r.get("active_ingredient", "")).upper()
        ingr = re.findall(r"[A-Z0-9-]+", ingr)
        ingr = ingr[0] if ingr else ""
        near = cand[(cand["ap_date"] - r["approval_date"]).abs().dt.days <= max_days]
        hit, how = near[near["brand"] == brand], "brand+date"
        if hit.empty and ingr:
            hit, how = near[near["ingr"] == ingr], "ingredient+date"
        if hit.empty:
            out.append({"appl_no": None, "drugsatfda_date": pd.NaT, "match_method": "unmatched"})
            continue
        hit = hit.assign(gap=(hit["ap_date"] - r["approval_date"]).abs()).sort_values("gap")
        out.append({"appl_no": hit["ApplNo"].iloc[0], "drugsatfda_date": hit["ap_date"].iloc[0],
                    "match_method": how})
    m = pd.concat([novel.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    a = apps[["ApplNo", "ApplType", "SponsorName"]].rename(
        columns={"ApplNo": "appl_no", "ApplType": "appl_type", "SponsorName": "drugsatfda_sponsor"})
    return m.merge(a, on="appl_no", how="left")


_LETTERHEAD = re.compile(r"(DEPARTMENT OF HEALTH|FOOD AND DRUG|PUBLIC HEALTH SERVICE|SILVER SPRING|"
                         r"ROCKVILLE|NEW HAMPSHIRE AVE|^\s*(NDA|BLA)\b|APPROVAL|ACCELERATED|MD 20|"
                         r"CENTER FOR|OFFICE OF|^\s*$)", re.I)


def extract_applicant(header: object) -> str:
    """Pull the addressee company from the first page of an FDA approval letter.

    Letters are addressed as 'Company Name / Attention: Person / Title / Address'.
    We take the nearest non-letterhead line above 'Attention'. Returns '' if not found.
    """
    if not isinstance(header, str) or not header.strip():
        return ""
    lines = [ln.strip() for ln in header.splitlines()]
    for i, ln in enumerate(lines):
        if re.match(r"^(Attention|Attn)\b", ln, re.I):
            # Some letters put company + 'Attention:' on one line.
            before = re.split(r"\b(Attention|Attn)\b", ln, flags=re.I)[0].strip(" ,:")
            if before and not _LETTERHEAD.search(before):
                return before
            for j in range(i - 1, max(i - 4, -1), -1):
                if lines[j] and not _LETTERHEAD.search(lines[j]) and not re.search(r"\d{5}", lines[j]):
                    return lines[j].strip(" ,:")
            return ""
    return ""


# --------------------------------------------------------------------------- ticker mapping
def sec_lookup(sec: pd.DataFrame) -> pd.DataFrame:
    """US exchange-listed SEC registrants with normalised names (OTC excluded)."""
    s = sec.copy()
    s = s[s["exchange"].isin(["Nasdaq", "NYSE", "NYSE American", "NYSE MKT", "CBOE", "BATS"])]
    s["key"] = s["sec_name"].map(normalize_company)
    s["core"] = s["key"].map(core_company)
    # One ticker per CIK: prefer the shortest symbol without '-' (common stock over preferred/warrants).
    s = s.assign(_bad=s["ticker"].str.contains(r"[-.]").astype(int), _len=s["ticker"].str.len())
    s = s.sort_values(["cik", "_bad", "_len"]).drop_duplicates("cik")
    return s.drop(columns=["_bad", "_len"])


def auto_match(applicant_key: str, sec_l: pd.DataFrame) -> tuple[str | None, str]:
    """Exact normalised-name match, else unique core-name match. Returns (ticker, method)."""
    if not applicant_key:
        return None, "no_name"
    hit = sec_l[sec_l["key"] == applicant_key]
    if len(hit) == 1:
        return hit["ticker"].iloc[0], "auto_exact"
    core = core_company(applicant_key)
    hit = sec_l[sec_l["core"] == core]
    if len(hit) == 1 and len(core) >= 4:
        return hit["ticker"].iloc[0], "auto_core"
    return None, "no_match"


def map_tickers(ev: pd.DataFrame, sec: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Assign a ticker per event. Hand-reviewed overrides win over automatic matches.

    overrides columns: sponsor_key, ticker (blank = not US-listed), status, note.
    """
    sec_l = sec_lookup(sec)
    ov = overrides.copy()
    ov["sponsor_key"] = ov["sponsor_key"].map(normalize_company)
    ov = ov.drop_duplicates("sponsor_key", keep="last").set_index("sponsor_key")
    tick, method, status = [], [], []
    for key in ev["sponsor_key"]:
        if key in ov.index:
            t = ov.at[key, "ticker"]
            t = t if isinstance(t, str) and t.strip() else None
            tick.append(t)
            method.append("override")
            status.append(ov.at[key, "status"])
        else:
            t, m = auto_match(key, sec_l)
            tick.append(t)
            method.append(m)
            status.append("listed" if t else "unmapped")
    out = ev.copy()
    out["ticker"], out["map_method"], out["map_status"] = tick, method, status
    return out


def flag_confounded(ev: pd.DataFrame, trading_index: pd.DatetimeIndex, gap: int) -> pd.Series:
    """True for an event whose ticker had another sample approval within `gap` trading days before it.

    The first approval in a cluster is kept; later ones would double count the same price move.
    Same-day approvals for one ticker are collapsed to a single event before this is called.
    """
    pos = pd.Series(trading_index.searchsorted(ev["event_date"].values), index=ev.index)
    flag = pd.Series(False, index=ev.index)
    for _, g in ev.assign(_pos=pos).sort_values("event_date").groupby("ticker"):
        prev = None
        for idx, p in zip(g.index, g["_pos"]):
            if prev is not None and p - prev <= gap:
                flag[idx] = True
            else:
                prev = p
    return flag
