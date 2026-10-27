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


# --------------------------------------------------------------------------- novel approvals
_BIO_SUFFIX = re.compile(r"-[a-z]{4}\b")


def ingredient_cores(ingredients: object) -> set[str]:
    """'TRASTUZUMAB-ANNS; HYALURONIDASE-OYSK' -> {'trastuzumab', 'hyaluronidase'}.

    Biologics approved since 2017 carry a 4-letter FDA suffix; stripping it lets a
    biosimilar or reformulation be recognised as an already-approved molecule.
    """
    if not isinstance(ingredients, str):
        return set()
    out = set()
    for part in re.split(r"[;,]| and ", ingredients.lower()):
        part = _BIO_SUFFIX.sub("", part).strip()
        part = re.sub(r"\s+", " ", part)
        if part:
            out.add(part)
    return out


def define_novel_approvals(apps: pd.DataFrame, products: pd.DataFrame, orig: pd.DataFrame,
                           years: list[int]) -> pd.DataFrame:
    """Novel drug approvals (CDER definition) from Drugs@FDA original approvals.

    * NDA: original approval with chemical-type class 'TYPE 1' (new molecular entity,
      including combinations such as 'TYPE 1/4').
    * BLA: original approval whose active ingredient(s) never appeared in any earlier
      original NDA/BLA approval -> a new therapeutic biologic. This excludes biosimilars,
      new formulations and the 2020 'deemed BLA' transitions of insulin etc.
    One row per application; same-day approvals of the same molecule are collapsed.
    """
    o = orig.copy()
    o["approval_date"] = pd.to_datetime(o["approval_date"].astype(str), format="%Y%m%d", errors="coerce")
    o = o.dropna(subset=["approval_date"]).sort_values("approval_date")
    o = o.drop_duplicates("appl_no")          # first original approval per application
    prod = (products.groupby("appl_no")
                    .agg(brand_name=("brand_name", "first"),
                         active_ingredients=("active_ingredients", "first")).reset_index())
    o = o.merge(prod, on="appl_no", how="left")
    o["appl_type"] = o["appl_no"].str[:3]
    o["cores"] = o["active_ingredients"].map(ingredient_cores)

    seen: dict[str, pd.Timestamp] = {}
    is_new_bio = []
    for _, r in o.iterrows():                  # chronological
        prior = [c for c in r["cores"] if c in seen and seen[c] < r["approval_date"]]
        is_new_bio.append(r["appl_type"] == "BLA" and bool(r["cores"]) and not prior)
        for c in r["cores"]:
            seen.setdefault(c, r["approval_date"])
    o["new_biologic"] = is_new_bio
    nme = (o["appl_type"] == "NDA") & o["class_code"].fillna("").str.upper().str.startswith("TYPE 1")
    o["novel_basis"] = np.where(nme, "NDA type 1 (NME)", np.where(o["new_biologic"], "BLA new biologic", ""))
    ev = o[(o["novel_basis"] != "") & o["approval_date"].dt.year.isin(years)].copy()
    ev["core_key"] = ev["cores"].map(lambda c: "|".join(sorted(c)))
    ev = ev.drop_duplicates(["core_key", "approval_date"])
    a = apps.drop_duplicates("appl_no")[["appl_no", "sponsor"]]
    ev = ev.merge(a, on="appl_no", how="left")
    ev["year"] = ev["approval_date"].dt.year
    return ev.drop(columns=["cores", "core_key", "new_biologic"]).reset_index(drop=True)


# --------------------------------------------------------------------------- ticker mapping
_SEC_NAME_JUNK = re.compile(
    r"\s*-?\s*(class [a-z] )?(common stock|common shares|ordinary shares|american depositary shares?|"
    r"american depository shares?|ads|sponsored adr|depositary shares).*$", re.I)


def listing_lookup(listings: pd.DataFrame) -> pd.DataFrame:
    """Current exchange listings (Nasdaq, NYSE, NYSE American) with normalised issuer names."""
    s = listings[listings["exchange"].isin(["Nasdaq", "NYSE", "NYSE American"])].copy()
    s["issuer"] = s["security_name"].astype(str).str.replace(_SEC_NAME_JUNK, "", regex=True).str.strip(" -,")
    s["key"] = s["issuer"].map(normalize_company)
    s["core"] = s["key"].map(core_company)
    # One symbol per issuer: prefer plain symbols (no '$', '.', '-': preferreds, units, classes).
    s = s.assign(_bad=s["symbol"].str.contains(r"[$.\-]").astype(int), _len=s["symbol"].str.len())
    s = s.sort_values(["key", "_bad", "_len"]).drop_duplicates("key")
    return s.drop(columns=["_bad", "_len"])


def auto_match(applicant_key: str, lk: pd.DataFrame) -> tuple[str | None, str]:
    """Exact normalised-name match, else unique core-name match. Returns (ticker, method)."""
    if not applicant_key:
        return None, "no_name"
    hit = lk[lk["key"] == applicant_key]
    if len(hit) == 1:
        return hit["symbol"].iloc[0], "auto_exact"
    core = core_company(applicant_key)
    hit = lk[lk["core"] == core]
    if len(hit) == 1 and len(core) >= 4:
        return hit["symbol"].iloc[0], "auto_core"
    return None, "no_match"


def map_tickers(ev: pd.DataFrame, listings: pd.DataFrame, overrides: pd.DataFrame,
                date_col: str | None = "approval_date") -> pd.DataFrame:
    """Assign a ticker per event. Hand-reviewed overrides win over automatic matches.

    overrides columns: sponsor_key, ticker (blank = no usable US listing), valid_from,
    valid_to (optional ISO dates bounding when that sponsor name meant that ticker, e.g. a
    subsidiary maps to its acquirer only after the deal closed), status, note.
    """
    lk = listing_lookup(listings)
    ov = overrides.copy()
    ov["sponsor_key"] = ov["sponsor_key"].map(normalize_company)
    ov["valid_from"] = pd.to_datetime(ov.get("valid_from"), errors="coerce")
    ov["valid_to"] = pd.to_datetime(ov.get("valid_to"), errors="coerce")
    by_key = {k: g for k, g in ov.groupby("sponsor_key")}
    tick, method, status = [], [], []
    for _, r in ev.iterrows():
        key = r["sponsor_key"]
        if key in by_key:
            g = by_key[key]
            if date_col is not None:
                d = r[date_col]
                g = g[(g["valid_from"].isna() | (g["valid_from"] <= d)) & (g["valid_to"].isna() | (d <= g["valid_to"]))]
            if g.empty:
                tick.append(None), method.append("override"), status.append("outside_override_dates")
                continue
            t = g["ticker"].iloc[-1]
            tick.append(t if isinstance(t, str) and t.strip() else None)
            method.append("override")
            status.append(g["status"].iloc[-1])
        else:
            t, m = auto_match(key, lk)
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
