import os
import re
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    import requests
except Exception:
    requests = None

# ============================================================
# KONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent
KONTOAUSZUG_CSV = BASE_DIR / "kontoauszug.csv"
BELEGE_CSV = BASE_DIR / "belege.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BETRAG_TOLERANZ = 0.01
DATUM_FENSTER_TAGE = 30

STATUS_SPALTE_MANUELL = None
DEFAULT_RATE_FALLBACK = None

BU_RATE_MAP = {"401": 0.19, "402": 0.07}
OUTGOING_KONTO_RATE = {"4400": 0.19, "4120": 0.19, "4125": 0.19}

RC_KEYWORDS = [
    "reverse charge", "rev. charge", "rc-verfahren",
    "§13b", "13b", "paragraf 13b",
    "innergemeinschaft", "innergemeinschaftlich", "intra community",
    "steuerfrei", "tax free", "vat exempt", "exempt",
    "vat 0", "mwst 0", "ust 0",
    "export", "ausfuhr", "ig-erwerb", "ig erwerb"
]

NEXTWAVE_ORANGE = "#DE6A00"
NEXTWAVE_DARK = "#111827"
NEXTWAVE_BLUE = "#2563eb"

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# ============================================================
# WECLAPP – ENV / STATUS / API-CHECK
# ============================================================

def _get_env_any(*names: str) -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""

def weclapp_base_url() -> str:
    raw = _get_env_any(
        "WECLAPP_BASE_URL",
        "WECLAPP_API_BASE_URL",
        "WECLAPP_URL",
        "WECLAPP_BASEURL",
    )
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return ""
    if not raw.lower().startswith(("http://", "https://")):
        raw = "https://" + raw
    raw = raw.rstrip("/")
    if "/webapp/api/v1" not in raw.lower():
        raw = raw + "/webapp/api/v1"
    return raw

def weclapp_token() -> str:
    raw = _get_env_any(
        "WECLAPP_API_TOKEN",
        "WECLAPP_TOKEN",
        "WECLAPP_AUTH_TOKEN",
        "WECLAPP_AUTHENTICATIONTOKEN",
        "AUTHENTICATIONTOKEN",
    )
    raw = raw.strip().strip('"').strip("'")
    return raw

def weclapp_configured() -> bool:
    return bool(weclapp_base_url() and weclapp_token())

def weclapp_headers() -> dict:
    return {"AuthenticationToken": weclapp_token()}

def log_env_keys():
    """
    Debug: zeigt NUR, ob die relevanten Keys existieren/leer sind.
    KEINE Werte (Token wird niemals geloggt).
    """
    keys = [
        "WECLAPP_BASE_URL", "WECLAPP_API_BASE_URL", "WECLAPP_URL", "WECLAPP_BASEURL",
        "WECLAPP_API_TOKEN", "WECLAPP_TOKEN", "WECLAPP_AUTH_TOKEN", "WECLAPP_AUTHENTICATIONTOKEN", "AUTHENTICATIONTOKEN",
    ]
    present = []
    for k in keys:
        v = os.getenv(k)
        if v is None:
            present.append(f"{k}=<missing>")
        else:
            present.append(f"{k}={'<set>' if str(v).strip() else '<empty>'}")
    print("[WECLAPP][ENV_KEYS] " + " | ".join(present))
    print("[WECLAPP] base_url_set:", bool(weclapp_base_url()), "token_set:", bool(weclapp_token()), "base_url:", weclapp_base_url())

def weclapp_check_company() -> tuple[bool, str]:
    if not weclapp_configured():
        return False, "nicht verbunden"

    if requests is None:
        return False, "requests fehlt (Package). Bitte 'requests' in requirements.txt aufnehmen."

    url = weclapp_base_url().rstrip("/") + "/company"
    try:
        r = requests.get(url, headers=weclapp_headers(), timeout=15)
        if r.status_code in (401, 403):
            return False, f"Token ungültig/keine Rechte (HTTP {r.status_code})"
        if r.status_code < 300:
            return True, "verbunden"
        return False, f"API-Fehler (HTTP {r.status_code})"
    except Exception as e:
        return False, f"Verbindung fehlgeschlagen: {str(e)}"

# ============================================================
# HELPER
# ============================================================

def safe_read_csv(path: Path, sep=";") -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=sep, dtype=str, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, sep=sep, dtype=str)

def find_column(df, keywords, default=None, prefer_contains=None):
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    for key in keywords:
        for orig, low in zip(cols, cols_lower):
            if key == low:
                return orig

    candidates = []
    for key in keywords:
        for orig, low in zip(cols, cols_lower):
            if key in low:
                candidates.append(orig)

    if not candidates:
        return default

    if prefer_contains:
        for c in candidates:
            if prefer_contains in c.lower():
                return c

    return candidates[0]

def normalize_amount(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("€", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan

def normalize_percent(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(" ", "").replace("%", "").replace(",", ".")
    if not s:
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if v > 1.0:
        v = v / 100.0
    return v

def normalize_date(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (datetime, pd.Timestamp)):
        return pd.to_datetime(x).date()
    s = str(x).strip()
    if not s:
        return pd.NaT
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d.%m.%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return pd.NaT

def build_text_field(df, candidate_keywords):
    cols = [c for c in df.columns if any(k in c.lower() for k in candidate_keywords)]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    return df[cols].astype(str).agg(" ".join, axis=1)

def extract_supplier_text(row, supplier_cols, invoice_cols):
    parts = []
    for c in supplier_cols:
        val = str(row.get(c, "")).strip()
        if val and val != "nan":
            parts.append(val)
    for c in invoice_cols:
        val = str(row.get(c, "")).strip()
        if val and val != "nan":
            parts.append(val)
    return " ".join(parts)

def clean_tokens(text):
    text = re.sub(r"[^a-z0-9äöüß ]", " ", str(text).lower())
    return [t for t in text.split() if len(t) >= 4]

def score_match(konto_text, beleg_supplier_text, invoice_number):
    konto_text_norm = re.sub(r"[^a-zA-Z0-9]", "", str(konto_text).lower())
    score = 0

    for t in clean_tokens(beleg_supplier_text):
        if t in str(konto_text).lower():
            score += 1

    if invoice_number:
        inv_clean = re.sub(r"[^a-zA-Z0-9]", "", str(invoice_number).strip().lower())
        partials = set()
        if len(inv_clean) >= 4:
            partials.add(inv_clean[-4:])
        if len(inv_clean) >= 6:
            partials.add(inv_clean[-6:])
        if len(inv_clean) >= 8:
            partials.add(inv_clean[-8:])
        partials.add(inv_clean)

        for p in partials:
            if p and p in konto_text_norm:
                score += 10

        if inv_clean and inv_clean in konto_text_norm:
            score += 20

    return score

def looks_like_cash_booking(konto_text, amount):
    text = str(konto_text).lower()
    kasse_keywords = [
        "edeka", "rewe", "netto", "aldi", "lidl", "penny",
        "kaufland", "denn", "dm ", "rossmann", "apotheke",
        "tankstelle", "shell", "aral", "esso", "omv", "bft",
        "pos ", "kartenzahlung", "ec-zahlung", "maestro", "visa",
        "mastercard"
    ]
    if any(kw in text for kw in kasse_keywords) and (amount is not None):
        try:
            amt = float(amount)
        except Exception:
            return False
        return abs(amt) <= 300
    return False

# ============================================================
# KPI / USt & Betriebsergebnis (unverändert)
# ============================================================

def classify_direction_by_partnerkonto(gp_konto: str) -> str:
    s = str(gp_konto or "").strip()
    if re.match(r"^1\d{3,}$", s):
        return "ausgang"
    if re.match(r"^7\d{3,}$", s):
        return "eingang"
    return "unknown"

def build_beleg_fulltext(belege_df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in belege_df.columns:
        cl = c.lower()
        if any(k in cl for k in [
            "geschäftspartner-name", "geschaeftspartner-name",
            "rechnungs-nr", "rechnungsnummer", "interne re",
            "ware/leistung", "konto-bezeichnung",
            "buchungstext", "verwendungszweck",
            "notiz", "bemerk", "text"
        ]):
            cols.append(c)

    if cols:
        belege_df["beleg_fulltext"] = belege_df[cols].astype(str).agg(" ".join, axis=1).str.lower()
    else:
        belege_df["beleg_fulltext"] = ""
    return belege_df

def is_usd(row) -> bool:
    wkz = str(row.get("WKZ", "")).strip().upper()
    return wkz == "USD"

def normalize_country_code(x: str) -> str:
    s = str(x or "").strip().upper()
    if not s or s == "NAN":
        return ""
    s = s.replace("DEUTSCHLAND", "DE").replace("GERMANY", "DE")
    s = s.replace("ÖSTERREICH", "AT").replace("OESTERREICH", "AT").replace("AUSTRIA", "AT")
    s = s.replace("SCHWEIZ", "CH").replace("SWITZERLAND", "CH")
    m = re.search(r"\b([A-Z]{2})\b", s)
    return m.group(1) if m else ""

def vatid_country(vatid: str) -> str:
    v = str(vatid or "").strip().upper().replace(" ", "")
    if len(v) >= 2 and re.match(r"^[A-Z]{2}", v):
        return v[:2]
    return ""

def infer_rate(row) -> tuple[float | None, str]:
    r = normalize_percent(row.get("Steuer in %", np.nan))
    if r == r:
        return r, "steuer_in_%"

    bu = str(row.get("BU", "")).strip()
    if bu in BU_RATE_MAP:
        return BU_RATE_MAP[bu], f"bu_{bu}"

    richtung = str(row.get("richtung", "unknown")).strip().lower()
    konto = str(row.get("Konto", "")).strip()
    if richtung == "ausgang" and konto in OUTGOING_KONTO_RATE:
        return OUTGOING_KONTO_RATE[konto], f"konto_{konto}"

    if bu and bu not in BU_RATE_MAP:
        return 0.0, f"bu_unbekannt_{bu}"

    if DEFAULT_RATE_FALLBACK is not None:
        return DEFAULT_RATE_FALLBACK, f"fallback_{int(DEFAULT_RATE_FALLBACK * 100)}"

    return None, "unbekannt"

def is_reverse_charge_or_foreign(row) -> bool:
    land = normalize_country_code(row.get("Land", ""))
    vatid = str(row.get("USt-IdNr.", "")).strip().upper().replace(" ", "")
    vatid_cc = vatid_country(vatid)

    rate = row.get("steuer_rate", np.nan)
    try:
        rate_num = float(rate) if rate == rate else np.nan
    except Exception:
        rate_num = np.nan
    rate_empty_or_zero = pd.isna(rate_num) or abs(rate_num) < 1e-9

    txt = str(row.get("beleg_fulltext", "")).lower()
    has_rc_keyword = any(k in txt for k in RC_KEYWORDS)

    if land and land != "DE":
        return True

    if vatid_cc and vatid_cc != "DE" and rate_empty_or_zero:
        return True

    if has_rc_keyword and rate_empty_or_zero:
        return True

    return False

def compute_vat_net_kpi(belege: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    brutto_col = find_column(belege, ["rechnungsbetrag", "bruttobetrag", "brutto", "betrag"], default=None)
    belege["brutto_calc"] = belege[brutto_col].apply(normalize_amount) if brutto_col else np.nan

    gp_col = find_column(belege, ["geschäftspartner-konto", "geschaeftspartner-konto"], default="Geschäftspartner-Konto")
    belege["richtung"] = belege[gp_col].apply(classify_direction_by_partnerkonto) if gp_col in belege.columns else "unknown"

    belege = build_beleg_fulltext(belege)

    belege["kpi_ignore_usd"] = belege.apply(is_usd, axis=1)

    rates = belege.apply(infer_rate, axis=1, result_type="expand")
    belege["steuer_rate"] = rates[0]
    belege["vat_methode"] = rates[1]

    belege["kpi_ignore_rc"] = belege.apply(is_reverse_charge_or_foreign, axis=1)
    belege["kpi_ignore"] = belege["kpi_ignore_usd"] | belege["kpi_ignore_rc"]

    belege["ust_calc"] = np.nan
    belege["netto_calc"] = np.nan

    mask_rate = belege["brutto_calc"].notna() & belege["steuer_rate"].notna() & (~belege["kpi_ignore"])
    belege.loc[mask_rate, "ust_calc"] = (
        belege.loc[mask_rate, "brutto_calc"]
        - (belege.loc[mask_rate, "brutto_calc"] / (1.0 + belege.loc[mask_rate, "steuer_rate"]))
    ).round(2)
    belege.loc[mask_rate, "netto_calc"] = (
        belege.loc[mask_rate, "brutto_calc"] - belege.loc[mask_rate, "ust_calc"]
    ).round(2)

    mask_zero = belege["brutto_calc"].notna() & (~belege["kpi_ignore"]) & (belege["steuer_rate"] == 0.0)
    belege.loc[mask_zero, "ust_calc"] = 0.0
    belege.loc[mask_zero, "netto_calc"] = belege.loc[mask_zero, "brutto_calc"].round(2)

    kpi = belege[~belege["kpi_ignore"]].copy()

    out_ok = (kpi["richtung"] == "ausgang") & kpi["ust_calc"].notna() & kpi["netto_calc"].notna()
    in_ok = (kpi["richtung"] == "eingang") & kpi["ust_calc"].notna() & kpi["netto_calc"].notna()

    ust_ausgang = float(kpi.loc[out_ok, "ust_calc"].sum())
    vorsteuer = float(kpi.loc[in_ok, "ust_calc"].sum())
    ust_saldo = round(ust_ausgang - vorsteuer, 2)

    umsatz_netto = float(kpi.loc[out_ok, "netto_calc"].sum())
    kosten_netto = float(kpi.loc[in_ok, "netto_calc"].sum())
    betriebsergebnis = round(umsatz_netto - kosten_netto, 2)

    out_total = int((kpi["richtung"] == "ausgang").sum())
    in_total = int((kpi["richtung"] == "eingang").sum())
    out_cov = int(out_ok.sum())
    in_cov = int(in_ok.sum())

    ignored_total = int(belege["kpi_ignore"].sum())
    ignored_usd = int(belege["kpi_ignore_usd"].sum())
    ignored_rc = int(belege["kpi_ignore_rc"].sum())
    brutto_ignored = float(belege.loc[belege["kpi_ignore"], "brutto_calc"].fillna(0).sum())

    methode_stats = (
        kpi["vat_methode"].value_counts(dropna=False).rename_axis("vat_methode").reset_index(name="count")
    )

    res = {
        "ust_ausgang_sum": round(ust_ausgang, 2),
        "vorsteuer_sum": round(vorsteuer, 2),
        "ust_saldo": ust_saldo,
        "ust_interpretation": "USt-Nachzahlung" if ust_saldo > 0 else ("USt-Erstattung" if ust_saldo < 0 else "USt-Ausgeglichen"),
        "umsatz_netto": round(umsatz_netto, 2),
        "kosten_netto": round(kosten_netto, 2),
        "betriebsergebnis": round(betriebsergebnis, 2),
        "ausgang_total_kpi": out_total,
        "ausgang_covered_kpi": out_cov,
        "eingang_total_kpi": in_total,
        "eingang_covered_kpi": in_cov,
        "ignored_total": ignored_total,
        "ignored_usd": ignored_usd,
        "ignored_rc": ignored_rc,
        "brutto_ignored": round(brutto_ignored, 2),
    }

    pd.DataFrame([res]).to_csv(OUTPUT_DIR / "ust_betriebsergebnis.csv", sep=";", index=False, encoding="utf-8-sig")
    methode_stats.to_csv(OUTPUT_DIR / "ust_methoden_stats.csv", sep=";", index=False, encoding="utf-8-sig")

    debug_cols = [c for c in [
        "Geschäftspartner-Name", "Geschäftspartner-Konto", "Rechnungsbetrag", "WKZ",
        "BU", "Konto", "Steuer in %", "USt-IdNr.", "Land", "richtung",
        "steuer_rate", "vat_methode", "brutto_calc", "netto_calc", "ust_calc", "Rechnungs-Nr."
    ] if c in belege.columns]

    belege.loc[belege["kpi_ignore"], debug_cols].to_csv(
        OUTPUT_DIR / "kpi_ignoriert.csv", sep=";", index=False, encoding="utf-8-sig"
    )

    kpi.loc[(kpi["netto_calc"].isna()) | (kpi["ust_calc"].isna()), debug_cols].to_csv(
        OUTPUT_DIR / "kpi_offen.csv", sep=";", index=False, encoding="utf-8-sig"
    )

    return belege, res

# ============================================================
# HAUPTLOGIK (Matching + KPI)
# ============================================================

def run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    konto = safe_read_csv(KONTOAUSZUG_CSV, sep=";")

    amount_col = find_column(
        konto,
        ["umsatz (ohne soll/haben-kz)", "betrag", "umsatz", "betrag in eur"],
        default=None
    )
    if not amount_col:
        raise ValueError("Konnte im Kontoauszug keine Betrags-Spalte finden.")
    konto["betrag_raw"] = konto[amount_col].apply(normalize_amount)

    date_col = find_column(konto, ["buchungstag", "buchungsdatum", "datum"], default=None)
    if not date_col:
        raise ValueError("Konnte im Kontoauszug keine Datums-Spalte finden.")
    konto["datum_norm"] = pd.to_datetime(konto[date_col].apply(normalize_date), errors="coerce")

    konto["text_gesamt"] = build_text_field(
        konto,
        ["buchungstext", "verwendungszweck", "name", "empfänger", "begünstigter", "auftraggeber"]
    )
    konto["konto_index"] = konto.index

    belege = safe_read_csv(BELEGE_CSV, sep=";")

    beleg_amount_col = find_column(
        belege,
        ["rechnungsbetrag", "bruttobetrag", "bruttowert", "brutto", "betrag"],
        default=None
    )
    if not beleg_amount_col:
        raise ValueError("Konnte in der Belegliste keine Brutto-Betrags-Spalte finden.")
    belege["betrag_raw"] = belege[beleg_amount_col].apply(normalize_amount)

    beleg_date_col = find_column(belege, ["belegdatum", "rechnungsdatum", "datum"], default=None)
    if beleg_date_col:
        belege["datum_norm"] = pd.to_datetime(belege[beleg_date_col].apply(normalize_date), errors="coerce")
    else:
        belege["datum_norm"] = pd.NaT

    supplier_cols = [
        c for c in belege.columns
        if any(k in c.lower() for k in ["geschäftspartner-name", "geschaeftspartner-name", "lieferant", "name", "kunde"])
    ]
    invoice_cols = [
        c for c in belege.columns
        if any(k in c.lower() for k in ["rechnungs-nr", "rechnungsnummer", "interne re"])
    ]
    belege["supplier_text"] = belege.apply(lambda row: extract_supplier_text(row, supplier_cols, invoice_cols), axis=1)

    invoice_main_col = find_column(belege, ["rechnungs-nr.", "rechnungsnummer"], default=None, prefer_contains="rechnungs")
    belege["invoice_number"] = belege[invoice_main_col] if invoice_main_col else ""

    status_col = find_column(belege, ["gebucht", "status", "belegstatus", "buchungsstatus"], default=None)
    if (not status_col) and STATUS_SPALTE_MANUELL and STATUS_SPALTE_MANUELL in belege.columns:
        status_col = STATUS_SPALTE_MANUELL

    if status_col:
        status_lower = belege[status_col].astype(str).str.lower().str.strip()
        belege["ist_gebucht"] = status_lower.eq("ja")
        belege["ist_posteingang"] = ~belege["ist_gebucht"]
    else:
        belege["ist_gebucht"] = True
        belege["ist_posteingang"] = False

    belege, kpi_res = compute_vat_net_kpi(belege)

    sichere_matches = []
    unklare_map = {}
    verwendete_konto_indices = set()

    gebuchte = belege[belege["ist_gebucht"] == True].copy()
    for _, beleg in gebuchte.iterrows():
        betrag = beleg.get("betrag_raw", np.nan)
        datum = beleg.get("datum_norm", pd.NaT)
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(float(betrag))
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)].copy()
        if candidates.empty:
            continue

        diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
        candidates["datum_diff_tage"] = diff_days
        candidates = candidates[diff_days <= 45]
        if candidates.empty:
            continue

        sup_txt = beleg.get("supplier_text", "")
        inv_nr = beleg.get("invoice_number", "")

        candidates["score"] = candidates["text_gesamt"].apply(lambda t: score_match(t, sup_txt, inv_nr))
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True])

        best = candidates.iloc[0]

        if best["score"] >= 6 or (len(candidates) == 1 and best["score"] >= 1):
            if best["konto_index"] not in verwendete_konto_indices:
                verwendete_konto_indices.add(best["konto_index"])
                sichere_matches.append({
                    "typ": "gebucht",
                    "score": best["score"],
                    "konto_index": best["konto_index"],
                    "konto_datum": best["datum_norm"],
                    "konto_betrag": best["betrag_raw"],
                    "konto_text": best["text_gesamt"],
                    "beleg_index": beleg.name,
                    "beleg_datum": beleg.get("datum_norm", pd.NaT),
                    "beleg_betrag": beleg.get("betrag_raw", np.nan),
                    "beleg_supplier": sup_txt,
                    "beleg_rechnungsnr": inv_nr,
                })
        else:
            entry = unklare_map.setdefault(beleg.name, {
                "typ": "gebucht",
                "beleg_index": beleg.name,
                "beleg_datum": beleg.get("datum_norm", pd.NaT),
                "beleg_betrag": beleg.get("betrag_raw", np.nan),
                "beleg_supplier": sup_txt,
                "beleg_rechnungsnr": inv_nr,
                "kandidaten": [],
            })
            for _, c in candidates.iterrows():
                entry["kandidaten"].append({
                    "konto_index": c["konto_index"],
                    "konto_datum": c["datum_norm"],
                    "konto_betrag": c["betrag_raw"],
                    "konto_text": c["text_gesamt"],
                    "score": c["score"],
                })

    posteingang = belege[belege["ist_posteingang"] == True].copy()
    for _, beleg in posteingang.iterrows():
        betrag = beleg.get("betrag_raw", np.nan)
        datum = beleg.get("datum_norm", pd.NaT)
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(float(betrag))
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)].copy()
        if candidates.empty:
            continue

        diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
        candidates["datum_diff_tage"] = diff_days
        candidates = candidates[diff_days <= DATUM_FENSTER_TAGE]
        if candidates.empty:
            continue

        sup_txt = beleg.get("supplier_text", "")
        inv_nr = beleg.get("invoice_number", "")

        candidates["score"] = candidates["text_gesamt"].apply(lambda t: score_match(t, sup_txt, inv_nr))
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True])

        best = candidates.iloc[0]
        second_score = candidates.iloc[1]["score"] if len(candidates) > 1 else None

        if len(candidates) > 1 and (best["score"] <= 0 or (second_score is not None and (best["score"] - second_score) < 2)):
            entry = unklare_map.setdefault(beleg.name, {
                "typ": "posteingang",
                "beleg_index": beleg.name,
                "beleg_datum": beleg.get("datum_norm", pd.NaT),
                "beleg_betrag": beleg.get("betrag_raw", np.nan),
                "beleg_supplier": sup_txt,
                "beleg_rechnungsnr": inv_nr,
                "kandidaten": [],
            })
            for _, c in candidates.iterrows():
                entry["kandidaten"].append({
                    "konto_index": c["konto_index"],
                    "konto_datum": c["datum_norm"],
                    "konto_betrag": c["betrag_raw"],
                    "konto_text": c["text_gesamt"],
                    "score": c["score"],
                })
            continue

        if best["konto_index"] not in verwendete_konto_indices:
            verwendete_konto_indices.add(best["konto_index"])
            sichere_matches.append({
                "typ": "posteingang",
                "score": best["score"],
                "konto_index": best["konto_index"],
                "konto_datum": best["datum_norm"],
                "konto_betrag": best["betrag_raw"],
                "konto_text": best["text_gesamt"],
                "beleg_index": beleg.name,
                "beleg_datum": beleg.get("datum_norm", pd.NaT),
                "beleg_betrag": beleg.get("betrag_raw", np.nan),
                "beleg_supplier": sup_txt,
                "beleg_rechnungsnr": inv_nr,
            })

    unklare_faelle = []
    for _, data in unklare_map.items():
        kandidaten = data["kandidaten"]
        if not kandidaten:
            continue
        best = max(kandidaten, key=lambda c: c["score"])
        konto_indices = sorted({k["konto_index"] for k in kandidaten})
        konto_scores_str = "; ".join(
            f"{k['konto_index']}:{k['score']}"
            for k in sorted(kandidaten, key=lambda c: (-c["score"], str(c["konto_index"])))
        )
        unklare_faelle.append({
            "typ": data["typ"],
            "beleg_index": data["beleg_index"],
            "beleg_datum": data["beleg_datum"],
            "beleg_betrag": data["beleg_betrag"],
            "beleg_supplier": data["beleg_supplier"],
            "beleg_rechnungsnr": data["beleg_rechnungsnr"],
            "anzahl_konto_kandidaten": len(konto_indices),
            "konto_indices": ",".join(str(i) for i in konto_indices),
            "best_konto_index": best["konto_index"],
            "best_konto_datum": best["konto_datum"],
            "best_konto_betrag": best["konto_betrag"],
            "best_konto_text": best["konto_text"],
            "best_score": best["score"],
            "konto_indices_scores": konto_scores_str,
        })

    alle_verwendeten_konto = {m["konto_index"] for m in sichere_matches}
    for data in unklare_map.values():
        for k in data["kandidaten"]:
            alle_verwendeten_konto.add(k["konto_index"])

    konto_ohne_beleg = konto[~konto["konto_index"].isin(alle_verwendeten_konto)].copy()
    konto_ohne_beleg["ist_kasse_vermutet"] = konto_ohne_beleg.apply(
        lambda row: looks_like_cash_booking(row["text_gesamt"], row["betrag_raw"]),
        axis=1
    )

    vorschlaege = []
    for _, beleg in posteingang.iterrows():
        betrag = beleg.get("betrag_raw", np.nan)
        datum = beleg.get("datum_norm", pd.NaT)
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(float(betrag))
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= (BETRAG_TOLERANZ * 2))].copy()
        if candidates.empty:
            continue

        diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
        candidates["datum_diff_tage"] = diff_days
        candidates = candidates[diff_days <= 60]
        if candidates.empty:
            continue

        sup_txt = beleg.get("supplier_text", "")
        inv_nr = beleg.get("invoice_number", "")

        candidates["score"] = candidates["text_gesamt"].apply(lambda t: score_match(t, sup_txt, inv_nr))
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True]).head(3)

        for _, c in candidates.iterrows():
            vorschlaege.append({
                "beleg_index": beleg.name,
                "beleg_datum": beleg.get("datum_norm", pd.NaT),
                "beleg_betrag": beleg.get("betrag_raw", np.nan),
                "beleg_supplier": sup_txt,
                "beleg_rechnungsnr": inv_nr,
                "konto_index": c["konto_index"],
                "konto_datum": c["datum_norm"],
                "konto_betrag": c["betrag_raw"],
                "konto_text": c["text_gesamt"],
                "datum_diff_tage": c["datum_diff_tage"],
                "score": c["score"],
            })

    df_sicher = pd.DataFrame(sichere_matches)
    df_unklar = pd.DataFrame(unklare_faelle)

    if not df_sicher.empty:
        df_sicher.to_csv(OUTPUT_DIR / "matches_sicher.csv", sep=";", index=False, encoding="utf-8-sig")
    else:
        (OUTPUT_DIR / "matches_sicher.csv").write_text("keine sicheren Matches gefunden", encoding="utf-8")

    if not df_unklar.empty:
        df_unklar.to_csv(OUTPUT_DIR / "matches_unklar.csv", sep=";", index=False, encoding="utf-8-sig")
    else:
        (OUTPUT_DIR / "matches_unklar.csv").write_text("keine unklaren Fälle gefunden", encoding="utf-8")

    konto_ohne_beleg.to_csv(OUTPUT_DIR / "konto_ohne_beleg.csv", sep=";", index=False, encoding="utf-8-sig")

    if vorschlaege:
        pd.DataFrame(vorschlaege).to_csv(OUTPUT_DIR / "posteingang_kandidaten.csv", sep=";", index=False, encoding="utf-8-sig")

    anzahl_sicher = len(df_sicher)
    anzahl_sicher_gebucht = len(df_sicher[df_sicher["typ"] == "gebucht"]) if not df_sicher.empty else 0
    anzahl_sicher_post = len(df_sicher[df_sicher["typ"] == "posteingang"]) if not df_sicher.empty else 0
    anzahl_unklar = len(df_unklar)
    anzahl_fehlende = len(konto_ohne_beleg)
    anzahl_kasse = int(konto_ohne_beleg["ist_kasse_vermutet"].sum()) if not konto_ohne_beleg.empty else 0

    ok, msg = weclapp_check_company()
    pd.DataFrame([{
        "configured_env": weclapp_configured(),
        "api_ok": ok,
        "status": msg,
        "base_url": weclapp_base_url(),
        "token_set": bool(weclapp_token())
    }]).to_csv(OUTPUT_DIR / "weclapp_status.csv", sep=";", index=False, encoding="utf-8-sig")

    return {
        "anzahl_sicher": anzahl_sicher,
        "anzahl_sicher_gebucht": anzahl_sicher_gebucht,
        "anzahl_sicher_post": anzahl_sicher_post,
        "anzahl_unklar": anzahl_unklar,
        "anzahl_fehlende": anzahl_fehlende,
        "anzahl_kasse": anzahl_kasse,
        "weclapp_api_ok": ok,
        "weclapp_status_msg": msg,
        **kpi_res,
    }

# ============================================================
# WEB UI
# ============================================================

def format_eur(x) -> str:
    try:
        return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00 €"

@app.get("/", response_class=HTMLResponse)
def index():
    # Debug in Logs (ohne Token zu leaken)
    log_env_keys()

    ok, msg = weclapp_check_company()
    status_badge = "✅ " + msg if ok else "⚠️ " + msg

    today = datetime.now().date()
    default_from = f"{today.year}-01-01"
    default_to = f"{today.year}-03-31"

    return f"""
    <html>
      <head>
        <title>NEXTWAVE Business Finance AI</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; }}
          .logo {{ max-width: 260px; height: auto; margin-bottom: 10px; display:block; }}
          .hint {{ color:#555; margin-bottom: 1.0rem; }}
          .box {{ border:1px solid #e5e7eb; border-radius:12px; padding:14px; background:#fafafa; margin-top: 12px; }}
          .row {{ display:flex; gap:12px; flex-wrap: wrap; align-items: end; }}
          label {{ font-weight:600; display:block; margin-bottom:6px; }}
          input[type="date"] {{ padding:10px; border:1px solid #d1d5db; border-radius:8px; min-width: 180px; }}
          .field {{ margin-top: 12px; }}
          .dropzone {{ border:2px dashed #999; border-radius:8px; padding:14px; text-align:center; cursor:pointer; transition:.2s; }}
          .dropzone.hover {{ border-color:{NEXTWAVE_BLUE}; background:#eff6ff; }}
          .filename {{ font-weight:600; margin-top:6px; }}
          .file-input {{ display:none; }}

          button {{ padding:10px 18px; font-size:16px; border-radius:8px; border:none; color:white; cursor:pointer; }}
          button:disabled {{ opacity:0.7; cursor:not-allowed; }}

          .btn-dark {{ background:{NEXTWAVE_DARK}; }}
          .btn-dark:hover {{ background:#0b1220; }}

          .btn-orange {{ background:{NEXTWAVE_ORANGE} !important; }}
          .btn-orange:hover {{ filter: brightness(0.92); }}

          .progress {{ margin-top:1rem; font-size:0.9rem; color:{NEXTWAVE_BLUE}; display:none; align-items:center; gap:8px; }}
          .progress.active {{ display:inline-flex; }}
          .spinner {{ width:18px; height:18px; border-radius:999px; border:3px solid #e5e7eb; border-top-color:{NEXTWAVE_BLUE}; animation: spin 0.8s linear infinite; }}
          @keyframes spin {{ from {{ transform: rotate(0deg);}} to {{ transform: rotate(360deg);}} }}

          .legal {{ margin-top:1.2rem; font-size:0.75rem; color:#777; line-height:1.4; }}
          .status {{ font-size: 0.95rem; color:#374151; }}
        </style>
      </head>
      <body>
        <img src="/logo.png" alt="NEXTWAVE Logo" class="logo" />
        <h1>Business Finance AI</h1>
        <p class="hint">
Willkommen bei deiner NEXTWAVE Business Finance AI!<br><br>
1) Zeitraum auswählen – USt-Check über Weclapp.<br>
2) Kontoauszug + DATEV-Belege hochladen – Abgleich inkl. Excel-Export.<br>
        </p>

        <div class="box">
          <h2>Zeitraum (Quartal)</h2>
          <div class="row">
            <div>
              <label>Von</label>
              <input type="date" id="date_from" name="date_from" value="{default_from}">
            </div>
            <div>
              <label>Bis</label>
              <input type="date" id="date_to" name="date_to" value="{default_to}">
            </div>
          </div>

          <div class="field status" style="margin-top:10px;">
            <strong>Weclapp:</strong> {status_badge}
          </div>

          <form id="weclappForm" action="/weclapp-ust" method="post" style="margin-top:12px;">
            <input type="hidden" name="date_from" id="wf_from" value="{default_from}">
            <input type="hidden" name="date_to" id="wf_to" value="{default_to}">
            <button type="submit" class="btn-dark">Weclapp USt-Status berechnen</button>
          </form>
        </div>

        <div class="box">
          <h2>DATEV Analyse (Kontoauszug + Belege)</h2>

          <form id="uploadForm" action="/run" method="post" enctype="multipart/form-data">
            <div class="field">
              <span style="font-weight:600;">Kontoauszug CSV</span>
              <div id="konto_drop" class="dropzone">
                <div>CSV hierhin ziehen oder klicken</div>
                <div class="filename" id="konto_filename">Keine Datei ausgewählt</div>
              </div>
              <input class="file-input" type="file" name="konto_file" id="konto_file" accept=".csv" required />
            </div>

            <div class="field">
              <span style="font-weight:600;">Belege CSV</span>
              <div id="belege_drop" class="dropzone">
                <div>CSV hierhin ziehen oder klicken</div>
                <div class="filename" id="belege_filename">Keine Datei ausgewählt</div>
              </div>
              <input class="file-input" type="file" name="belege_file" id="belege_file" accept=".csv" required />
            </div>

            <button type="submit" id="submitBtn" class="btn-orange">DATEV Analyse starten</button>
            <div id="progress" class="progress"><div class="spinner"></div><span>Analyse läuft …</span></div>
          </form>
        </div>

        <div class="legal">
          © NEXTWAVE GmbH – Alle Rechte vorbehalten.<br>
          Die Nutzung dieses Programms oder von Teilen daraus ohne vorherige schriftliche Zustimmung der NEXTWAVE GmbH
          ist untersagt und kann zivil- und strafrechtliche Schritte nach sich ziehen.
        </div>

        <script>
          function setupDropzone(dropId, inputId, labelId) {{
            const drop = document.getElementById(dropId);
            const input = document.getElementById(inputId);
            const label = document.getElementById(labelId);

            drop.addEventListener('click', function(){{ input.click(); }});

            input.addEventListener('change', function(){{
              if (input.files && input.files.length > 0) label.textContent = input.files[0].name;
              else label.textContent = "Keine Datei ausgewählt";
            }});

            ['dragenter','dragover'].forEach(eventName => {{
              drop.addEventListener(eventName, function(e){{
                e.preventDefault(); e.stopPropagation();
                drop.classList.add('hover');
              }}, false);
            }});

            ['dragleave','drop'].forEach(eventName => {{
              drop.addEventListener(eventName, function(e){{
                e.preventDefault(); e.stopPropagation();
                drop.classList.remove('hover');
              }}, false);
            }});

            drop.addEventListener('drop', function(e){{
              const files = e.dataTransfer.files;
              if (files && files.length > 0) {{
                input.files = files;
                label.textContent = files[0].name;
              }}
            }});
          }}

          function syncDatesToForms() {{
            const df = document.getElementById('date_from').value;
            const dt = document.getElementById('date_to').value;
            document.getElementById('wf_from').value = df;
            document.getElementById('wf_to').value = dt;
          }}

          document.addEventListener('DOMContentLoaded', function(){{
            setupDropzone('konto_drop','konto_file','konto_filename');
            setupDropzone('belege_drop','belege_file','belege_filename');

            const form = document.getElementById('uploadForm');
            const submitBtn = document.getElementById('submitBtn');
            const progress = document.getElementById('progress');
            const kontoInput = document.getElementById('konto_file');
            const belegeInput = document.getElementById('belege_file');

            document.getElementById('date_from').addEventListener('change', syncDatesToForms);
            document.getElementById('date_to').addEventListener('change', syncDatesToForms);
            syncDatesToForms();

            form.addEventListener('submit', function(e){{
              if (!kontoInput.files.length || !belegeInput.files.length) {{
                e.preventDefault();
                alert('Bitte sowohl Kontoauszug-CSV als auch Belege-CSV auswählen.');
                return;
              }}
              submitBtn.disabled = true;
              submitBtn.textContent = 'Analyse läuft ...';
              progress.classList.add('active');
            }});
          }});
        </script>
      </body>
    </html>
    """

@app.post("/weclapp-ust", response_class=HTMLResponse)
async def weclapp_ust(date_from: str = Form(""), date_to: str = Form("")):
    ok, msg = weclapp_check_company()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "date_from": date_from,
        "date_to": date_to,
        "configured_env": weclapp_configured(),
        "api_ok": ok,
        "status": msg,
        "base_url": weclapp_base_url(),
        "token_set": bool(weclapp_token())
    }]).to_csv(OUTPUT_DIR / "weclapp_status.csv", sep=";", index=False, encoding="utf-8-sig")

    color = "#065f46" if ok else "#991b1b"
    headline = "Weclapp OK" if ok else "Weclapp Problem"

    return f"""
    <html>
      <head>
        <title>{headline}</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 16px; }}
          .box {{ border:1px solid #e5e7eb; border-radius:12px; padding:14px; background:#fafafa; }}
          .badge {{ display:inline-block; padding:6px 10px; border-radius:999px; background:{color}; color:white; font-weight:600; }}
          a.button {{ display:inline-block; margin-top: 1.2rem; padding:10px 18px; background:{NEXTWAVE_BLUE}; color:#fff; text-decoration:none; border-radius:8px; }}
          a.button:hover {{ background:#1d4ed8; }}
          code {{ background:#f3f4f6; padding:2px 6px; border-radius:6px; }}
        </style>
      </head>
      <body>
        <h1>{headline}</h1>
        <div class="box">
          <div class="badge">{msg}</div>
          <p style="margin-top:12px; color:#374151;">
            Zeitraum: <strong>{date_from or "–"}</strong> bis <strong>{date_to or "–"}</strong><br>
            API Base: <code>{weclapp_base_url() or "-"}</code><br>
          </p>
          <a class="button" href="/">Zurück</a>
        </div>
      </body>
    </html>
    """

@app.post("/run", response_class=HTMLResponse)
async def run(konto_file: UploadFile = File(...), belege_file: UploadFile = File(...)):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(KONTOAUSZUG_CSV, "wb") as f:
        f.write(await konto_file.read())
    with open(BELEGE_CSV, "wb") as f:
        f.write(await belege_file.read())

    res = run_analysis()

    zip_path = OUTPUT_DIR / "datev_analyse_ergebnisse.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_file in OUTPUT_DIR.glob("*.csv"):
            zf.write(csv_file, arcname=csv_file.name)

    return f"""
    <html>
      <head>
        <title>Analyse abgeschlossen</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 16px; }}
          ul {{ line-height: 1.7; }}
          a.button {{ display:inline-block; margin-top: 1.2rem; padding:10px 18px; background:{NEXTWAVE_BLUE}; color:#fff; text-decoration:none; border-radius:8px; }}
          a.button:hover {{ background:#1d4ed8; }}
          .box {{ border:1px solid #e5e7eb; border-radius:12px; padding:14px; background:#fafafa; margin-top: 12px; }}
          .legal {{ margin-top:2rem; font-size:0.75rem; color:#777; line-height:1.4; }}
          code {{ background:#f3f4f6; padding:2px 6px; border-radius:6px; }}
        </style>
      </head>
      <body>
        <h1>Analyse abgeschlossen</h1>

        <div class="box">
          <h2>Matching</h2>
          <ul>
            <li><strong>Sichere Matches gesamt:</strong> {res['anzahl_sicher']}</li>
            <li>&nbsp;&nbsp;&bull; davon gebuchte Belege: {res['anzahl_sicher_gebucht']}</li>
            <li>&nbsp;&nbsp;&bull; davon Posteingang-Rechnungen: {res['anzahl_sicher_post']}</li>
            <li><strong>Unklare Fälle:</strong> {res['anzahl_unklar']}</li>
            <li><strong>Fehlende Belege gesamt:</strong> {res['anzahl_fehlende']}</li>
            <li>&nbsp;&nbsp;&bull; davon Kassenbuchungen vermutet: {res['anzahl_kasse']}</li>
          </ul>
        </div>

        <div class="box">
          <h2>USt &amp; Betriebsergebnis (Quartal)</h2>
          <p style="color:#374151; margin-top:0;">
            Details in der ZIP: <code>ust_betriebsergebnis.csv</code>, <code>ust_methoden_stats.csv</code>, <code>kpi_offen.csv</code>, <code>kpi_ignoriert.csv</code>.
          </p>
          <ul>
            <li><strong>USt (Ausgang):</strong> {format_eur(res.get('ust_ausgang_sum',0))}</li>
            <li><strong>Vorsteuer (Eingang):</strong> {format_eur(res.get('vorsteuer_sum',0))}</li>
            <li><strong>USt-Saldo:</strong> {format_eur(res.get('ust_saldo',0))} <small>({res.get('ust_interpretation','')})</small></li>
            <li><strong>Umsatz netto:</strong> {format_eur(res.get('umsatz_netto',0))}</li>
            <li><strong>Kosten netto:</strong> {format_eur(res.get('kosten_netto',0))}</li>
            <li><strong>Betriebsergebnis:</strong> {format_eur(res.get('betriebsergebnis',0))}</li>
          </ul>
        </div>

        <div class="box">
          <h2>Weclapp Status</h2>
          <p style="color:#374151; margin-top:0;">
            Details in der ZIP: <code>weclapp_status.csv</code>
          </p>
          <ul>
            <li><strong>Weclapp:</strong> {"✅" if res.get("weclapp_api_ok") else "⚠️"} {res.get("weclapp_status_msg","")}</li>
          </ul>
        </div>

        <a href="/download" class="button">Ergebnis-ZIP herunterladen</a>
        <div style="margin-top:1rem;"><a href="/">Neue Analyse starten</a></div>

        <div class="legal">
          © NEXTWAVE GmbH – Alle Rechte vorbehalten.<br>
          Die Nutzung dieses Programms oder von Teilen daraus ohne vorherige schriftliche Zustimmung der NEXTWAVE GmbH
          ist untersagt und kann zivil- und strafrechtliche Schritte nach sich ziehen.
        </div>
      </body>
    </html>
    """

@app.get("/download")
def download_zip():
    zip_path = OUTPUT_DIR / "datev_analyse_ergebnisse.zip"
    if not zip_path.exists():
        return HTMLResponse("<h1>Keine ZIP gefunden</h1><p>Bitte zuerst eine Analyse starten.</p>", status_code=404)
    return FileResponse(zip_path, media_type="application/zip", filename="datev_analyse_ergebnisse.zip")

@app.get("/logo.png")
def logo():
    return FileResponse(BASE_DIR / "nextwave_logo.png", media_type="image/png")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
