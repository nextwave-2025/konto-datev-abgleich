import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import re
import os
import zipfile
import requests

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ============================================================
# KONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

KONTOAUSZUG_CSV = BASE_DIR / "kontoauszug.csv"
BELEGE_CSV     = BASE_DIR / "belege.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BETRAG_TOLERANZ = 0.01
DATUM_FENSTER_TAGE = 30
STATUS_SPALTE_MANUELL = None

# Weclapp ENV
WECLAPP_BASE_URL = os.getenv("WECLAPP_BASE_URL", "").rstrip("/")  # z.B. https://nextwave.weclapp.com
WECLAPP_API_TOKEN = os.getenv("WECLAPP_API_TOKEN", "").strip()

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# ============================================================
# HELPER (CSV / Normalisierung / Matching)
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

def normalize_for_invoice_match(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^0-9a-z]", "", str(s).lower())

def make_invoice_variants(inv_clean: str) -> set[str]:
    if not inv_clean:
        return set()
    variants = {inv_clean}

    no_prefix = re.sub(r"^[a-z]+", "", inv_clean)
    if no_prefix and no_prefix != inv_clean:
        variants.add(no_prefix)

    for n in (4, 6, 8):
        if len(inv_clean) >= n:
            variants.add(inv_clean[-n:])
        if len(no_prefix) >= n:
            variants.add(no_prefix[-n:])

    variants = {v for v in variants if len(v) >= 6}
    return variants

def score_match(konto_text, beleg_supplier_text, invoice_number):
    konto_text_norm = re.sub(r"[^a-zA-Z0-9]", "", str(konto_text).lower())
    score = 0

    for t in clean_tokens(beleg_supplier_text):
        if t in str(konto_text).lower():
            score += 1

    if invoice_number:
        inv_clean = normalize_for_invoice_match(invoice_number)
        partials = set()
        if len(inv_clean) >= 4: partials.add(inv_clean[-4:])
        if len(inv_clean) >= 6: partials.add(inv_clean[-6:])
        if len(inv_clean) >= 8: partials.add(inv_clean[-8:])
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
        "mastercard", "amazon", "amzn"
    ]
    if any(kw in text for kw in kasse_keywords) and (amount is not None):
        try:
            amt = float(amount)
        except Exception:
            return False
        return abs(amt) <= 300
    return False

# ============================================================
# INVOICE-FIRST INDEX (Belege.csv -> Variantenindex)
# ============================================================

def build_invoice_index(belege: pd.DataFrame) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}

    inv_cols = [c for c in belege.columns if any(k in c.lower() for k in [
        "rechnungs-nr", "rechnungsnummer", "interne re", "belegfeld 1", "belegfeld1"
    ])]
    if not inv_cols:
        belege["invoice_number"] = ""
        return idx

    main = inv_cols[0]
    belege["invoice_number"] = belege[main].astype(str)

    for i, row in belege.iterrows():
        raw_values = []
        for c in inv_cols:
            v = str(row.get(c, "")).strip()
            if v and v.lower() != "nan":
                raw_values.append(v)

        for raw in raw_values:
            inv_clean = normalize_for_invoice_match(raw)
            for v in make_invoice_variants(inv_clean):
                idx.setdefault(v, []).append(i)

    return idx

def find_invoices_in_konto_text(konto_text: str, invoice_index: dict[str, list[int]]) -> set[int]:
    text_norm = normalize_for_invoice_match(konto_text)
    hits: set[int] = set()
    if not text_norm:
        return hits

    chunks = re.findall(r"[0-9a-z]{6,}", text_norm)

    for ch in chunks:
        if ch in invoice_index:
            hits.update(invoice_index[ch])

        ch2 = re.sub(r"^[a-z]+", "", ch)
        if ch2 and ch2 in invoice_index:
            hits.update(invoice_index[ch2])

        for n in (8, 6):
            if len(ch) >= n and ch[-n:] in invoice_index:
                hits.update(invoice_index[ch[-n:]])
            if ch2 and len(ch2) >= n and ch2[-n:] in invoice_index:
                hits.update(invoice_index[ch2[-n:]])

    return hits

# ============================================================
# WECLAPP: USt Status für Zeitraum
# ============================================================

def _parse_iso_date(s: str) -> date:
    s = (s or "").strip()
    return datetime.strptime(s, "%Y-%m-%d").date()

def weclapp_enabled() -> bool:
    return bool(WECLAPP_BASE_URL and WECLAPP_API_TOKEN)

def weclapp_headers() -> dict:
    # Weclapp nutzt API-Token im Header (üblich: Authentication oder X-...).
    # Da Tenant-Setups variieren, unterstützen wir mehrere Varianten.
    # (Wenn es bei dir nicht greift, sag mir kurz den Header-Namen aus deinem Setup.)
    return {
        "AuthenticationToken": WECLAPP_API_TOKEN,
        "Authorization": f"Bearer {WECLAPP_API_TOKEN}",
        "Accept": "application/json",
    }

def weclapp_get_all(entity: str, filters: dict, page_size: int = 500) -> list[dict]:
    """
    Lädt paginiert. Weclapp API v1-style: /webapp/api/v1/{entity}
    Filter Syntax: property-operator=value, z.B. invoiceDate-ge=2025-10-01 :contentReference[oaicite:1]{index=1}
    """
    base = f"{WECLAPP_BASE_URL}/webapp/api/v1/{entity}"
    params = {"pageSize": page_size, **filters}
    out = []
    page = 1
    while True:
        params["page"] = page
        r = requests.get(base, headers=weclapp_headers(), params=params, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Weclapp API Fehler {r.status_code} bei {entity}: {r.text[:500]}")
        data = r.json()
        # Weclapp liefert meist { "result": [...], "page":..., "pageSize":..., "totalPages":... }
        if isinstance(data, dict) and "result" in data:
            batch = data.get("result") or []
            out.extend(batch)
            total_pages = data.get("totalPages")
            if total_pages is not None and page >= int(total_pages):
                break
            if len(batch) < page_size and total_pages is None:
                break
        elif isinstance(data, list):
            out.extend(data)
            break
        else:
            break
        page += 1
        if page > 2000:
            break
    return out

def pick_first(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return default

def to_float_safe(x):
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(" ", "").replace(",", ".")
        return float(s)
    except Exception:
        return 0.0

def summarize_weclapp_vat(period_from: str, period_to: str) -> dict:
    """
    Summiert:
    - Vorsteuer: aus purchaseInvoice (fallback: purchaseOrder)
    - Umsatzsteuer: aus salesInvoice
    Und erstellt Details als Excel in output/.
    """
    if not weclapp_enabled():
        return {
            "enabled": False,
            "error": "Weclapp ist nicht konfiguriert (WECLAPP_BASE_URL / WECLAPP_API_TOKEN fehlen).",
            "vorsteuer_sum": 0.0,
            "umsatzsteuer_sum": 0.0,
            "saldo": 0.0,
            "status": "Unbekannt",
            "details_file": "",
        }

    d_from = _parse_iso_date(period_from)
    d_to = _parse_iso_date(period_to)

    # Filterfelder – je nach Entity heißen Datumsfelder anders. Wir versuchen robust mehrere.
    # 1) Sales Invoices
    sales_filters_candidates = [
        {"invoiceDate-ge": period_from, "invoiceDate-le": period_to},
        {"createdDate-ge": period_from, "createdDate-le": period_to},
        {"dueDate-ge": period_from, "dueDate-le": period_to},
    ]
    sales_docs = []
    last_sales_err = None
    for f in sales_filters_candidates:
        try:
            sales_docs = weclapp_get_all("salesInvoice", f)
            last_sales_err = None
            break
        except Exception as e:
            last_sales_err = str(e)

    # 2) Purchase Invoices (Fallback purchaseOrder)
    purchase_docs = []
    last_purchase_err = None

    purchase_entity_try = ["purchaseInvoice", "incomingInvoice", "purchaseOrder"]
    purchase_filters_candidates = [
        {"invoiceDate-ge": period_from, "invoiceDate-le": period_to},
        {"createdDate-ge": period_from, "createdDate-le": period_to},
        {"orderDate-ge": period_from, "orderDate-le": period_to},
    ]

    for ent in purchase_entity_try:
        for f in purchase_filters_candidates:
            try:
                purchase_docs = weclapp_get_all(ent, f)
                last_purchase_err = None
                purchase_entity_used = ent
                raise StopIteration
            except StopIteration:
                break
            except Exception as e:
                last_purchase_err = str(e)
        else:
            continue
        break
    else:
        purchase_entity_used = "purchaseInvoice"

    # Summierung: wir versuchen typische Tax-Felder (robust)
    def extract_tax_amount(doc: dict, direction: str) -> float:
        if direction == "sales":  # Umsatzsteuer
            return to_float_safe(pick_first(doc, [
                "salesTaxAmount", "vatAmount", "taxAmount", "totalTaxAmount", "outputTaxAmount"
            ], 0.0))
        else:  # Vorsteuer
            return to_float_safe(pick_first(doc, [
                "inputTaxAmount", "vatAmount", "taxAmount", "totalTaxAmount"
            ], 0.0))

    def extract_net(doc: dict) -> float:
        return to_float_safe(pick_first(doc, [
            "netAmount", "netTotal", "totalNetAmount", "amountNet"
        ], 0.0))

    def extract_gross(doc: dict) -> float:
        return to_float_safe(pick_first(doc, [
            "grossAmount", "grossTotal", "totalGrossAmount", "amountGross", "totalAmount"
        ], 0.0))

    def extract_number(doc: dict) -> str:
        return str(pick_first(doc, [
            "invoiceNumber", "number", "documentNumber", "orderNumber"
        ], ""))

    def extract_doc_date(doc: dict) -> str:
        return str(pick_first(doc, [
            "invoiceDate", "createdDate", "orderDate", "postingDate"
        ], ""))

    def extract_partner(doc: dict) -> str:
        # Kunde/Lieferant
        return str(pick_first(doc, [
            "customerName", "supplierName", "customer", "supplier", "businessPartnerName"
        ], ""))

    # Details DataFrames
    sales_rows = []
    for doc in sales_docs:
        sales_rows.append({
            "type": "salesInvoice",
            "date": extract_doc_date(doc),
            "number": extract_number(doc),
            "partner": extract_partner(doc),
            "net": round(extract_net(doc), 2),
            "tax": round(extract_tax_amount(doc, "sales"), 2),
            "gross": round(extract_gross(doc), 2),
            "id": doc.get("id", ""),
        })
    df_sales = pd.DataFrame(sales_rows)

    purchase_rows = []
    for doc in purchase_docs:
        purchase_rows.append({
            "type": purchase_entity_used,
            "date": extract_doc_date(doc),
            "number": extract_number(doc),
            "partner": extract_partner(doc),
            "net": round(extract_net(doc), 2),
            "tax": round(extract_tax_amount(doc, "purchase"), 2),
            "gross": round(extract_gross(doc), 2),
            "id": doc.get("id", ""),
        })
    df_purchase = pd.DataFrame(purchase_rows)

    umsatzsteuer_sum = float(df_sales["tax"].sum()) if not df_sales.empty else 0.0
    vorsteuer_sum = float(df_purchase["tax"].sum()) if not df_purchase.empty else 0.0
    saldo = round(vorsteuer_sum - umsatzsteuer_sum, 2)

    if saldo > 0:
        status = "Vorsteuerüberhang (Erstattung)"
    elif saldo < 0:
        status = "Vorsteuerlast (Nachzahlung)"
    else:
        status = "Ausgeglichen"

    # Excel Export
    details_xlsx = OUTPUT_DIR / "weclapp_ust_details.xlsx"
    with pd.ExcelWriter(details_xlsx, engine="openpyxl") as writer:
        df_sales.to_excel(writer, index=False, sheet_name="Umsatzsteuer (Sales)")
        df_purchase.to_excel(writer, index=False, sheet_name="Vorsteuer (Purchase)")
        pd.DataFrame([{
            "period_from": period_from,
            "period_to": period_to,
            "umsatzsteuer_sum": round(umsatzsteuer_sum, 2),
            "vorsteuer_sum": round(vorsteuer_sum, 2),
            "saldo_vorsteuer_minus_umsatzsteuer": round(saldo, 2),
            "status": status,
            "sales_count": int(len(df_sales)),
            "purchase_count": int(len(df_purchase)),
            "purchase_entity_used": purchase_entity_used,
            "sales_error_last": last_sales_err or "",
            "purchase_error_last": last_purchase_err or "",
        }]).to_excel(writer, index=False, sheet_name="Summary")

    return {
        "enabled": True,
        "error": "",
        "vorsteuer_sum": round(vorsteuer_sum, 2),
        "umsatzsteuer_sum": round(umsatzsteuer_sum, 2),
        "saldo": round(saldo, 2),
        "status": status,
        "details_file": str(details_xlsx.name),
        "sales_count": int(len(df_sales)),
        "purchase_count": int(len(df_purchase)),
        "purchase_entity_used": purchase_entity_used,
        "sales_error_last": last_sales_err or "",
        "purchase_error_last": last_purchase_err or "",
    }

# ============================================================
# HAUPTLOGIK: Matching + Weclapp Status
# ============================================================

def run_analysis(period_from: str, period_to: str):
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

    supplier_cols = [c for c in belege.columns if any(k in c.lower() for k in ["lieferant", "name", "adressat", "empfänger", "kunde", "geschäftspartner-name", "geschaeftspartner-name"])]
    invoice_cols = [c for c in belege.columns if any(k in c.lower() for k in ["rechnungsnummer", "rechnungs-nr", "belegfeld 1", "belegfeld1", "interne re"])]

    belege["supplier_text"] = belege.apply(lambda row: extract_supplier_text(row, supplier_cols, invoice_cols), axis=1)

    # Status
    status_col = find_column(belege, ["status", "verarbeitungsstatus", "belegstatus", "buchungsstatus", "gebucht"], default=None)
    if (not status_col) and STATUS_SPALTE_MANUELL and STATUS_SPALTE_MANUELL in belege.columns:
        status_col = STATUS_SPALTE_MANUELL

    if status_col:
        status_lower = belege[status_col].astype(str).str.lower().str.strip()
        belege["ist_gebucht"] = status_lower.eq("ja")
        belege["ist_posteingang"] = ~belege["ist_gebucht"]
    else:
        belege["ist_gebucht"] = True
        belege["ist_posteingang"] = False

    # Invoice Index
    invoice_index = build_invoice_index(belege)

    sichere_matches = []
    unklare_map = {}
    verwendete_konto_indices = set()

    # A) Invoice-first (Konto -> Beleg)
    invoice_hits_rows = []
    for _, krow in konto.iterrows():
        kidx = int(krow["konto_index"])
        if kidx in verwendete_konto_indices:
            continue
        hits = find_invoices_in_konto_text(krow.get("text_gesamt", ""), invoice_index)
        if not hits:
            continue

        hits_list = sorted(list(hits))
        if len(hits_list) > 1:
            unklare_map[f"konto_{kidx}_invoice"] = {
                "typ": "invoice_text_multi",
                "beleg_index": f"multi:{','.join(map(str, hits_list))}",
                "beleg_datum": pd.NaT,
                "beleg_betrag": np.nan,
                "beleg_supplier": "mehrere Belege via Rechnungsnr im Konto-Text",
                "beleg_rechnungsnr": "",
                "kandidaten": [{
                    "konto_index": kidx,
                    "konto_datum": krow.get("datum_norm", pd.NaT),
                    "konto_betrag": krow.get("betrag_raw", np.nan),
                    "konto_text": krow.get("text_gesamt", ""),
                    "score": 999,
                }],
            }
            verwendete_konto_indices.add(kidx)
            continue

        beleg_idx = hits_list[0]
        b = belege.loc[beleg_idx]

        verwendete_konto_indices.add(kidx)
        sichere_matches.append({
            "typ": "invoice_text",
            "score": 999,
            "konto_index": kidx,
            "konto_datum": krow.get("datum_norm", pd.NaT),
            "konto_betrag": krow.get("betrag_raw", np.nan),
            "konto_text": krow.get("text_gesamt", ""),
            "beleg_index": int(beleg_idx),
            "beleg_datum": b.get("datum_norm", pd.NaT),
            "beleg_betrag": b.get("betrag_raw", np.nan),
            "beleg_supplier": b.get("supplier_text", ""),
            "beleg_rechnungsnr": b.get("invoice_number", ""),
        })

        invoice_hits_rows.append({
            "konto_index": kidx,
            "beleg_index": int(beleg_idx),
            "invoice_number": b.get("invoice_number", ""),
            "konto_text": krow.get("text_gesamt", ""),
        })

    if invoice_hits_rows:
        pd.DataFrame(invoice_hits_rows).to_csv(OUTPUT_DIR / "invoice_text_matches.csv", sep=";", index=False, encoding="utf-8-sig")
    else:
        (OUTPUT_DIR / "invoice_text_matches.csv").write_text("keine invoice-text matches gefunden", encoding="utf-8")

    # B) Beleg -> Konto (Betrag/Datum/Score) nur für nicht belegte Kontozeilen
    def add_unklar(beleg, typ, candidates):
        entry = unklare_map.setdefault(
            int(beleg.name),
            {
                "typ": typ,
                "beleg_index": int(beleg.name),
                "beleg_datum": beleg.get("datum_norm", pd.NaT),
                "beleg_betrag": beleg.get("betrag_raw", np.nan),
                "beleg_supplier": beleg.get("supplier_text", ""),
                "beleg_rechnungsnr": beleg.get("invoice_number", ""),
                "kandidaten": [],
            }
        )
        for _, c in candidates.iterrows():
            entry["kandidaten"].append({
                "konto_index": int(c["konto_index"]),
                "konto_datum": c["datum_norm"],
                "konto_betrag": c["betrag_raw"],
                "konto_text": c["text_gesamt"],
                "score": c["score"],
            })

    # Gebuchte
    gebuchte = belege[belege["ist_gebucht"] == True].copy()
    for _, beleg in gebuchte.iterrows():
        betrag = beleg.get("betrag_raw", np.nan)
        datum  = beleg.get("datum_norm", pd.NaT)
        if pd.isna(betrag):
            continue

        betrag_abs = abs(float(betrag))
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)].copy()
        if candidates.empty:
            continue

        if not pd.isna(datum):
            diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
            candidates["datum_diff_tage"] = diff_days
            candidates = candidates[diff_days <= 45]
            if candidates.empty:
                continue
        else:
            candidates["datum_diff_tage"] = np.nan

        candidates["score"] = candidates["text_gesamt"].apply(
            lambda t: score_match(t, beleg.get("supplier_text", ""), beleg.get("invoice_number", ""))
        )
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True])
        best = candidates.iloc[0]

        if int(best["konto_index"]) in verwendete_konto_indices:
            continue

        if best["score"] >= 6 or (len(candidates) == 1 and best["score"] >= 1):
            verwendete_konto_indices.add(int(best["konto_index"]))
            sichere_matches.append({
                "typ": "gebucht",
                "score": best["score"],
                "konto_index": int(best["konto_index"]),
                "konto_datum": best["datum_norm"],
                "konto_betrag": best["betrag_raw"],
                "konto_text": best["text_gesamt"],
                "beleg_index": int(beleg.name),
                "beleg_datum": beleg.get("datum_norm", pd.NaT),
                "beleg_betrag": beleg.get("betrag_raw", np.nan),
                "beleg_supplier": beleg.get("supplier_text", ""),
                "beleg_rechnungsnr": beleg.get("invoice_number", ""),
            })
        else:
            add_unklar(beleg, "gebucht", candidates)

    # Posteingang
    posteingang = belege[belege["ist_posteingang"] == True].copy()
    for _, beleg in posteingang.iterrows():
        betrag = beleg.get("betrag_raw", np.nan)
        datum  = beleg.get("datum_norm", pd.NaT)
        if pd.isna(betrag):
            continue

        betrag_abs = abs(float(betrag))
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)].copy()
        if candidates.empty:
            continue

        if not pd.isna(datum):
            diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
            candidates["datum_diff_tage"] = diff_days
            candidates = candidates[diff_days <= DATUM_FENSTER_TAGE]
            if candidates.empty:
                continue
        else:
            candidates["datum_diff_tage"] = np.nan

        candidates["score"] = candidates["text_gesamt"].apply(
            lambda t: score_match(t, beleg.get("supplier_text", ""), beleg.get("invoice_number", ""))
        )
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True])
        best = candidates.iloc[0]

        if int(best["konto_index"]) in verwendete_konto_indices:
            continue

        second_score = candidates.iloc[1]["score"] if len(candidates) > 1 else None
        if len(candidates) > 1 and (best["score"] <= 0 or (second_score is not None and (best["score"] - second_score) < 2)):
            add_unklar(beleg, "posteingang", candidates)
            continue

        verwendete_konto_indices.add(int(best["konto_index"]))
        sichere_matches.append({
            "typ": "posteingang",
            "score": best["score"],
            "konto_index": int(best["konto_index"]),
            "konto_datum": best["datum_norm"],
            "konto_betrag": best["betrag_raw"],
            "konto_text": best["text_gesamt"],
            "beleg_index": int(beleg.name),
            "beleg_datum": beleg.get("datum_norm", pd.NaT),
            "beleg_betrag": beleg.get("betrag_raw", np.nan),
            "beleg_supplier": beleg.get("supplier_text", ""),
            "beleg_rechnungsnr": beleg.get("invoice_number", ""),
        })

    # Unklare zusammenfassen
    unklare_faelle = []
    for _, data in unklare_map.items():
        kandidaten = data.get("kandidaten", [])
        if not kandidaten:
            continue
        best = max(kandidaten, key=lambda c: c.get("score", -1))
        konto_indices = sorted({k["konto_index"] for k in kandidaten})
        konto_scores_str = "; ".join(
            f"{k['konto_index']}:{k['score']}"
            for k in sorted(kandidaten, key=lambda c: (-c.get("score", 0), str(c.get("konto_index"))))
        )
        unklare_faelle.append({
            "typ": data.get("typ", ""),
            "beleg_index": data.get("beleg_index", ""),
            "beleg_datum": data.get("beleg_datum", pd.NaT),
            "beleg_betrag": data.get("beleg_betrag", np.nan),
            "beleg_supplier": data.get("beleg_supplier", ""),
            "beleg_rechnungsnr": data.get("beleg_rechnungsnr", ""),
            "anzahl_konto_kandidaten": len(konto_indices),
            "konto_indices": ",".join(str(i) for i in konto_indices),
            "best_konto_index": best.get("konto_index", ""),
            "best_konto_datum": best.get("konto_datum", pd.NaT),
            "best_konto_betrag": best.get("konto_betrag", np.nan),
            "best_konto_text": best.get("konto_text", ""),
            "best_score": best.get("score", 0),
            "konto_indices_scores": konto_scores_str,
        })

    alle_verwendeten_konto = set(verwendete_konto_indices)
    for data in unklare_map.values():
        for k in data.get("kandidaten", []):
            alle_verwendeten_konto.add(int(k["konto_index"]))

    konto_ohne_beleg = konto[~konto["konto_index"].isin(alle_verwendeten_konto)].copy()
    konto_ohne_beleg["ist_kasse_vermutet"] = konto_ohne_beleg.apply(
        lambda row: looks_like_cash_booking(row.get("text_gesamt", ""), row.get("betrag_raw", np.nan)),
        axis=1
    )

    # Export CSV
    df_sicher = pd.DataFrame(sichere_matches)
    df_unklar = pd.DataFrame(unklare_faelle)

    (OUTPUT_DIR / "matches_sicher.csv").write_text("keine sicheren Matches gefunden", encoding="utf-8")
    if not df_sicher.empty:
        df_sicher.to_csv(OUTPUT_DIR / "matches_sicher.csv", sep=";", index=False, encoding="utf-8-sig")

    (OUTPUT_DIR / "matches_unklar.csv").write_text("keine unklaren Fälle gefunden", encoding="utf-8")
    if not df_unklar.empty:
        df_unklar.to_csv(OUTPUT_DIR / "matches_unklar.csv", sep=";", index=False, encoding="utf-8-sig")

    konto_ohne_beleg.to_csv(OUTPUT_DIR / "konto_ohne_beleg.csv", sep=";", index=False, encoding="utf-8-sig")

    # WECLAPP USt Status
    weclapp_res = summarize_weclapp_vat(period_from, period_to)

    return {
        "anzahl_sicher": len(df_sicher),
        "anzahl_sicher_gebucht": int((~df_sicher.empty) and (df_sicher["typ"] == "gebucht").sum()) if not df_sicher.empty else 0,
        "anzahl_sicher_post": int((~df_sicher.empty) and (df_sicher["typ"] == "posteingang").sum()) if not df_sicher.empty else 0,
        "anzahl_sicher_invoice": int((~df_sicher.empty) and (df_sicher["typ"] == "invoice_text").sum()) if not df_sicher.empty else 0,
        "anzahl_unklar": len(df_unklar),
        "anzahl_fehlende": len(konto_ohne_beleg),
        "anzahl_kasse": int(konto_ohne_beleg["ist_kasse_vermutet"].sum()) if not konto_ohne_beleg.empty else 0,
        "period_from": period_from,
        "period_to": period_to,
        "weclapp": weclapp_res,
    }

# ============================================================
# WEB UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    # Zeitraum UI: Standard = leer (User setzt), JS kann default last quarter setzen
    return f"""
    <html>
      <head>
        <title>NEXTWAVE AI Buchhaltung</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 16px;
          }}
          h1 {{ margin-bottom: 0.5rem; }}
          .logo {{ max-width: 260px; height: auto; margin-bottom: 10px; display: block; }}
          .hint {{ color: #555; margin-bottom: 1.2rem; line-height: 1.5; }}
          .field {{ margin-bottom: 1.2rem; }}
          .label {{ display: block; margin-bottom: 0.25rem; font-weight: 600; }}
          .dropzone {{
            border: 2px dashed #999;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background-color 0.2s;
          }}
          .dropzone.hover {{ border-color: #2563eb; background-color: #eff6ff; }}
          .filename {{ font-weight: 600; margin-top: 6px; }}
          .file-input {{ display: none; }}
          .row {{
            display:flex; gap:14px; align-items:flex-end; flex-wrap: wrap;
            border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fafafa;
            margin-bottom: 14px;
          }}
          .row .field {{ margin: 0; }}
          input[type="date"] {{
            padding: 10px; border-radius: 8px; border:1px solid #d1d5db; min-width: 200px;
          }}
          button {{
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 6px;
            border: none;
            background-color: #2563eb;
            color: white;
            cursor: pointer;
          }}
          button:hover {{ background-color: #1d4ed8; }}
          .progress {{
            margin-top: 1rem;
            font-size: 0.9rem;
            color: #2563eb;
            display: none;
            align-items: center;
            gap: 8px;
          }}
          .progress.active {{ display: inline-flex; }}
          .spinner {{
            width: 18px;
            height: 18px;
            border-radius: 999px;
            border: 3px solid #e5e7eb;
            border-top-color: #2563eb;
            animation: spin 0.8s linear infinite;
          }}
          @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
          .legal {{
            margin-top: 1.2rem;
            font-size: 0.75rem;
            color: #777;
            line-height: 1.4;
          }}
          .badge {{
            display:inline-block; padding: 4px 10px; border-radius: 999px; background:#e5e7eb; color:#111827; font-size: 12px;
          }}
        </style>
      </head>
      <body>
        <img src="/logo.png" alt="NEXTWAVE Logo" class="logo" />
        <h1>DATEV Kontoauszug / Belege Analyse</h1>

        <p class="hint">
          1) CSVs hochladen (Kontoauszug &amp; Belege).<br>
          2) Zeitraum wählen (z. B. Quartal).<br>
          3) Analyse starten → Matching + Weclapp USt-Status (Vorsteuer vs. Umsatzsteuer).<br>
          <span class="badge">Weclapp: {"aktiv" if weclapp_enabled() else "nicht konfiguriert"}</span>
        </p>

        <form id="uploadForm" action="/run" method="post" enctype="multipart/form-data">

          <div class="row">
            <div class="field">
              <span class="label">Zeitraum von</span>
              <input type="date" name="period_from" id="period_from" required />
            </div>
            <div class="field">
              <span class="label">Zeitraum bis</span>
              <input type="date" name="period_to" id="period_to" required />
            </div>
          </div>

          <div class="field">
            <span class="label">Kontoauszug CSV</span>
            <div id="konto_drop" class="dropzone">
              <div>CSV hierhin ziehen oder klicken</div>
              <div class="filename" id="konto_filename">Keine Datei ausgewählt</div>
            </div>
            <input class="file-input" type="file" name="konto_file" id="konto_file" accept=".csv" required />
          </div>

          <div class="field">
            <span class="label">Belege CSV</span>
            <div id="belege_drop" class="dropzone">
              <div>CSV hierhin ziehen oder klicken</div>
              <div class="filename" id="belege_filename">Keine Datei ausgewählt</div>
            </div>
            <input class="file-input" type="file" name="belege_file" id="belege_file" accept=".csv" required />
          </div>

          <button type="submit" id="submitBtn">Analyse starten</button>
          <div id="progress" class="progress">
            <div class="spinner"></div>
            <span>Analyse läuft …</span>
          </div>
        </form>

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

            drop.addEventListener('click', function() {{
              input.click();
            }});

            input.addEventListener('change', function() {{
              label.textContent = (input.files && input.files.length) ? input.files[0].name : "Keine Datei ausgewählt";
            }});

            ['dragenter','dragover'].forEach(eventName => {{
              drop.addEventListener(eventName, function(e) {{
                e.preventDefault(); e.stopPropagation();
                drop.classList.add('hover');
              }}, false);
            }});

            ['dragleave','drop'].forEach(eventName => {{
              drop.addEventListener(eventName, function(e) {{
                e.preventDefault(); e.stopPropagation();
                drop.classList.remove('hover');
              }}, false);
            }});

            drop.addEventListener('drop', function(e) {{
              const files = e.dataTransfer.files;
              if (files && files.length > 0) {{
                input.files = files;
                label.textContent = files[0].name;
              }}
            }});
          }}

          function setLastQuarterDefaults() {{
            const fromEl = document.getElementById('period_from');
            const toEl = document.getElementById('period_to');
            if (fromEl.value || toEl.value) return;

            const now = new Date();
            const y = now.getFullYear();
            const m = now.getMonth() + 1; // 1-12

            // Current quarter (1..4)
            const q = Math.floor((m - 1) / 3) + 1;
            // Last quarter
            let lq = q - 1;
            let ly = y;
            if (lq === 0) {{ lq = 4; ly = y - 1; }}

            const startMonth = (lq - 1) * 3 + 1;
            const endMonth = startMonth + 2;

            const pad = (n) => String(n).padStart(2, '0');
            const start = `${{ly}}-${{pad(startMonth)}}-01`;

            // Last day of endMonth
            const endDate = new Date(ly, endMonth, 0); // day 0 of next month
            const end = `${{ly}}-${{pad(endMonth)}}-${{pad(endDate.getDate())}}`;

            fromEl.value = start;
            toEl.value = end;
          }}

          document.addEventListener('DOMContentLoaded', function() {{
            setupDropzone('konto_drop', 'konto_file', 'konto_filename');
            setupDropzone('belege_drop', 'belege_file', 'belege_filename');
            setLastQuarterDefaults();

            const form = document.getElementById('uploadForm');
            const submitBtn = document.getElementById('submitBtn');
            const progress = document.getElementById('progress');
            const kontoInput = document.getElementById('konto_file');
            const belegeInput = document.getElementById('belege_file');
            const pf = document.getElementById('period_from');
            const pt = document.getElementById('period_to');

            form.addEventListener('submit', function(e) {{
              if (!kontoInput.files.length || !belegeInput.files.length) {{
                e.preventDefault();
                alert('Bitte sowohl Kontoauszug-CSV als auch Belege-CSV auswählen.');
                return;
              }}
              if (!pf.value || !pt.value) {{
                e.preventDefault();
                alert('Bitte Zeitraum (von/bis) auswählen.');
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

@app.post("/run", response_class=HTMLResponse)
async def run(
    konto_file: UploadFile = File(...),
    belege_file: UploadFile = File(...),
    period_from: str = Form(...),
    period_to: str = Form(...)
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(KONTOAUSZUG_CSV, "wb") as f:
        f.write(await konto_file.read())
    with open(BELEGE_CSV, "wb") as f:
        f.write(await belege_file.read())

    res = run_analysis(period_from, period_to)

    # ZIP bauen: alle CSVs + weclapp Excel
    zip_path = OUTPUT_DIR / "datev_analyse_ergebnisse.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUTPUT_DIR.glob("*.csv"):
            zf.write(p, arcname=p.name)
        for p in OUTPUT_DIR.glob("weclapp_ust_details.xlsx"):
            zf.write(p, arcname=p.name)

    # Weclapp Summary
    w = res.get("weclapp", {}) or {}
    weclapp_html = ""
    if not w.get("enabled"):
        weclapp_html = f"""
          <div class="box">
            <h2>Weclapp USt-Status</h2>
            <p style="color:#b91c1c;"><strong>Weclapp nicht aktiv:</strong> {w.get("error","")}</p>
            <p style="color:#374151;">Tipp: ENV setzen: <code>WECLAPP_BASE_URL</code> &amp; <code>WECLAPP_API_TOKEN</code></p>
          </div>
        """
    else:
        weclapp_html = f"""
          <div class="box">
            <h2>Weclapp USt-Status ({res['period_from']} bis {res['period_to']})</h2>
            <ul>
              <li><strong>Vorsteuer (Purchase):</strong> {w.get('vorsteuer_sum',0):.2f} €</li>
              <li><strong>Umsatzsteuer (Sales):</strong> {w.get('umsatzsteuer_sum',0):.2f} €</li>
              <li><strong>Saldo (Vorsteuer - Umsatzsteuer):</strong> {w.get('saldo',0):.2f} €</li>
              <li><strong>Status:</strong> {w.get('status','')}</li>
              <li><strong>Berücksichtigte Belege:</strong> Sales {w.get('sales_count',0)}, Purchase {w.get('purchase_count',0)} (Entity: {w.get('purchase_entity_used','')})</li>
              <li><strong>Export:</strong> <code>weclapp_ust_details.xlsx</code> (in der ZIP)</li>
            </ul>
          </div>
        """

    return f"""
    <html>
      <head>
        <title>Analyse abgeschlossen</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 16px; }}
          ul {{ line-height: 1.7; }}
          a.button {{ display:inline-block; margin-top: 1.2rem; padding:10px 18px; background:#2563eb; color:#fff; text-decoration:none; border-radius:6px; }}
          a.button:hover {{ background:#1d4ed8; }}
          .box {{ border:1px solid #e5e7eb; border-radius:10px; padding:14px; background:#fafafa; margin-top: 12px; }}
          .legal {{ margin-top:2rem; font-size:0.75rem; color:#777; line-height:1.4; }}
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
            <li>&nbsp;&nbsp;&bull; davon Rechnungsnr im Kontoauszug (Invoice-First): {res.get('anzahl_sicher_invoice',0)}</li>
            <li><strong>Unklare Fälle:</strong> {res['anzahl_unklar']}</li>
            <li><strong>Konto ohne Beleg:</strong> {res['anzahl_fehlende']}</li>
            <li>&nbsp;&nbsp;&bull; davon Kassenbuchungen vermutet: {res['anzahl_kasse']}</li>
          </ul>
        </div>

        {weclapp_html}

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
    uvicorn.run(app, host="127.0.0.1", port=8000)
