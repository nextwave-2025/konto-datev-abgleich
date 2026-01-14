import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import zipfile


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

DEBUG = True  # <- bei Bedarf auf False


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

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

    s = str(x)
    if not s.strip():
        return np.nan

    s = s.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    s = s.replace("€", "").replace("+", "")

    if s.endswith("-"):
        s = "-" + s[:-1]

    s = re.sub(r"[^0-9\-\.,]", "", s)
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


def score_match(konto_text, beleg_supplier_text, invoice_number):
    konto_text_norm = re.sub(r"[^a-zA-Z0-9]", "", str(konto_text).lower())
    score = 0

    for t in clean_tokens(beleg_supplier_text):
        if t in str(konto_text).lower():
            score += 1

    if invoice_number:
        inv = str(invoice_number).strip()
        inv_clean = re.sub(r"[^a-zA-Z0-9]", "", inv.lower())

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
        "kaufland", "denn", "dm", "rossmann", "apotheke",
        "tankstelle", "shell", "aral", "esso", "omv", "bft",
        "pos", "kartenzahlung", "ec-zahlung", "maestro", "visa",
        "mastercard"
    ]
    if any(kw in text for kw in kasse_keywords):
        try:
            amt = float(amount)
        except Exception:
            return False
        return abs(amt) <= 300
    return False


def pick_best_beleg_date_column(belege_df: pd.DataFrame) -> str | None:
    for keys in [
        ["rechnungsdatum", "rechnungsdat"],
        ["belegdatum"],
        ["leistungsdatum", "leistungsdat"],
        ["eingangsdatum", "eingangsdat", "eingangsd"],
        ["datum"],
    ]:
        c = find_column(belege_df, keys, default=None)
        if c:
            return c
    return None


def normalize_for_search(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def extract_refs(text_norm: str):
    if not text_norm:
        return []

    nums = re.findall(r"\d{7,12}", text_norm)

    def is_ddmmyyyy(n: str) -> bool:
        # ddmmyyyy
        if len(n) != 8:
            return False
        try:
            d = int(n[0:2])
            m = int(n[2:4])
            y = int(n[4:8])
            return 1 <= d <= 31 and 1 <= m <= 12 and 1990 <= y <= 2100
        except Exception:
            return False

    def is_yyyymmdd(n: str) -> bool:
        # yyyymmdd
        if len(n) != 8:
            return False
        try:
            y = int(n[0:4])
            m = int(n[4:6])
            d = int(n[6:8])
            return 1990 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31
        except Exception:
            return False

    out = []
    seen = set()
    for n in nums:
        # Filter: Datumszahlen raus
        if is_ddmmyyyy(n) or is_yyyymmdd(n):
            continue

        if n not in seen:
            seen.add(n)
            out.append(n)

    return out



# ============================================================
# HAUPTLOGIK
# ============================================================

def run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    debug_lines = []

    def dlog(*args):
        if DEBUG:
            debug_lines.append(" ".join(str(a) for a in args))

    # ------------------ Kontoauszug ------------------
    konto = pd.read_csv(KONTOAUSZUG_CSV, sep=";", dtype=str, encoding="latin1")

    amount_col = find_column(konto, ["umsatz (ohne soll/haben-kz)", "betrag", "umsatz", "betrag in eur"], default=None)
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
    konto["konto_index"] = konto.index.astype(int)
    konto["text_norm"] = konto["text_gesamt"].apply(lambda x: re.sub(r"[^0-9a-z]", "", str(x).lower()))

    # Ref-Index Konto: ref -> [konto_index,...]
    ref_to_konto = {}
    for _, row in konto.iterrows():
        kidx = int(row["konto_index"])
        for r in extract_refs(row.get("text_norm", "")):
            ref_to_konto.setdefault(r, []).append(kidx)
    dlog("Ref-Index Konto Größe:", len(ref_to_konto))

    # ------------------ Belege ------------------
    belege = pd.read_csv(BELEGE_CSV, sep=";", dtype=str, encoding="latin1")

    beleg_amount_col = find_column(belege, ["betrag", "rechnungsbetrag", "bruttobetrag", "bruttowert"], default=None)
    if not beleg_amount_col:
        raise ValueError("Konnte in der Belegliste keine Brutto-Betrags-Spalte finden.")
    belege["betrag_raw"] = belege[beleg_amount_col].apply(normalize_amount)

    beleg_date_col = pick_best_beleg_date_column(belege)
    if beleg_date_col:
        belege["datum_norm"] = pd.to_datetime(belege[beleg_date_col].apply(normalize_date), errors="coerce")
    else:
        belege["datum_norm"] = pd.NaT

    supplier_cols = [
        c for c in belege.columns
        if any(k in c.lower() for k in ["lieferant", "name", "adressat", "empfänger", "kunde", "geschäftspartner"])
    ]

    # erweitert
    invoice_cols = [
        c for c in belege.columns
        if any(k in c.lower() for k in [
            "rechnungsnummer", "rechnungs-nr", "rechnung nr", "invoice",
            "belegfeld 1", "belegfeld1",
            "referenz", "referenznr", "referenznummer",
            "vorgang", "vorgangsnummer",
            "kundenreferenz", "customer reference",
            "mandatsreferenz", "mandat",
            "payment reference", "verwendungszweck"
        ])
    ]

    belege["supplier_text"] = belege.apply(lambda row: extract_supplier_text(row, supplier_cols, invoice_cols), axis=1)

    beleg_invoice_col = invoice_cols[0] if invoice_cols else None
    belege["invoice_number"] = belege[beleg_invoice_col] if beleg_invoice_col else ""
    belege["invoice_norm"] = belege["invoice_number"].apply(normalize_for_search)

    # Ganz wichtig: „ALLE SPALTEN“-Text, um Referenzen überall zu finden
    belege["text_all"] = belege.astype(str).agg(" ".join, axis=1)
    belege["text_norm_all"] = belege["text_all"].apply(lambda x: re.sub(r"[^0-9a-z]", "", str(x).lower()))

    # Ref-Index Belege: ref -> [beleg_index,...]
    ref_to_beleg = {}
    for idx, row in belege.iterrows():
        for r in extract_refs(row.get("text_norm_all", "")):
            ref_to_beleg.setdefault(r, []).append(idx)
    dlog("Ref-Index Belege Größe:", len(ref_to_beleg))

    # ------------------ Status ------------------
    status_col = find_column(belege, ["gebucht", "status", "belegstatus", "buchungsstatus"], default=None)
    if (not status_col) and STATUS_SPALTE_MANUELL and STATUS_SPALTE_MANUELL in belege.columns:
        status_col = STATUS_SPALTE_MANUELL

    if status_col:
        status_lower = belege[status_col].astype(str).str.lower().str.strip()
        belege["ist_gebucht"] = status_lower.isin(["ja", "true", "1", "x", "gebucht"])
        belege["ist_posteingang"] = ~belege["ist_gebucht"]
    else:
        belege["ist_gebucht"] = True
        belege["ist_posteingang"] = False

    # ------------------ Matching (Beleg -> Konto wie gehabt) ------------------
    sichere_matches = []
    unklare_map = {}
    verwendete_konto_indices = set()
    verwendete_beleg_indices = set()

    def add_unklar(beleg_idx, beleg_row, kandidaten_rows, typ, base_score=0):
        entry = unklare_map.setdefault(
            beleg_idx,
            {
                "typ": typ,
                "beleg_index": beleg_idx,
                "beleg_datum": beleg_row["datum_norm"],
                "beleg_betrag": beleg_row["betrag_raw"],
                "beleg_supplier": beleg_row["supplier_text"],
                "beleg_rechnungsnr": beleg_row["invoice_number"],
                "kandidaten": [],
            },
        )
        for _, c in kandidaten_rows.iterrows():
            entry["kandidaten"].append({
                "konto_index": int(c["konto_index"]),
                "konto_datum": c["datum_norm"],
                "konto_betrag": c["betrag_raw"],
                "konto_text": c["text_gesamt"],
                "score": int(c.get("score", base_score)),
            })

    gebuchte = belege[belege["ist_gebucht"] == True].copy()

    for beleg_idx, beleg in gebuchte.iterrows():
        betrag = beleg["betrag_raw"]
        datum = beleg["datum_norm"]
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(betrag)

        # 1) Betrag Kandidaten
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)].copy()
        if candidates.empty:
            continue

        candidates["datum_diff_tage"] = (candidates["datum_norm"] - datum).dt.days.abs()

        sup_txt = beleg["supplier_text"]
        inv_nr = beleg["invoice_number"]

        candidates["score"] = candidates["text_gesamt"].apply(lambda t: score_match(t, sup_txt, inv_nr))
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True])

        best = candidates.iloc[0]
        best_kidx = int(best["konto_index"])

        if len(candidates) == 1 or int(best["score"]) >= 6:
            if best_kidx not in verwendete_konto_indices:
                verwendete_konto_indices.add(best_kidx)
                verwendete_beleg_indices.add(beleg_idx)
                sichere_matches.append({
                    "typ": "gebucht",
                    "score": int(best["score"]),
                    "konto_index": best_kidx,
                    "konto_datum": best["datum_norm"],
                    "konto_betrag": best["betrag_raw"],
                    "konto_text": best["text_gesamt"],
                    "beleg_index": beleg_idx,
                    "beleg_datum": beleg["datum_norm"],
                    "beleg_betrag": beleg["betrag_raw"],
                    "beleg_supplier": beleg["supplier_text"],
                    "beleg_rechnungsnr": beleg["invoice_number"],
                })
        else:
            add_unklar(beleg_idx, beleg, candidates, "gebucht", base_score=0)

    # ------------------ Reverse Ref Matching (Konto -> Beleg über Referenz) ------------------
    # Damit verschwinden Stadtwerke & Co aus konto_ohne_beleg, auch wenn Ref nicht in invoice_cols steckt.
    ref_kandidaten_rows = []

    # nur Konten, die bisher NICHT verwendet wurden
    remaining_konto = konto[~konto["konto_index"].isin(verwendete_konto_indices)].copy()

    for _, krow in remaining_konto.iterrows():
        kidx = int(krow["konto_index"])
        krefs = extract_refs(krow.get("text_norm", ""))

        if not krefs:
            continue

        # Sammle alle Belege, die eine dieser Referenzen irgendwo im Text haben
        beleg_hits = []
        hit_ref = None
        for r in krefs:
            if r in ref_to_beleg:
                beleg_hits.extend(ref_to_beleg[r])
                hit_ref = r  # nur fürs Logging
        beleg_hits = sorted(set(beleg_hits))

        if not beleg_hits:
            continue

        # Wenn genau 1 Beleg -> sicher matchen (wenn Konto noch frei)
        if len(beleg_hits) == 1:
            bidx = beleg_hits[0]
            if bidx in verwendete_beleg_indices:
                # Beleg schon anders gematcht -> trotzdem Konto markieren, damit nicht in "ohne_beleg" bleibt
                verwendete_konto_indices.add(kidx)
                ref_kandidaten_rows.append({
                    "konto_index": kidx,
                    "konto_datum": krow["datum_norm"],
                    "konto_betrag": krow["betrag_raw"],
                    "konto_text": krow["text_gesamt"],
                    "ref": hit_ref,
                    "beleg_index": bidx,
                    "hinweis": "Beleg bereits gematcht – Konto nur entfernt (Ref erkannt)"
                })
                continue

            brow = belege.loc[bidx]
            verwendete_konto_indices.add(kidx)
            verwendete_beleg_indices.add(bidx)

            typ = "gebucht" if bool(brow.get("ist_gebucht", True)) else "posteingang"
            sichere_matches.append({
                "typ": f"ref_reverse_{typ}",
                "score": 850,
                "konto_index": kidx,
                "konto_datum": krow["datum_norm"],
                "konto_betrag": krow["betrag_raw"],
                "konto_text": krow["text_gesamt"],
                "beleg_index": bidx,
                "beleg_datum": brow.get("datum_norm", pd.NaT),
                "beleg_betrag": brow.get("betrag_raw", np.nan),
                "beleg_supplier": brow.get("supplier_text", ""),
                "beleg_rechnungsnr": brow.get("invoice_number", ""),
            })
            dlog("SICHER via REF_REVERSE:", "konto", kidx, "ref", hit_ref, "-> beleg", bidx)
        else:
            # Mehrere Belege -> unklar, aber Konto muss aus "ohne_beleg" raus
            verwendete_konto_indices.add(kidx)
            ref_kandidaten_rows.append({
                "konto_index": kidx,
                "konto_datum": krow["datum_norm"],
                "konto_betrag": krow["betrag_raw"],
                "konto_text": krow["text_gesamt"],
                "ref": hit_ref,
                "beleg_indices": ",".join(str(x) for x in beleg_hits),
                "hinweis": "Mehrere Belege zur Referenz – Konto entfernt, bitte manuell prüfen"
            })
            dlog("UNKLAR via REF_REVERSE:", "konto", kidx, "ref", hit_ref, "belege", len(beleg_hits))

    # ------------------ Unklare zusammenfassen ------------------
    unklare_faelle = []
    for beleg_idx, data in unklare_map.items():
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

    # ------------------ Konto ohne Beleg ------------------
    alle_verwendeten_konto = set(int(m["konto_index"]) for m in sichere_matches)

    for data in unklare_map.values():
        for k in data["kandidaten"]:
            alle_verwendeten_konto.add(int(k["konto_index"]))

    # Dazu: verwendete_konto_indices (inkl. reverse-ref)
    alle_verwendeten_konto.update(set(int(x) for x in verwendete_konto_indices))

    konto_ohne_beleg = konto[~konto["konto_index"].isin(alle_verwendeten_konto)].copy()
    konto_ohne_beleg["ist_kasse_vermutet"] = konto_ohne_beleg.apply(
        lambda row: looks_like_cash_booking(row["text_gesamt"], row["betrag_raw"]),
        axis=1
    )

    # ------------------ CSV-Ausgaben ------------------
    df_sicher = pd.DataFrame(sichere_matches)
    df_unklar = pd.DataFrame(unklare_faelle)
    df_ref_kand = pd.DataFrame(ref_kandidaten_rows)

    if not df_sicher.empty:
        df_sicher.to_csv(OUTPUT_DIR / "matches_sicher.csv", sep=";", index=False, encoding="utf-8-sig")
    else:
        (OUTPUT_DIR / "matches_sicher.csv").write_text("keine sicheren Matches gefunden", encoding="utf-8")

    if not df_unklar.empty:
        df_unklar.to_csv(OUTPUT_DIR / "matches_unklar.csv", sep=";", index=False, encoding="utf-8-sig")
    else:
        (OUTPUT_DIR / "matches_unklar.csv").write_text("keine unklaren Fälle gefunden", encoding="utf-8")

    if not df_ref_kand.empty:
        df_ref_kand.to_csv(OUTPUT_DIR / "konto_ref_kandidaten.csv", sep=";", index=False, encoding="utf-8-sig")
    else:
        # optional: nicht erzeugen oder leer schreiben
        (OUTPUT_DIR / "konto_ref_kandidaten.csv").write_text("keine ref-kandidaten", encoding="utf-8")

    konto_ohne_beleg.to_csv(OUTPUT_DIR / "konto_ohne_beleg.csv", sep=";", index=False, encoding="utf-8-sig")

    if DEBUG:
        (OUTPUT_DIR / "debug_logs.txt").write_text("\n".join(debug_lines), encoding="utf-8")

    anzahl_sicher = len(df_sicher) if not df_sicher.empty else 0
    anzahl_sicher_gebucht = len(df_sicher[df_sicher["typ"].astype(str).str.contains("gebucht")]) if not df_sicher.empty else 0
    anzahl_sicher_post = len(df_sicher[df_sicher["typ"].astype(str).str.contains("posteingang")]) if not df_sicher.empty else 0
    anzahl_unklar = len(df_unklar) if not df_unklar.empty else 0
    anzahl_fehlende = len(konto_ohne_beleg)
    anzahl_kasse = int(konto_ohne_beleg["ist_kasse_vermutet"].sum()) if not konto_ohne_beleg.empty else 0

    return {
        "anzahl_sicher": anzahl_sicher,
        "anzahl_sicher_gebucht": anzahl_sicher_gebucht,
        "anzahl_sicher_post": anzahl_sicher_post,
        "anzahl_unklar": anzahl_unklar,
        "anzahl_fehlende": anzahl_fehlende,
        "anzahl_kasse": anzahl_kasse,
    }


# ============================================================
# WEB UI
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <head>
        <title>NEXTWAVE AI Buchhaltung</title>
        <style>
          body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 800px; margin: 40px auto; padding: 0 16px; }
          .logo { max-width: 260px; height: auto; margin-bottom: 10px; display: block; }
          .hint { color: #555; margin-bottom: 1.5rem; }
          .field { margin-bottom: 1.2rem; }
          .label { display: block; margin-bottom: 0.25rem; font-weight: 600; }
          .dropzone { border: 2px dashed #999; border-radius: 8px; padding: 14px; text-align: center; cursor: pointer; transition: border-color 0.2s, background-color 0.2s; }
          .dropzone.hover { border-color: #2563eb; background-color: #eff6ff; }
          .filename { font-weight: 600; margin-top: 6px; }
          .file-input { display: none; }
          button { padding: 10px 20px; font-size: 16px; border-radius: 6px; border: none; background-color: #2563eb; color: white; cursor: pointer; }
          button:hover { background-color: #1d4ed8; }
          .progress { margin-top: 1rem; font-size: 0.9rem; color: #2563eb; display: none; align-items: center; gap: 8px; }
          .progress.active { display: inline-flex; }
          .spinner { width: 18px; height: 18px; border-radius: 999px; border: 3px solid #e5e7eb; border-top-color: #2563eb; animation: spin 0.8s linear infinite; }
          @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
          .legal { margin-top: 0.8rem; font-size: 0.75rem; color: #777; line-height: 1.4; }
        </style>
      </head>
      <body>
        <img src="/logo.png" alt="NEXTWAVE Logo" class="logo" />
        <h1>DATEV Kontoauszug / Belege Analyse</h1>
        <p class="hint">
          CSVs hochladen (Kontoauszug & Belege) und Analyse starten. Ergebnis als ZIP herunterladen.
        </p>

        <form id="uploadForm" action="/run" method="post" enctype="multipart/form-data">
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
            <span>Analyse läuft, bitte warten …</span>
          </div>
        </form>

        <div class="legal">
          © NEXTWAVE GmbH – Alle Rechte vorbehalten.<br>
          Die Nutzung dieses Programms oder von Teilen daraus ohne vorherige schriftliche Zustimmung der NEXTWAVE GmbH
          ist untersagt und kann zivil- und strafrechtliche Schritte nach sich ziehen.
        </div>

        <script>
          function setupDropzone(dropId, inputId, labelId) {
            const drop = document.getElementById(dropId);
            const input = document.getElementById(inputId);
            const label = document.getElementById(labelId);

            drop.addEventListener('click', function() { input.click(); });

            input.addEventListener('change', function() {
              if (input.files && input.files.length > 0) label.textContent = input.files[0].name;
              else label.textContent = "Keine Datei ausgewählt";
            });

            ['dragenter', 'dragover'].forEach(eventName => {
              drop.addEventListener(eventName, function(e) {
                e.preventDefault(); e.stopPropagation();
                drop.classList.add('hover');
              }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
              drop.addEventListener(eventName, function(e) {
                e.preventDefault(); e.stopPropagation();
                drop.classList.remove('hover');
              }, false);
            });

            drop.addEventListener('drop', function(e) {
              const files = e.dataTransfer.files;
              if (files && files.length > 0) {
                input.files = files;
                label.textContent = files[0].name;
              }
            });
          }

          document.addEventListener('DOMContentLoaded', function() {
            setupDropzone('konto_drop', 'konto_file', 'konto_filename');
            setupDropzone('belege_drop', 'belege_file', 'belege_filename');

            const form = document.getElementById('uploadForm');
            const submitBtn = document.getElementById('submitBtn');
            const progress = document.getElementById('progress');
            const kontoInput = document.getElementById('konto_file');
            const belegeInput = document.getElementById('belege_file');

            form.addEventListener('submit', function(e) {
              if (!kontoInput.files.length || !belegeInput.files.length) {
                e.preventDefault();
                alert('Bitte sowohl Kontoauszug-CSV als auch Belege-CSV auswählen.');
                return;
              }
              submitBtn.disabled = true;
              submitBtn.textContent = 'Analyse läuft ...';
              progress.classList.add('active');
            });
          });
        </script>
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
        for file in OUTPUT_DIR.iterdir():
            if file.is_file() and file.suffix.lower() in [".csv", ".txt"]:
                zf.write(file, arcname=file.name)

    return f"""
    <html>
      <head>
        <title>Analyse abgeschlossen</title>
        <style>
          body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 800px; margin: 40px auto; padding: 0 16px; }}
          ul {{ line-height: 1.6; }}
          a.button {{ display: inline-block; margin-top: 1.2rem; padding: 10px 18px; background-color: #2563eb; color: white; text-decoration: none; border-radius: 6px; }}
          a.button:hover {{ background-color: #1d4ed8; }}
          .legal {{ margin-top: 2rem; font-size: 0.75rem; color: #777; line-height: 1.4; }}
        </style>
      </head>
      <body>
        <h1>Analyse abgeschlossen</h1>
        <ul>
          <li><strong>Sichere Matches gesamt:</strong> {res['anzahl_sicher']}</li>
          <li>&nbsp;&nbsp;&bull; davon gebuchte Belege: {res['anzahl_sicher_gebucht']}</li>
          <li>&nbsp;&nbsp;&bull; davon Posteingang-Rechnungen: {res['anzahl_sicher_post']}</li>
          <li><strong>Unklare Fälle:</strong> {res['anzahl_unklar']}</li>
          <li><strong>Fehlende Belege gesamt:</strong> {res['anzahl_fehlende']}</li>
          <li>&nbsp;&nbsp;&bull; davon Kassenbuchungen mit Beleg vermutlich im Posteingang: {res['anzahl_kasse']}</li>
        </ul>

        <a href="/download" class="button">Ergebnis-ZIP herunterladen</a>
        <div style="margin-top: 1rem;"><a href="/">Neue Analyse starten</a></div>

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
