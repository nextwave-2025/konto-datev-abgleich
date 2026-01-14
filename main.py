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
    """
    Robust gegen:
    - NBSP / schmale Leerzeichen
    - + / Minus am Ende (z.B. 1888,53-)
    - Tausender-Trennzeichen . oder Leerzeichen
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x)
    if not s.strip():
        return np.nan

    # Sonder-Leerzeichen entfernen
    s = s.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")

    # Euro & Plus entfernen
    s = s.replace("€", "").replace("+", "")

    # Minus am Ende -> Minus vorne
    if s.endswith("-"):
        s = "-" + s[:-1]

    # Nur erlaubte Zeichen lassen (Ziffern, - , . , ,)
    s = re.sub(r"[^0-9\-\.,]", "", s)

    # DE-Schreibweise -> float
    # Tausenderpunkte entfernen, Komma zu Punkt
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
    # Priorität: Rechnungsdatum -> Belegdatum -> Leistungsdatum -> Eingangsdatum -> irgendein Datum
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


def normalize_for_invoice_search(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


# ============================================================
# HAUPTLOGIK
# ============================================================

def run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    konto["konto_index"] = konto.index
    konto["text_norm"] = konto["text_gesamt"].apply(lambda x: re.sub(r"[^0-9a-z]", "", str(x).lower()))

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
    invoice_cols = [
        c for c in belege.columns
        if any(k in c.lower() for k in ["rechnungsnummer", "rechnungs-nr", "belegfeld 1", "belegfeld1"])
    ]

    belege["supplier_text"] = belege.apply(lambda row: extract_supplier_text(row, supplier_cols, invoice_cols), axis=1)

    beleg_invoice_col = invoice_cols[0] if invoice_cols else None
    belege["invoice_number"] = belege[beleg_invoice_col] if beleg_invoice_col else ""
    belege["invoice_norm"] = belege["invoice_number"].apply(normalize_for_invoice_search)

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

    # ------------------ Matching ------------------
    sichere_matches = []
    unklare_map = {}
    verwendete_konto_indices = set()

    gebuchte = belege[belege["ist_gebucht"] == True].copy()

    for _, beleg in gebuchte.iterrows():
        betrag = beleg["betrag_raw"]
        datum = beleg["datum_norm"]

        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(betrag)

        # 1) Primär: Betrag-Matching
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)].copy()

        # ✅ 2) Fallback: Rechnungsnummer direkt im Banktext suchen (wenn Betrag-Matching nichts findet)
        if candidates.empty and beleg.get("invoice_norm"):
            inv = beleg["invoice_norm"]
            if inv and len(inv) >= 4:
                inv_hits = konto[konto["text_norm"].str.contains(inv, na=False)].copy()
                # Wenn eindeutig genau 1 Treffer -> direkt sicher matchen
                if len(inv_hits) == 1:
                    best = inv_hits.iloc[0]
                    if best["konto_index"] not in verwendete_konto_indices:
                        verwendete_konto_indices.add(best["konto_index"])
                        sichere_matches.append({
                            "typ": "gebucht",
                            "score": 999,
                            "konto_index": best["konto_index"],
                            "konto_datum": best["datum_norm"],
                            "konto_betrag": best["betrag_raw"],
                            "konto_text": best["text_gesamt"],
                            "beleg_index": beleg.name,
                            "beleg_datum": beleg["datum_norm"],
                            "beleg_betrag": beleg["betrag_raw"],
                            "beleg_supplier": beleg["supplier_text"],
                            "beleg_rechnungsnr": beleg["invoice_number"],
                        })
                    continue
                # Wenn mehrere Treffer -> unklar
                elif len(inv_hits) > 1:
                    entry = unklare_map.setdefault(
                        beleg.name,
                        {
                            "typ": "gebucht",
                            "beleg_index": beleg.name,
                            "beleg_datum": beleg["datum_norm"],
                            "beleg_betrag": beleg["betrag_raw"],
                            "beleg_supplier": beleg["supplier_text"],
                            "beleg_rechnungsnr": beleg["invoice_number"],
                            "kandidaten": [],
                        },
                    )
                    for _, c in inv_hits.iterrows():
                        entry["kandidaten"].append({
                            "konto_index": c["konto_index"],
                            "konto_datum": c["datum_norm"],
                            "konto_betrag": c["betrag_raw"],
                            "konto_text": c["text_gesamt"],
                            "score": 500,
                        })
                    continue

        if candidates.empty:
            continue

        candidates["datum_diff_tage"] = (candidates["datum_norm"] - datum).dt.days.abs()

        sup_txt = beleg["supplier_text"]
        inv_nr = beleg["invoice_number"]

        candidates["score"] = candidates["text_gesamt"].apply(lambda t: score_match(t, sup_txt, inv_nr))
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True])

        best = candidates.iloc[0]

        if len(candidates) == 1:
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
                    "beleg_datum": beleg["datum_norm"],
                    "beleg_betrag": beleg["betrag_raw"],
                    "beleg_supplier": beleg["supplier_text"],
                    "beleg_rechnungsnr": beleg["invoice_number"],
                })
            continue

        if best["score"] >= 6:
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
                    "beleg_datum": beleg["datum_norm"],
                    "beleg_betrag": beleg["betrag_raw"],
                    "beleg_supplier": beleg["supplier_text"],
                    "beleg_rechnungsnr": beleg["invoice_number"],
                })
        else:
            entry = unklare_map.setdefault(
                beleg.name,
                {
                    "typ": "gebucht",
                    "beleg_index": beleg.name,
                    "beleg_datum": beleg["datum_norm"],
                    "beleg_betrag": beleg["betrag_raw"],
                    "beleg_supplier": beleg["supplier_text"],
                    "beleg_rechnungsnr": beleg["invoice_number"],
                    "kandidaten": [],
                },
            )
            for _, c in candidates.iterrows():
                entry["kandidaten"].append({
                    "konto_index": c["konto_index"],
                    "konto_datum": c["datum_norm"],
                    "konto_betrag": c["betrag_raw"],
                    "konto_text": c["text_gesamt"],
                    "score": c["score"],
                })

    # Posteingang bleibt wie gehabt (optional)
    posteingang = belege[belege["ist_posteingang"] == True].copy()
    for _, beleg in posteingang.iterrows():
        betrag = beleg["betrag_raw"]
        datum = beleg["datum_norm"]
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(betrag)
        candidates = konto[(konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)].copy()
        if candidates.empty:
            continue

        candidates["datum_diff_tage"] = (candidates["datum_norm"] - datum).dt.days.abs()
        candidates = candidates[candidates["datum_diff_tage"] <= DATUM_FENSTER_TAGE]
        if candidates.empty:
            continue

        sup_txt = beleg["supplier_text"]
        inv_nr = beleg["invoice_number"]
        candidates["score"] = candidates["text_gesamt"].apply(lambda t: score_match(t, sup_txt, inv_nr))
        candidates = candidates.sort_values(["score", "datum_diff_tage"], ascending=[False, True])

        best = candidates.iloc[0]
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
                "beleg_datum": beleg["datum_norm"],
                "beleg_betrag": beleg["betrag_raw"],
                "beleg_supplier": beleg["supplier_text"],
                "beleg_rechnungsnr": beleg["invoice_number"],
            })

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
    alle_verwendeten_konto = {m["konto_index"] for m in sichere_matches}
    for data in unklare_map.values():
        for k in data["kandidaten"]:
            alle_verwendeten_konto.add(k["konto_index"])

    konto_ohne_beleg = konto[~konto["konto_index"].isin(alle_verwendeten_konto)].copy()
    konto_ohne_beleg["ist_kasse_vermutet"] = konto_ohne_beleg.apply(
        lambda row: looks_like_cash_booking(row["text_gesamt"], row["betrag_raw"]),
        axis=1
    )

    # ------------------ CSV-Ausgaben ------------------
    df_sicher = pd.DataFrame(sichere_matches)
    df_unklar = pd.DataFrame(unklare_faelle)

    df_sicher.to_csv(OUTPUT_DIR / "matches_sicher.csv", sep=";", index=False, encoding="utf-8-sig") \
        if not df_sicher.empty else (OUTPUT_DIR / "matches_sicher.csv").write_text("keine sicheren Matches gefunden", encoding="utf-8")

    df_unklar.to_csv(OUTPUT_DIR / "matches_unklar.csv", sep=";", index=False, encoding="utf-8-sig") \
        if not df_unklar.empty else (OUTPUT_DIR / "matches_unklar.csv").write_text("keine unklaren Fälle gefunden", encoding="utf-8")

    konto_ohne_beleg.to_csv(OUTPUT_DIR / "konto_ohne_beleg.csv", sep=";", index=False, encoding="utf-8-sig")

    anzahl_sicher = len(df_sicher)
    anzahl_sicher_gebucht = len(df_sicher[df_sicher["typ"] == "gebucht"]) if not df_sicher.empty else 0
    anzahl_sicher_post = len(df_sicher[df_sicher["typ"] == "posteingang"]) if not df_sicher.empty else 0
    anzahl_unklar = len(df_unklar)
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
        for csv_file in OUTPUT_DIR.glob("*.csv"):
            zf.write(csv_file, arcname=csv_file.name)

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
