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
# evtl. weitere Imports von dir hier (os, etc.)

# ============================================================
# KONFIGURATION – HIER BEI BEDARF ANPASSEN
# ============================================================

BASE_DIR = Path(__file__).parent

# Eingangsdateien, werden bei Upload überschrieben
KONTOAUSZUG_CSV = BASE_DIR / "kontoauszug.csv"
BELEGE_CSV     = BASE_DIR / "belege.csv"

# Ausgabe-Ordner
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FASTAPI – WEB-INTERFACE ...
# ============================================================

app = FastAPI()

# *** WICHTIG: Logo & andere Dateien aus dem Projektordner serven ***
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


# Toleranzen
BETRAG_TOLERANZ = 0.01        # 1 Cent
DATUM_FENSTER_TAGE = 30       # +/- 30 Tage für Posteingang-Matches

# Falls du eine bestimmte Status-Spalte erzwingen willst (z.B. "Gebucht"):
STATUS_SPALTE_MANUELL = None  # oder z.B. "Gebucht"


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def find_column(df, keywords, default=None, prefer_contains=None):
    """
    Sucht in df-Spalten nach einem der keywords (lowercase-Vergleich).
    prefer_contains: wenn gesetzt, wird diese Teilzeichenfolge bevorzugt.
    """
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]

    # 1. exakte Treffer
    for key in keywords:
        for orig, low in zip(cols, cols_lower):
            if key == low:
                return orig

    # 2. Teilzeichenfolge
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
    s = str(x).strip().replace(" ", "")
    s = s.replace("€", "")
    # deutsche Schreibweise
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
    txt = " ".join(parts)
    return txt


def clean_tokens(text):
    text = re.sub(r"[^a-z0-9äöüß ]", " ", str(text).lower())
    tokens = [t for t in text.split() if len(t) >= 4]
    return tokens


def normalize_for_invoice_match(s):
    """Nur Buchstaben/Ziffern, alles klein – für robuste Rechnungsnummer-Suche."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def score_match(konto_text, beleg_supplier_text, invoice_number):
    """
    Scoring:
    - Lieferanten-Tokens im Konto-Text: +1 pro Treffer
    - Rechnungsnummer im Konto-Text (robust gesucht): +10
    - Exakte Rechnungsnummer (streng) im Text: +20
    """
    konto_text_norm = re.sub(r"[^a-zA-Z0-9]", "", konto_text.lower())

    score = 0

    # -----------------------------------------
    # 1) Supplier / Lieferanten-Name Tokens
    # -----------------------------------------
    supplier_tokens = clean_tokens(beleg_supplier_text)
    for t in supplier_tokens:
        if t in konto_text.lower():
            score += 1

    # -----------------------------------------
    # 2) Rechnungsnummer – starke Gewichtung
    # -----------------------------------------
    if invoice_number:
        inv = str(invoice_number).strip()

        # Normalisierung
        inv_clean = re.sub(r"[^a-zA-Z0-9]", "", inv.lower())

        # Teilnummern extrahieren (z.B. letzten 4–8 Stellen)
        partials = set()
        if len(inv_clean) >= 4:
            partials.add(inv_clean[-4:])
        if len(inv_clean) >= 6:
            partials.add(inv_clean[-6:])
        if len(inv_clean) >= 8:
            partials.add(inv_clean[-8:])
        partials.add(inv_clean)

        # Volltext-Suche
        for p in partials:
            if p and p in konto_text_norm:
                score += 10

        # extra hoch bewerten: exakte Übereinstimmung
        if inv_clean in konto_text_norm:
            score += 20

    return score



def looks_like_cash_booking(konto_text, amount):
    """
    Heuristik für Kassenbuchung:
    - Typische Begriffe: Supermärkte, Tankstellen, EC/POS etc.
    - eher kleinere Beträge
    """
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
        if abs(amt) <= 300:
            return True
    return False


# ============================================================
# HAUPTLOGIK
# ============================================================

def run_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------ Kontoauszug einlesen ------------------
    konto = pd.read_csv(KONTOAUSZUG_CSV, sep=";", dtype=str, encoding="latin1")

    # Beträge
    amount_col = find_column(
        konto,
        ["umsatz (ohne soll/haben-kz)", "betrag", "umsatz", "betrag in eur"],
        default=None
    )
    if not amount_col:
        raise ValueError("Konnte im Kontoauszug keine Betrags-Spalte finden.")

    konto["betrag_raw"] = konto[amount_col].apply(normalize_amount)

    # Datum
    date_col = find_column(
        konto,
        ["buchungstag", "buchungsdatum", "datum"],
        default=None
    )
    if not date_col:
        raise ValueError("Konnte im Kontoauszug keine Datums-Spalte finden.")

    konto["datum_norm"] = konto[date_col].apply(normalize_date)
    konto["datum_norm"] = pd.to_datetime(konto["datum_norm"], errors="coerce")

    # Text
    konto["text_gesamt"] = build_text_field(
        konto,
        ["buchungstext", "verwendungszweck", "name", "empfänger", "begünstigter", "auftraggeber"]
    )

    # Index merken
    konto["konto_index"] = konto.index

    # ------------------ Belege einlesen ------------------
    belege = pd.read_csv(BELEGE_CSV, sep=";", dtype=str, encoding="latin1")

    # Betrag / Brutto
    beleg_amount_col = find_column(
        belege,
        ["bruttobetrag", "bruttowert", "rechnungsbetrag", "betrag"],
        default=None
    )
    if not beleg_amount_col:
        raise ValueError("Konnte in der Belegliste keine Brutto-Betrags-Spalte finden.")

    belege["betrag_raw"] = belege[beleg_amount_col].apply(normalize_amount)

    # Datum: bevorzugt Belegdatum/Rechnungsdatum
    beleg_date_col = find_column(
        belege,
        ["belegdatum", "rechnungsdatum", "rechnungsdat", "datum"],
        default=None
    )
    if beleg_date_col:
        belege["datum_norm"] = belege[beleg_date_col].apply(normalize_date)
        belege["datum_norm"] = pd.to_datetime(belege["datum_norm"], errors="coerce")
    else:
        belege["datum_norm"] = pd.NaT

    # Lieferanten- / Namensspalten
    supplier_cols = [
        c for c in belege.columns
        if any(k in c.lower() for k in ["lieferant", "name", "adressat", "empfänger", "kunde"])
    ]

    # Rechnungsnummer-Spalten
    invoice_cols = [
        c for c in belege.columns
        if any(k in c.lower() for k in ["rechnungsnummer", "rechnungs-nr", "belegfeld 1", "belegfeld1"])
    ]

    # zusammengesetzter Text je Beleg
    belege["supplier_text"] = belege.apply(
        lambda row: extract_supplier_text(row, supplier_cols, invoice_cols),
        axis=1
    )

    # Haupt-Rechnungsnummer
    beleg_invoice_col = invoice_cols[0] if invoice_cols else None
    if beleg_invoice_col:
        belege["invoice_number"] = belege[beleg_invoice_col]
    else:
        belege["invoice_number"] = ""

    # ------------------ Status erkennen (Gebucht / Posteingang) ------------------

    status_col = find_column(
        belege,
        ["status", "verarbeitungsstatus", "belegstatus", "buchungsstatus", "gebucht"],
        default=None
    )

    if (not status_col) and STATUS_SPALTE_MANUELL and STATUS_SPALTE_MANUELL in belege.columns:
        status_col = STATUS_SPALTE_MANUELL

    if status_col:
        status_raw = belege[status_col].astype(str)
        status_lower = status_raw.str.lower().str.strip()

        print("Status-Spalte erkannt:", status_col)
        print("Einzigartige Status-Werte:", sorted(status_lower.unique())[:20])

        belege["ist_gebucht"] = status_lower.eq("ja")

        eingangs_col = find_column(
            belege,
            ["eingangsdat", "eingangsdatum", "eingang"],
            default=None
        )

        if eingangs_col:
            eingangs_dt = pd.to_datetime(belege[eingangs_col], errors="coerce")
            belege["eingangsdatum_norm"] = eingangs_dt
            belege["ist_posteingang"] = ~belege["ist_gebucht"]
        else:
            belege["ist_posteingang"] = ~belege["ist_gebucht"]

    else:
        print("Keine Status-Spalte gefunden – setze alle Belege als 'gebucht'.")
        belege["ist_gebucht"] = True
        belege["ist_posteingang"] = False

    print("Belege gesamt:", len(belege))
    print("davon ist_gebucht:", int(belege["ist_gebucht"].sum()))
    print("davon ist_posteingang:", int(belege["ist_posteingang"].sum()))
    print("Nicht gebucht gesamt:", int((~belege["ist_gebucht"]).sum()))

    # ------------------ Matching ------------------

    sichere_matches = []
    # Für Variante B: unklare Fälle erst sammeln, später pro Beleg zusammenfassen
    unklare_map = {}  # beleg_index -> {"typ":..., "beleg_...":..., "kandidaten":[...]}

    verwendete_konto_indices = set()

    # 1) Gebuchte Belege – konservativ, aber Rechnungsnummer stark gewichtet
    gebuchte = belege[belege["ist_gebucht"] == True].copy()

    for _, beleg in gebuchte.iterrows():
        betrag = beleg["betrag_raw"]
        datum = beleg["datum_norm"]
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(betrag)

        candidates = konto[
            (konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)
        ].copy()

        if candidates.empty:
            continue

        diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
        candidates["datum_diff_tage"] = diff_days
        # Zahlungsziele etc. → großzügig
        candidates = candidates[diff_days <= 45]

        if candidates.empty:
            continue

        sup_txt = beleg["supplier_text"]
        inv_nr = beleg["invoice_number"]

        candidates["score"] = candidates["text_gesamt"].apply(
            lambda t: score_match(t, sup_txt, inv_nr)
        )

        candidates = candidates.sort_values(
            ["score", "datum_diff_tage"],
            ascending=[False, True]
        )

        if candidates.empty:
            continue

        best = candidates.iloc[0]

        # Wenn Rechnungsnummer getroffen wurde → Score >= 6 → sicher
        # oder: nur ein Kandidat mit Score >= 1
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
                    "beleg_datum": beleg["datum_norm"],
                    "beleg_betrag": beleg["betrag_raw"],
                    "beleg_supplier": beleg["supplier_text"],
                    "beleg_rechnungsnr": beleg["invoice_number"],
                })
        else:
            # alle Kandidaten sammeln, aber später pro Beleg zusammenfassen
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

    # 2) Posteingang-Belege – etwas mutiger, aber mit Plausibilitätscheck
    posteingang = belege[belege["ist_posteingang"] == True].copy()

    for _, beleg in posteingang.iterrows():
        betrag = beleg["betrag_raw"]
        datum = beleg["datum_norm"]
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(betrag)

        candidates = konto[
            (konto["betrag_raw"].abs().sub(betrag_abs).abs() <= BETRAG_TOLERANZ)
        ].copy()

        if candidates.empty:
            continue

        diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
        candidates["datum_diff_tage"] = diff_days
        candidates = candidates[diff_days <= DATUM_FENSTER_TAGE]

        if candidates.empty:
            continue

        sup_txt = beleg["supplier_text"]
        inv_nr = beleg["invoice_number"]

        candidates["score"] = candidates["text_gesamt"].apply(
            lambda t: score_match(t, sup_txt, inv_nr)
        )

        candidates = candidates.sort_values(
            ["score", "datum_diff_tage"],
            ascending=[False, True]
        )

        if candidates.empty:
            continue

        best = candidates.iloc[0]
        second_score = candidates.iloc[1]["score"] if len(candidates) > 1 else None

        # Wenn mehrere Kandidaten und der beste nur minimal besser → unklar
        if len(candidates) > 1 and (
            best["score"] <= 0 or
            (second_score is not None and (best["score"] - second_score) < 2)
        ):
            entry = unklare_map.setdefault(
                beleg.name,
                {
                    "typ": "posteingang",
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
            continue

        # ansonsten besten Treffer als sicher nehmen
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

    # ------------------ Unklare Fälle pro Beleg zusammenfassen ------------------

    unklare_faelle = []

    for beleg_idx, data in unklare_map.items():
        kandidaten = data["kandidaten"]
        if not kandidaten:
            continue
        # Bester Kandidat nach Score (und ggf. später nach Datum)
        best = max(kandidaten, key=lambda c: c["score"])
        konto_indices = sorted({k["konto_index"] for k in kandidaten})
        konto_indices_str = ",".join(str(i) for i in konto_indices)

        # Optional: Übersicht Score je Kontoindex
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
            "konto_indices": konto_indices_str,
            "best_konto_index": best["konto_index"],
            "best_konto_datum": best["konto_datum"],
            "best_konto_betrag": best["konto_betrag"],
            "best_konto_text": best["konto_text"],
            "best_score": best["score"],
            "konto_indices_scores": konto_scores_str,
        })

    # ------------------ Konto ohne Beleg / Kassenbuchungen ------------------

    alle_verwendeten_konto = {m["konto_index"] for m in sichere_matches}
    # Unklare Fälle blocken diese Kontozeilen auch – die sollen nicht als "ohne Beleg" erscheinen
    for data in unklare_map.values():
        for k in data["kandidaten"]:
            alle_verwendeten_konto.add(k["konto_index"])

    konto_ohne_beleg = konto[~konto["konto_index"].isin(alle_verwendeten_konto)].copy()

    konto_ohne_beleg["ist_kasse_vermutet"] = konto_ohne_beleg.apply(
        lambda row: looks_like_cash_booking(row["text_gesamt"], row["betrag_raw"]),
        axis=1
    )

    konto_ohne_beleg_final = konto_ohne_beleg.copy()

    # ------------------ Vorschlagsliste für Posteingang-Belege ------------------

    vorschlaege = []

    for _, beleg in posteingang.iterrows():
        betrag = beleg["betrag_raw"]
        datum = beleg["datum_norm"]
        if pd.isna(betrag) or pd.isna(datum):
            continue

        betrag_abs = abs(betrag)

        candidates = konto[
            (konto["betrag_raw"].abs().sub(betrag_abs).abs() <= (BETRAG_TOLERANZ * 2))
        ].copy()

        if candidates.empty:
            continue

        diff_days = (candidates["datum_norm"] - datum).dt.days.abs()
        candidates["datum_diff_tage"] = diff_days

        candidates = candidates[diff_days <= 60]  # Vorschläge etwas großzügiger

        if candidates.empty:
            continue

        sup_txt = beleg["supplier_text"]
        inv_nr = beleg["invoice_number"]

        candidates["score"] = candidates["text_gesamt"].apply(
            lambda t: score_match(t, sup_txt, inv_nr)
        )

        candidates = candidates.sort_values(
            ["score", "datum_diff_tage"],
            ascending=[False, True]
        ).head(3)

        for _, c in candidates.iterrows():
            vorschlaege.append({
                "beleg_index": beleg.name,
                "beleg_datum": beleg["datum_norm"],
                "beleg_betrag": beleg["betrag_raw"],
                "beleg_supplier": beleg["supplier_text"],
                "beleg_rechnungsnr": beleg["invoice_number"],
                "konto_index": c["konto_index"],
                "konto_datum": c["datum_norm"],
                "konto_betrag": c["betrag_raw"],
                "konto_text": c["text_gesamt"],
                "datum_diff_tage": c["datum_diff_tage"],
                "score": c["score"],
            })

    if vorschlaege:
        df_vorschlaege = pd.DataFrame(vorschlaege)
        df_vorschlaege.to_csv(
            OUTPUT_DIR / "posteingang_kandidaten.csv",
            sep=";",
            index=False,
            encoding="utf-8-sig"
        )
        print(f"Posteingang-Kandidaten gespeichert: {len(df_vorschlaege)} Zeilen.")
    else:
        print("Keine Posteingang-Kandidaten gefunden.")

    # ------------------ CSV-Ausgaben ------------------

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

    konto_ohne_beleg_final.to_csv(
        OUTPUT_DIR / "konto_ohne_beleg.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------ Zusammenfassung für Konsole ------------------

    anzahl_sicher = len(df_sicher)
    anzahl_sicher_gebucht = len(df_sicher[df_sicher["typ"] == "gebucht"]) if not df_sicher.empty else 0
    anzahl_sicher_post = len(df_sicher[df_sicher["typ"] == "posteingang"]) if not df_sicher.empty else 0

    anzahl_unklar = len(df_unklar)
    anzahl_fehlende = len(konto_ohne_beleg_final)
    anzahl_kasse = int(konto_ohne_beleg_final["ist_kasse_vermutet"].sum()) if not konto_ohne_beleg_final.empty else 0

    print("Analyse abgeschlossen")
    print(f"Sichere Matches gesamt: {anzahl_sicher}\n")
    print(f"davon gebuchte Belege: {anzahl_sicher_gebucht}")
    print(f"davon Posteingang-Rechnungen: {anzahl_sicher_post}")
    print(f"Unklare Fälle: {anzahl_unklar}\n")
    print(f"Fehlende Belege gesamt: {anzahl_fehlende}\n")
    print(f"davon Kassenbuchungen mit Beleg vermutlich im Posteingang: {anzahl_kasse}\n")
    print(f"Ergebnisse liegen im Ordner {OUTPUT_DIR} als CSV-Dateien.")

    return {
        "anzahl_sicher": anzahl_sicher,
        "anzahl_sicher_gebucht": anzahl_sicher_gebucht,
        "anzahl_sicher_post": anzahl_sicher_post,
        "anzahl_unklar": anzahl_unklar,
        "anzahl_fehlende": anzahl_fehlende,
        "anzahl_kasse": anzahl_kasse,
    }


# ============================================================
# FASTAPI – WEB-INTERFACE MIT DRAG & DROP + ZUSAMMENFASSUNG + ZIP-DOWNLOAD
# ============================================================

app = FastAPI()


# Static-Files (z.B. Logo) aus dem Projektordner bereitstellen
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <html>
      <head>
        <title>NEXTWAVE AI Buchhaltung</title>
        <style>
          body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 16px;
          }
          h1 {
            margin-bottom: 0.5rem;
          }
          .logo {
            max-width: 260px;
            height: auto;
            margin-bottom: 10px;
            display: block;
          }
          .hint {
            color: #555;
            margin-bottom: 1.5rem;
          }
          form {
            margin-top: 1.5rem;
          }
          .field {
            margin-bottom: 1.2rem;
          }
          .label {
            display: block;
            margin-bottom: 0.25rem;
            font-weight: 600;
          }
          .dropzone {
            border: 2px dashed #999;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background-color 0.2s;
          }
          .dropzone.hover {
            border-color: #2563eb;
            background-color: #eff6ff;
          }
          .dropzone small {
            display: block;
            color: #666;
          }
          .filename {
            font-weight: 600;
            margin-top: 6px;
          }
          .file-input {
            display: none;
          }
          button {
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 6px;
            border: none;
            background-color: #2563eb;
            color: white;
            cursor: pointer;
          }
          button:hover {
            background-color: #1d4ed8;
          }
          .footer {
            margin-top: 2rem;
            font-size: 0.85rem;
            color: #666;
          }
          .footer code {
            font-size: 0.8rem;
          }
          .legal {
            margin-top: 0.8rem;
            font-size: 0.75rem;
            color: #777;
            line-height: 1.4;
          }
          .progress {
            margin-top: 1rem;
            font-size: 0.9rem;
            color: #2563eb;
            display: none;
            align-items: center;
            gap: 8px;
          }
          .progress.active {
            display: inline-flex;
          }
          .spinner {
            width: 18px;
            height: 18px;
            border-radius: 999px;
            border: 3px solid #e5e7eb;
            border-top-color: #2563eb;
            animation: spin 0.8s linear infinite;
          }
          @keyframes spin {
            from { transform: rotate(0deg); }
            to   { transform: rotate(360deg); }
          }
        </style>
      </head>
      <body>
        <img src="/logo.png" alt="NEXTWAVE Logo" class="logo" />
        <h1>DATEV Kontoauszug / Belege Analyse</h1>
        <p class="hint">
          1. Exportiere in DATEV zwei CSV-Dateien (Kontoauszug &amp; Belege) und speichere sie z.B. in deinen OneDrive-Buchhaltungsordner.<br>
          2. Ziehe die Dateien per Drag &amp; Drop in die Felder unten ODER klicke auf die Drop-Zonen zum Auswählen.<br>
          3. Klicke auf "Analyse starten". Du siehst eine Zusammenfassung und kannst die Ergebnis-ZIP herunterladen.
        </p>

        <form id="uploadForm" action="/run" method="post" enctype="multipart/form-data">
          <div class="field">
            <span class="label">Kontoauszug CSV</span>
            <div id="konto_drop" class="dropzone">
              <div>CSV hierhin ziehen oder klicken</div>
              <small>Erwartet: Kontoauszug-Export aus DATEV</small>
              <div class="filename" id="konto_filename">Keine Datei ausgewählt</div>
            </div>
            <input class="file-input" type="file" name="konto_file" id="konto_file" accept=".csv" required />
          </div>

          <div class="field">
            <span class="label">Belege CSV</span>
            <div id="belege_drop" class="dropzone">
              <div>CSV hierhin ziehen oder klicken</div>
              <small>Erwartet: Beleg-Export aus DATEV (mit Spalte "Gebucht")</small>
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

        <div class="footer">
          Tipp: Speichere die spätere Ergebnis-ZIP in deinem Quartalsordner, z.B.
          <code>...\\0_Buchhaltung\\2025\\Q4-2025\\Ergebnisse</code>.
          <div class="legal">
            © NEXTWAVE GmbH – Alle Rechte vorbehalten.<br>
            Die Nutzung dieses Programms oder von Teilen daraus ohne vorherige schriftliche Zustimmung der NEXTWAVE GmbH
            ist untersagt und kann zivil- und strafrechtliche Schritte nach sich ziehen.
          </div>
        </div>

        <script>
          function setupDropzone(dropId, inputId, labelId) {
            const drop = document.getElementById(dropId);
            const input = document.getElementById(inputId);
            const label = document.getElementById(labelId);

            // Klicken auf Dropzone öffnet Dateidialog
            drop.addEventListener('click', function() {
              input.click();
            });

            // Dateiauswahl per Dialog
            input.addEventListener('change', function() {
              if (input.files && input.files.length > 0) {
                label.textContent = input.files[0].name;
              } else {
                label.textContent = "Keine Datei ausgewählt";
              }
            });

            // Drag & Drop Events
            ['dragenter', 'dragover'].forEach(eventName => {
              drop.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
                drop.classList.add('hover');
              }, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
              drop.addEventListener(eventName, function(e) {
                e.preventDefault();
                e.stopPropagation();
                drop.classList.remove('hover');
              }, false);
            });

            drop.addEventListener('drop', function(e) {
              const dt = e.dataTransfer;
              const files = dt.files;
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
    return html



@app.post("/run", response_class=HTMLResponse)
async def run(konto_file: UploadFile = File(...), belege_file: UploadFile = File(...)):
    """
    Nimmt die beiden hochgeladenen CSVs, speichert sie als
    uploaded_kontoauszug.csv / uploaded_belege.csv,
    führt die Analyse aus und baut eine ZIP im OUTPUT_DIR.
    Danach wird eine HTML-Zusammenfassung mit Download-Link angezeigt.
    """
    # Sicherstellen, dass der Ausgabeordner existiert
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Upload-Dateien in unser Projektverzeichnis schreiben
    with open(KONTOAUSZUG_CSV, "wb") as f:
        f.write(await konto_file.read())
    with open(BELEGE_CSV, "wb") as f:
        f.write(await belege_file.read())

    # Analyse ausführen
    res = run_analysis()

    # ZIP im OUTPUT_DIR bauen
    zip_path = OUTPUT_DIR / "datev_analyse_ergebnisse.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_file in OUTPUT_DIR.glob("*.csv"):
            zf.write(csv_file, arcname=csv_file.name)

    # HTML-Zusammenfassung + Download-Link zurückgeben
    html = f"""
    <html>
      <head>
        <title>Analyse abgeschlossen</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 16px;
          }}
          h1 {{
            margin-bottom: 0.5rem;
          }}
          ul {{
            line-height: 1.6;
          }}
          a.button {{
            display: inline-block;
            margin-top: 1.2rem;
            padding: 10px 18px;
            background-color: #2563eb;
            color: white;
            text-decoration: none;
            border-radius: 6px;
          }}
          a.button:hover {{
            background-color: #1d4ed8;
          }}
          .back {{
            margin-top: 1rem;
          }}
          .legal {{
            margin-top: 2rem;
            font-size: 0.75rem;
            color: #777;
            line-height: 1.4;
          }}
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

        <div class="back">
          <a href="/">Neue Analyse starten</a>
        </div>

        <div class="legal">
          © NEXTWAVE GmbH – Alle Rechte vorbehalten.<br>
          Die Nutzung dieses Programms oder von Teilen daraus ohne vorherige schriftliche Zustimmung der NEXTWAVE GmbH
          ist untersagt und kann zivil- und strafrechtliche Schritte nach sich ziehen.
        </div>
      </body>
    </html>
    """
    return html


@app.get("/download")
def download_zip():
    """
    Gibt die letzte erzeugte ZIP-Datei aus dem OUTPUT_DIR zurück.
    """
    zip_path = OUTPUT_DIR / "datev_analyse_ergebnisse.zip"
    if not zip_path.exists():
        return HTMLResponse(
            "<h1>Keine ZIP gefunden</h1><p>Bitte zuerst eine Analyse starten.</p>",
            status_code=404,
        )

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="datev_analyse_ergebnisse.zip",
    )

@app.get("/logo.png")
def logo():
    # Datei liegt im gleichen Ordner wie main.py
    return FileResponse(BASE_DIR / "nextwave_logo.png", media_type="image/png")


if __name__ == "__main__":
    # lokal starten
    uvicorn.run(app, host="127.0.0.1", port=8000)