"""
model.py
--------
Definiert die einstufige AM-Pipeline mit einem einzigen spaCy-Spancat-Modell.

    Eingabe: nur die Sätze aus Stufe 1
    Ausgabe: CLAIM / DATA / WARRANT / REBUTTAL Spans

GBERT (deepset/gbert-large) als Transformer-Backbone.
Das Modell wird lokal betrieben — kein Cloud-Zugriff.

Verwendung:
  # Training:       python pipeline/training.py
  # Inferenz:       from pipeline.model import predict
  # Streamlit-App:  wird von app/streamlit_app.py importiert
"""

from pathlib import Path
from typing import List, Dict, Optional
import spacy
from spacy.language import Language


# ── Konstanten ────────────────────────────────────────────────────────────────

LABELS = ["CLAIM", "DATA", "WARRANT", "REBUTTAL"]  # Stufe 2: alle TAP-Elemente

# Pfade (relativ zum Projektordner)
MODEL_DIR = Path("models/spacy_output")

# ── Modell laden ──────────────────────────────────────────────────────────────

def load_pipeline(
    model_path: Path = MODEL_DIR / "model-best",
) -> Language:
    """
    Lädt das trainierte einstufige AM-Modell.
    Wirft FileNotFoundError wenn noch nicht trainiert.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_path}\n"
            "Bitte zuerst python pipeline/training.py ausführen."
        )
    print(f"Lade Modell: {model_path}")
    return spacy.load(str(model_path))


# ── Inferenz ──────────────────────────────────────────────────────────────────

def predict(
    text:      str,
    nlp:       Language,
    threshold: float = 0.6,
) -> List[Dict]:
    """
    Einstufige Inferenz: erkennt alle TAP-Elemente direkt im Text.

    DistilBERT klassifiziert CLAIM / DATA / WARRANT / REBUTTAL

    Rückgabe: Liste von Span-Dicts:
      {"start": int, "end": int, "label": str, "text": str, "score": float}
    """
    doc = nlp(text)
    
    # 1. Alle validen Spans extrahieren
    raw_spans = []
    for span in doc.spans.get("sc", []):
        score = getattr(span._, "score", 1.0)
        if score >= threshold:
            raw_spans.append(span)
            
    # 2. Post-Processing: Label-aware NMS
    # Sortiere nach Score absteigend
    raw_spans.sort(key=lambda s: getattr(s._, "score", 1.0), reverse=True)
    
    final_spans = []
    for span in raw_spans:
        overlap = False
        for accepted in final_spans:
            # Check auf Überlappung
            if span.start < accepted.end and accepted.start < span.end:
                # NUR wenn die Labels gleich sind, löschen wir den schlechteren Span
                if span.label_ == accepted.label_:
                    overlap = True
                    break
                # Wenn die Labels unterschiedlich sind, lassen wir ihn drin 
                # (kein break, kein overlap = True für das Löschen)
        
        if not overlap:
            final_spans.append(span)
            
    # 3. Formatierung
    result = []
    for span in final_spans:
        result.append({
            "start": span.start_char,
            "end":   span.end_char,
            "label": span.label_,
            "text":  span.text,
            "score": round(getattr(span._, "score", 1.0), 3),
        })
        
    result.sort(key=lambda x: x["start"])
    return result




