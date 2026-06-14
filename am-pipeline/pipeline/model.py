"""
model.py
--------
Kapselt das trainierte AM-Modell als Klasse AMModel.

Verwendung:
    model = AMModel(model_path="models/spacy_output/model-best")
    model.load()
    doc = model.predict(text)
    for span in doc.spans:
        print(span.label, span.text, span.score)
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

TAP_LABELS = ["CLAIM", "DATA", "WARRANT", "REBUTTAL"]


@dataclass
class Span:
    """Ein einzelnes TAP-Element im Text."""
    start: int
    end:   int
    label: str
    text:  str
    score: float = 0.0

    def to_dict(self) -> dict:
        """Für JSON-Export / Streamlit-Kompatibilität."""
        return {
            "start": self.start, "end": self.end,
            "label": self.label, "text": self.text, "score": self.score,
        }


@dataclass
class ClassifiedText:
    """Ein Text mit allen erkannten TAP-Spans."""
    text:        str
    spans:       List[Span] = field(default_factory=list)
    source_file: str        = ""

    @property
    def span_count(self) -> int:
        return len(self.spans)

    def spans_by_label(self, label: str) -> List[Span]:
        return [s for s in self.spans if s.label == label]


class AMModel:
    """
    Kapselt das trainierte AM-Modell.

    Attribute:
        model_path : Pfad zu model-best
        labels     : Liste der TAP-Labels
        threshold  : Minimale Konfidenz für akzeptierte Spans
        is_loaded  : ob das Modell aktuell geladen ist
    """

    def __init__(self, 
                 model_path: Path, 
                 threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.labels     = TAP_LABELS
        self.threshold  = threshold
        self.is_loaded  = False
        self._nlp       = None 


    def load(self) -> "AMModel":
        """
        Lädt das spaCy-Modell von model_path.
        Wirft FileNotFoundError wenn noch nicht trainiert.
        Gibt self zurück für Method-Chaining.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modell nicht gefunden: {self.model_path}\n"
                "Bitte zuerst Training.run() ausführen."
            )

        import spacy
        print(f"Lade Modell: {self.model_path}")
        self._nlp = spacy.load(str(self.model_path))

        # Falls Labels nicht aus der Config initialisiert wurden
        spancat = self._nlp.get_pipe("spancat")
        if not spancat.labels:
            for label in self.labels:
                spancat.add_label(label)

        self.is_loaded = True
        return self

    def predict(self, text: str) -> ClassifiedText:
        """
        Führt Inferenz auf einem Text aus.
        Verarbeitet satzweise um lange OHI-Texte robust zu handhaben.

        Gibt ein AnnotatedDocument mit allen erkannten Spans zurück.
        """
        if not self.is_loaded:
            raise RuntimeError("Modell nicht geladen. Zuerst load() aufrufen.")

        sentences = self._split_sentences(text)
        spans: List[Span] = []

        for sent_text, sent_start in sentences:
            try:
                doc = self._nlp(sent_text)
            except Exception as e:
                print(f"[predict] Satz übersprungen: {e}")
                continue

            for span in doc.spans.get("sc", []):
                score = getattr(span._, "score", 1.0)
                if score < self.threshold:
                    continue
                spans.append(Span(
                    start = sent_start + span.start_char,
                    end   = sent_start + span.end_char,
                    label = span.label_,
                    text  = span.text,
                    score = round(score, 3),
                ))

        spans.sort(key=lambda s: s.start)
        return ClassifiedText(text=text, spans=spans)

    def _split_sentences(self, text: str) -> List[tuple]:
        """
        Teilt Text in Sätze mit Zeichenpositionen auf.
        Einfacher Regex-Splitter — vermeidet Abhängigkeit von
        de_core_news_sm und den tuple-index-Fehler beim Batch-Processing.
        """
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        raw_sentences = sentence_endings.split(text.strip())

        sentences, cursor = [], 0
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue
            start = text.find(sent, cursor)
            if start == -1:
                continue
            sentences.append((sent, start))
            cursor = start + len(sent)

        return sentences

