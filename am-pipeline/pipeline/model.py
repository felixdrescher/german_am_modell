"""
model.py
--------
Kapselt das trainierte AM-Modell als Klasse AMModel.
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

TAP_LABELS = ["CLAIM", "DATA", "WARRANT", "REBUTTAL"]

@dataclass
class Span:
    """Repräsentiert ein einzelnes TAP-Element im Text.

    Attributes:
        start (int): Inklusiver Startoffset im Originaltext.
        end (int): Exklusiver Endoffset im Originaltext.
        label (str): TAP-Kategorie, beispielsweise ``CLAIM`` oder ``DATA``.
        text (str): Textausschnitt innerhalb der angegebenen Offsets.
        score (float): Konfidenz der Modellvorhersage; ``0.0`` für Fallbacks.
    """
    start: int
    end:   int
    label: str
    text:  str
    score: float = 0.0

    def to_dict(self) -> dict:
        """Serialisiert den Span für JSON-Export und Streamlit.

        Returns:
            dict: Dictionary mit Start, Ende, Label, Text und Konfidenz.
        """
        return {
            "start": self.start, "end": self.end,
            "label": self.label, "text": self.text, "score": self.score,
        }


@dataclass
class ClassifiedText:
    """Speichert einen Text mit seinen erkannten TAP-Spans.

    Attributes:
        text (str): Vollständiger analysierter Text.
        spans (List[Span]): Alle erkannten oder zugeordneten TAP-Spans.
        source_file (str): Optionaler Ursprungspfad oder Dateiname des Texts.
        span_count (int): Berechnete Anzahl der enthaltenen Spans.
    """
    text:        str
    spans:       List[Span] = field(default_factory=list)
    source_file: str        = ""

    @property
    def span_count(self) -> int:
        """Gibt die Anzahl der erkannten Spans zurück.

        Returns:
            int: Anzahl der in ``spans`` enthaltenen Elemente.
        """
        return len(self.spans)

    def spans_by_label(self, label: str) -> List[Span]:
        """Filtert die Spans nach einem TAP-Label.

        Args:
            label (str): TAP-Label, nach dem gefiltert werden soll.

        Returns:
            List[Span]: Alle Spans mit dem angegebenen Label.
        """
        return [s for s in self.spans if s.label == label]


class AMModel:
    """
    Kapselt das trainierte Argumentation-Mining-Modell.

    Attributes:
        model_path (Path): Pfad zum Verzeichnis des trainierten spaCy-Modells.
        labels (List[str]): Unterstützte TAP-Labels.
        threshold (float): Minimale Konfidenz für akzeptierte Spans.
        is_loaded (bool): Gibt an, ob die spaCy-Pipeline geladen wurde.
        _nlp (Optional[Language]): Geladene spaCy-Pipeline oder ``None`` vor
            dem Laden des Modells.
    """

    def __init__(self, 
                 model_path: Path, 
                 threshold: float = 0.5):
        """Initialisiert das Argumentation-Mining-Modell.

        Args:
            model_path (Path): Pfad zum trainierten spaCy-Modell.
            threshold (float): Minimale Konfidenz akzeptierter Spans.
        """
        self.model_path = Path(model_path)
        self.labels     = TAP_LABELS
        self.threshold  = threshold
        self.is_loaded  = False
        self._nlp       = None 


    def load(self) -> "AMModel":
        """Lädt das spaCy-Modell vom konfigurierten Pfad.

        Returns:
            AMModel: Modellinstanz mit geladener spaCy-Pipeline.

        Raises:
            FileNotFoundError: Wenn der Modellpfad nicht existiert.
            OSError: Wenn spaCy die Modelldateien nicht lesen kann.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modell nicht gefunden unter: {self.model_path}\n"
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
        """Führt die Klassifikation von TAP-Elementen für einen Text aus.

        Args:
            text (str): Zu analysierender Text.

        Returns:
            ClassifiedText: Text mit allen erkannten Spans.

        Raises:
            RuntimeError: Wenn das Modell vor der Vorhersage nicht geladen wurde.
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
        """Teilt einen Text in Sätze mit ihren Zeichenpositionen.

        Args:
            text (str): Zu segmentierender Text.

        Returns:
            List[tuple]: Paare aus Satztext und Startoffset im Originaltext.
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

