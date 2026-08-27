"""
model.py
--------
Kapselt das trainierte AM-Modell als Klasse AMModel.
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from pipeline.postprocessing import SpanPostprocessor, postprocess_spans

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
    Kapselt das trainierte Argumentation-Mining-Modell samt Postprocessing.

    Attributes:
        model_path (Path): Pfad zum Verzeichnis des trainierten spaCy-Modells.
        labels (List[str]): Unterstützte TAP-Labels.
        threshold (float): Minimale Konfidenz für akzeptierte Spans (Standard: 0.65).
        postprocessor (SpanPostprocessor): Filter- und Nachverarbeitungskomponente.
        is_loaded (bool): Gibt an, ob die spaCy-Pipeline geladen wurde.
        _nlp (Optional[Language]): Geladene spaCy-Pipeline oder ``None`` vor
            dem Laden des Modells.
    """

    def __init__(
        self, 
        model_path: Path, 
        threshold: float = 0.65,
        min_words: int = 3,
        min_chars: int = 12,
        nms_iou_threshold: float = 0.3,
        containment_threshold: float = 0.6,
        clean_boundaries: bool = True,
    ):
        """Initialisiert das Argumentation-Mining-Modell.

        Args:
            model_path (Path): Pfad zum trainierten spaCy-Modell.
            threshold (float): Minimale Konfidenz akzeptierter Spans (Standard: 0.65).
            min_words (int): Minimale Wortanzahl pro Span (Standard: 3).
            min_chars (int): Minimale Zeichenanzahl pro Span (Standard: 12).
            nms_iou_threshold (float): NMS IoU-Grenzwert für Überlappungen.
            containment_threshold (float): Überdeckungsanteil für Schachtelung.
            clean_boundaries (bool): Ob Ränder bereinigt werden sollen.
        """
        self.model_path = Path(model_path)
        self.labels     = TAP_LABELS
        self.threshold  = threshold
        self.postprocessor = SpanPostprocessor(
            threshold=threshold,
            min_words=min_words,
            min_chars=min_chars,
            nms_iou_threshold=nms_iou_threshold,
            containment_threshold=containment_threshold,
            clean_boundaries=clean_boundaries,
        )
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

        # Synchronisiere internen spancat-Schwellenwert mit dem Modell-Schwellenwert
        if "threshold" in spancat.cfg:
            spancat.cfg["threshold"] = self.threshold

        self.is_loaded = True
        return self

    def predict(
        self, 
        text: str, 
        threshold: Optional[float] = None,
        min_words: Optional[int] = None,
    ) -> ClassifiedText:
        """Führt die Klassifikation und Nachverarbeitung von TAP-Elementen aus.

        Args:
            text (str): Zu analysierender Text.
            threshold (Optional[float]): Optionaler Überschreib-Schwellenwert.
            min_words (Optional[int]): Optionale Überschreib-Mindestwortanzahl.

        Returns:
            ClassifiedText: Text mit gefilterten und nachverarbeiteten Spans.

        Raises:
            RuntimeError: Wenn das Modell vor der Vorhersage nicht geladen wurde.
        """
        if not self.is_loaded:
            raise RuntimeError("Modell nicht geladen. Zuerst load() aufrufen.")

        eff_threshold = threshold if threshold is not None else self.threshold
        spancat = self._nlp.get_pipe("spancat")
        if "threshold" in spancat.cfg:
            spancat.cfg["threshold"] = eff_threshold

        sentences = self._split_sentences(text)
        raw_spans: List[Span] = []

        for sent_text, sent_start in sentences:
            try:
                doc = self._nlp(sent_text)
            except Exception as e:
                print(f"[predict] Satz übersprungen: {e}")
                continue

            for span in doc.spans.get("sc", []):
                score = getattr(span._, "score", 0.0)
                raw_spans.append(Span(
                    start = sent_start + span.start_char,
                    end   = sent_start + span.end_char,
                    label = span.label_,
                    text  = span.text,
                    score = round(score, 3) if score > 0 else round(eff_threshold, 3),
                ))

        # Postprocessing: Filtern nach Schwellenwert, Mindestlänge und NMS-Entfernung
        if threshold is not None or min_words is not None:
            postproc = SpanPostprocessor(
                threshold=eff_threshold,
                min_words=min_words if min_words is not None else self.postprocessor.min_words,
                min_chars=self.postprocessor.min_chars,
                nms_iou_threshold=self.postprocessor.nms_iou_threshold,
                containment_threshold=self.postprocessor.containment_threshold,
                clean_boundaries=self.postprocessor.clean_boundaries,
            )
            filtered_spans = postproc.process(raw_spans, full_text=text)
        else:
            filtered_spans = self.postprocessor.process(raw_spans, full_text=text)

        return ClassifiedText(text=text, spans=filtered_spans)

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

