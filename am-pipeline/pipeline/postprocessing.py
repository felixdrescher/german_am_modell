"""
postprocessing.py
-----------------
Filter- und Nachverarbeitungslogik für Argumentation-Mining-Spans.

Entfernt Rauschen, redundante/überlappende Spans (Non-Maximum Suppression),
zu kurze Fragmente und filtert Spans anhand konfigurierbarer Schwellenwerte.

Verwendung:
    from pipeline.postprocessing import SpanPostprocessor

    postprocessor = SpanPostprocessor(threshold=0.65, min_words=3)
    cleaned_spans = postprocessor.process(spans, full_text=text)
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


# Zeichen, die an den Rändern eines Spans entfernt werden sollen
STRIP_PUNCTUATION = ' \t\n\r"\'„“»«`.,;:!?()[]{}<>-–—'


def compute_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """Berechnet die Intersection-over-Union (IoU) zweier Zeichenbereiche.

    Args:
        start1: Startoffset des ersten Spans.
        end1: Endoffset des ersten Spans.
        start2: Startoffset des zweiten Spans.
        end2: Endoffset des zweiten Spans.

    Returns:
        float: IoU-Wert zwischen 0.0 (keine Überlappung) und 1.0 (identisch).
    """
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter_len = max(0, inter_end - inter_start)

    if inter_len == 0:
        return 0.0

    len1 = end1 - start1
    len2 = end2 - start2
    union_len = len1 + len2 - inter_len

    return inter_len / union_len if union_len > 0 else 0.0


def compute_containment(start1: int, end1: int, start2: int, end2: int) -> float:
    """Berechnet den Überdeckungsgrad des kleineren Spans im größeren.

    Args:
        start1: Startoffset des ersten Spans.
        end1: Endoffset des ersten Spans.
        start2: Startoffset des zweiten Spans.
        end2: Endoffset des zweiten Spans.

    Returns:
        float: Anteil des kleineren Spans, der im größeren liegt (0.0 bis 1.0).
    """
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter_len = max(0, inter_end - inter_start)

    min_len = min(end1 - start1, end2 - start2)
    return inter_len / min_len if min_len > 0 else 0.0


class SpanPostprocessor:
    """
    Nachverarbeitungskomponente zum Filtern und Bereinigen von Modell-Spans.

    Attributes:
        threshold (float): Minimaler Konfidenzwert für Spans (Standard: 0.65).
        class_thresholds (Optional[Dict[str, float]]): Optionale Schwellenwerte je TAP-Klasse.
        min_words (int): Minimale Wortanzahl pro Span (Standard: 3).
        min_chars (int): Minimale Zeichenlänge pro Span (Standard: 12).
        nms_iou_threshold (float): Max. erlaubte IoU für Non-Maximum Suppression (Standard: 0.3).
        containment_threshold (float): Max. erlaubter Überdeckungsanteil verschachtelter Spans (Standard: 0.6).
        clean_boundaries (bool): Ob Ränder von Sonderzeichen und Anführungszeichen befreit werden sollen.
        filter_duplicates (bool): Ob identische oder fast identische Spans entfernt werden sollen.
    """

    def __init__(
        self,
        threshold: float = 0.65,
        class_thresholds: Optional[Dict[str, float]] = None,
        min_words: int = 3,
        min_chars: int = 12,
        nms_iou_threshold: float = 0.3,
        containment_threshold: float = 0.6,
        clean_boundaries: bool = True,
        filter_duplicates: bool = True,
    ):
        """Initialisiert den Postprocessor mit Filterregeln.

        Args:
            threshold: Globaler Mindestscore (0.0 bis 1.0).
            class_thresholds: Spezifische Schwellenwerte pro TAP-Label.
            min_words: Mindestanzahl an Wörtern.
            min_chars: Mindestanzahl an Zeichen.
            nms_iou_threshold: IoU-Grenze für Überlappungsfilterung (NMS).
            containment_threshold: Überdeckungsanteil für Schachtelung.
            clean_boundaries: Bereinigung führender/nachfolgender Satzzeichen.
            filter_duplicates: Duplikatentfernung aktivieren.
        """
        self.threshold = threshold
        self.class_thresholds = class_thresholds or {}
        self.min_words = min_words
        self.min_chars = min_chars
        self.nms_iou_threshold = nms_iou_threshold
        self.containment_threshold = containment_threshold
        self.clean_boundaries = clean_boundaries
        self.filter_duplicates = filter_duplicates

    def get_threshold_for_label(self, label: str) -> float:
        """Gibt den effektiven Schwellenwert für ein Label zurück."""
        return self.class_thresholds.get(label, self.threshold)

    def clean_span(self, span: Any, full_text: str = "") -> Any:
        """Bereinigt die Zeichenränder eines Spans (z.B. führende Anführungszeichen).

        Args:
            span: Span-Objekt mit start, end, label, text, score.
            full_text: Vollständiger Quelltext für exakte Offset-Neuberechnung.

        Returns:
            Aktualisiertes Span-Objekt mit angepassten Offsets und bereinigtem Text.
        """
        if not self.clean_boundaries:
            return span

        raw_text = getattr(span, "text", "")
        start = getattr(span, "start", 0)
        end = getattr(span, "end", 0)

        cleaned_text = raw_text.strip(STRIP_PUNCTUATION)
        if not cleaned_text or cleaned_text == raw_text:
            return span

        # Start-Offset im ursprünglichen Text ermitteln
        leading_cut = raw_text.find(cleaned_text)
        if leading_cut != -1:
            new_start = start + leading_cut
            new_end = new_start + len(cleaned_text)
            
            if full_text and new_end <= len(full_text):
                cleaned_from_source = full_text[new_start:new_end]
            else:
                cleaned_from_source = cleaned_text

            # Werte auf Span übertragen
            span.start = new_start
            span.end = new_end
            span.text = cleaned_from_source

        return span

    def is_valid_span(self, span: Any) -> bool:
        """Prüft, ob ein Span die Qualitäts- und Mindestlängenkriterien erfüllt.

        Args:
            span: Zu prüfender Span.

        Returns:
            bool: True wenn der Span akzeptiert wird, sonst False.
        """
        label = getattr(span, "label", "")
        score = getattr(span, "score", 1.0)
        text = (getattr(span, "text", "") or "").strip()

        # 1. Konfidenz-Schwellenwert
        min_score = self.get_threshold_for_label(label)
        if score < min_score:
            return False

        # 2. Mindestlänge in Zeichen
        if len(text) < self.min_chars:
            return False

        # 3. Mindestanzahl an Wörtern
        words = [w for w in re.split(r"\s+", text) if w]
        if len(words) < self.min_words:
            return False

        # 4. Mindestens ein alphanumerisches Zeichen vorhanden
        if not re.search(r"\w", text):
            return False

        return True

    def apply_nms(self, spans: List[Any]) -> List[Any]:
        """Führt Non-Maximum Suppression (NMS) durch, um überlappende Spans zu bereinigen.

        Bei stark überlappenden Spans wird der Span mit dem höheren Score
        (oder bei gleichem Score der längere Span) bevorzugt.

        Args:
            spans: Liste vorsortierter oder ungeordneter Spans.

        Returns:
            Gefilterte Span-Liste ohne störende Überlappungen.
        """
        if not spans:
            return []

        # Nach Score absteigend, dann nach Textlänge absteigend sortieren
        sorted_spans = sorted(
            spans,
            key=lambda s: (getattr(s, "score", 0.0), len(getattr(s, "text", ""))),
            reverse=True,
        )

        kept: List[Any] = []

        for candidate in sorted_spans:
            c_start = getattr(candidate, "start", 0)
            c_end = getattr(candidate, "end", 0)

            suppress = False
            for k in kept:
                k_start = getattr(k, "start", 0)
                k_end = getattr(k, "end", 0)

                iou = compute_iou(c_start, c_end, k_start, k_end)
                containment = compute_containment(c_start, c_end, k_start, k_end)

                if iou >= self.nms_iou_threshold or containment >= self.containment_threshold:
                    suppress = True
                    break

            if not suppress:
                kept.append(candidate)

        # Nach Startoffset aufsteigend sortieren
        kept.sort(key=lambda s: getattr(s, "start", 0))
        return kept

    def process(self, spans: List[Any], full_text: str = "") -> List[Any]:
        """Führt den gesamten Postprocessing-Ablauf auf einer Liste von Spans aus.

        1. Bereinigung von Rändern und Sonderzeichen
        2. Filterung nach Mindestlänge und Schwellenwert (Threshold)
        3. Non-Maximum Suppression (NMS) zur Auflösung redundanter Überlappungen

        Args:
            spans: Rohspans aus der Modell-Inferenz.
            full_text: Vollständiger Originaltext.

        Returns:
            Bereinigte und gefilterte Liste von Spans.
        """
        if not spans:
            return []

        # 1. Ränder bereinigen
        if self.clean_boundaries:
            spans = [self.clean_span(s, full_text=full_text) for s in spans]

        # 2. Kriterienfilter (Score-Threshold, Mindestwörter, Mindestlänge)
        valid = [s for s in spans if self.is_valid_span(s)]

        # 3. NMS / Überlappungsbereinigung
        if self.filter_duplicates:
            valid = self.apply_nms(valid)

        return valid


def postprocess_spans(
    spans: List[Any],
    full_text: str = "",
    threshold: float = 0.65,
    min_words: int = 3,
    min_chars: int = 12,
    nms_iou_threshold: float = 0.3,
    containment_threshold: float = 0.6,
    clean_boundaries: bool = True,
) -> List[Any]:
    """Bequemlichkeitsfunktion zum Ausführen des Postprocessings.

    Args:
        spans: Liste der zu filternden Spans.
        full_text: Vollständiger Quelltext.
        threshold: Schwellenwert für Konfidenzwerte.
        min_words: Mindestwortanzahl.
        min_chars: Mindestzeichenanzahl.
        nms_iou_threshold: NMS IoU-Grenzwert.
        containment_threshold: Überdeckungsgrenzwert.
        clean_boundaries: Ob Ränder getrimmt werden sollen.

    Returns:
        Bereinigte Spans.
    """
    processor = SpanPostprocessor(
        threshold=threshold,
        min_words=min_words,
        min_chars=min_chars,
        nms_iou_threshold=nms_iou_threshold,
        containment_threshold=containment_threshold,
        clean_boundaries=clean_boundaries,
    )
    return processor.process(spans, full_text=full_text)
