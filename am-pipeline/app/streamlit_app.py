"""
streamlit_app.py
----------------
Prototypische Test- und Evaluierungsumgebung für das AM-Modell.
Implementierung mit Streamlit, spacy-streamlit und Bootstrap-ähnlichem Layout.

Starten mit CLI:
    streamlit run app/streamlit_app.py
"""

import json
import re
import sys
from html import escape
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import spacy_streamlit
import streamlit as st

# Pipeline-Imports ermöglichen (Projektstruktur: app/ liegt unter Projektroot)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.postprocessing import SpanPostprocessor, postprocess_spans


# ── Konfiguration ─────────────────────────────────────────────────────────────

TAP_LABELS = ["CLAIM", "DATA", "WARRANT", "REBUTTAL"]

TAP_COLORS = {
    "CLAIM": "#4A90D9",
    "DATA": "#27AE60",
    "WARRANT": "#E67E22",
    "REBUTTAL": "#E74C3C",
}

TAP_DESCRIPTIONS = {
    "CLAIM": "Behauptung - die zentrale These oder Position",
    "DATA": "Daten - stützende Fakten oder Belege",
    "WARRANT": "Warrant - Begründung, warum die Daten die Behauptung stützen",
    "REBUTTAL": "Rebuttal - Einschränkung oder Gegenargument",
}

# Pfad zum trainierten Modell relativ zum Projektordner
MODEL_PATH = Path(__file__).parent.parent / "models" / "spacy_output" / "model-best"


# ── Datenklassen ──────────────────────────────────────────────────────────────

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
        """Serialisiert den Span in ein Dictionary.

        Returns:
            dict: Span-Attribute, die sich für die JSON-Serialisierung eignen.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        """Erstellt einen Span aus serialisierten Annotationsdaten.

        Args:
            d (dict): Zuordnung mit ``start``, ``end`` und ``label``;
                ``text`` und ``score`` sind optional.

        Returns:
            Span: Der rekonstruierte Span.

        Raises:
            KeyError: Wenn ein erforderliches Span-Feld in ``d`` fehlt.
        """
        return cls(
            start=d["start"], end=d["end"],
            label=d["label"], text=d.get("text", ""),
            score=d.get("score", 0.0),
        )


@dataclass
class TextFile:
    """Repräsentiert eine geladene Textdatei samt Metadaten.

    Attributes:
        filename (str): Name der Datei ohne Verzeichnispfad.
        filepath (Path): Absoluter oder relativer Pfad der geladenen Datei.
        content (str): Vollständiger dekodierter Dateiinhalt.
        encoding (str): Beim Lesen verwendete Zeichenkodierung.
    """
    filename: str
    filepath: Path
    content:  str
    encoding: str = "utf-8"

    @property
    def line_count(self) -> int:
        """Gibt die Anzahl der Zeilen im Dateiinhalt zurück.

        Returns:
            int: Anzahl der Zeilen bzw. null bei leerem Inhalt.
        """
        return len(self.content.splitlines())

    @property
    def char_count(self) -> int:
        """Gibt die Anzahl der Zeichen im Dateiinhalt zurück.

        Returns:
            int: Zeichenanzahl von ``content``.
        """
        return len(self.content)


@dataclass
class AnnotationExport:
    """Bereitet Annotationen für den JSON-Download auf.

    Attributes:
        text (str): Vollständiger Text, auf den sich die Spans beziehen.
        spans (List[Span]): Zu exportierende Annotationen mit Zeichenoffsets.
    """
    text:  str
    spans: List[Span]

    def to_json(self) -> str:
        """Serialisiert den Annotationsexport als formatiertes JSON.

        Returns:
            str: UTF-8-sicheres JSON mit Text, Spans und Exportmetadaten.

        Raises:
            TypeError: Wenn ein Span einen nicht JSON-kodierbaren Wert enthält.
        """
        return json.dumps({
            "text":  self.text,
            "spans": [s.to_dict() for s in self.spans],
            "meta": {
                "timestamp":  datetime.now().isoformat(),
                "source":     "streamlit_eval",
                "span_count": len(self.spans),
            },
        }, ensure_ascii=False, indent=2)


# ── FileScanner ───────────────────────────────────────────────────────────────

class FileScanner:
    """Findet und lädt unterstützte Textdateien aus einem Verzeichnis.

    Attributes:
        directory (Path): Zu durchsuchendes Verzeichnis.
        SUPPORTED (set[str]): Unterstützte Dateiendungen für Textdateien.
    """

    SUPPORTED = {".txt", ".md", ".text"}

    def __init__(self, directory: str):
        """Initialisiert den Scanner für ein Verzeichnis.

        Args:
            directory (str): Pfad des zu durchsuchenden Verzeichnisses.
        """
        self.directory = Path(directory)

    def scan(self) -> Dict[str, Path]:
        """Sucht unterstützte Textdateien im konfigurierten Verzeichnis.

        Returns:
            Dict[str, Path]: Zuordnung von Dateinamen zu Pfaden oder eine leere
                Zuordnung, wenn das Verzeichnis ungültig oder nicht zugreifbar ist.
        """
        result = {}
        try:
            if self.directory.is_dir():
                for f in sorted(self.directory.iterdir()):
                    if f.is_file() and f.suffix.lower() in self.SUPPORTED:
                        result[f.name] = f
        except PermissionError:
            pass
        return result

    def load(self, path: Path) -> TextFile:
        """Lädt eine Textdatei und versucht zuerst UTF-8, dann Latin-1.

        Args:
            path (Path): Pfad zur unterstützten Textdatei.

        Returns:
            TextFile: Dateimetadaten und dekodierter Inhalt.

        Raises:
            FileNotFoundError: Wenn ``path`` nicht existiert.
            PermissionError: Wenn ``path`` nicht gelesen werden kann.
            OSError: Wenn beim Lesen ein anderer Betriebssystemfehler auftritt.
        """
        for encoding in ("utf-8", "latin-1"):
            try:
                return TextFile(
                    filename=path.name,
                    filepath=path,
                    content=path.read_text(encoding=encoding),
                    encoding=encoding,
                )
            
            except UnicodeDecodeError:
                continue

        return TextFile(
            filename=path.name, filepath=path,
            content=path.read_text(encoding="utf-8", errors="replace"),
        )


# ── AMModelAdapter ────────────────────────────────────────────────────────────

class AMModelAdapter:
    """
    Kapselt das trainierte spaCy-Modell mit konfigurierbarem Postprocessing.

    Attributes:
        model_path (Path): Verzeichnis des trainierten spaCy-Modells.
        threshold (float): Minimale Konfidenz für übernommene Modell-Spans (Standard: 0.65).
        is_loaded (bool): Gibt an, ob das trainierte Modell geladen wurde.
        _nlp (Optional[Language]): Geladene spaCy-Pipeline oder ``None`` im
            Fallback-Modus.
    """

    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = 0.65):
        """Initialisiert einen Adapter für ein trainiertes spaCy-Modell.

        Args:
            model_path (Path): Verzeichnis mit dem trainierten Modell.
            threshold (float): Minimale Konfidenz für übernommene Vorhersagen (Standard: 0.65).
        """
        self.model_path = Path(model_path)
        self.threshold  = threshold
        self.is_loaded  = False
        self._nlp       = None   # spaCy Language-Objekt nach load()

    def load(self) -> "AMModelAdapter":
        """Lädt das spaCy-Modell und konfiguriert dessen Span-Labels.

        Returns:
            AMModelAdapter: Dieser Adapter für Modell- oder Fallback-Inferenz.

        Raises:
            RuntimeError: Kann beim Zugriff auf den Streamlit-Sitzungszustand auftreten.
        """
        if not self.model_path.exists():
            return self  

        try:
            import spacy
            self._nlp = spacy.load(str(self.model_path))

            spancat = self._nlp.get_pipe("spancat")
            existing = set(spancat.labels)
            
            for label in TAP_LABELS:
                if label not in existing:
                    spancat.add_label(label)

            if "threshold" in spancat.cfg:
                spancat.cfg["threshold"] = self.threshold

            self.is_loaded = True

        except Exception as e:
            # Meldung in der sidebar anzeigen
            st.session_state["model_load_error"] = str(e)

        return self

    def predict(
        self, 
        text: str, 
        threshold: Optional[float] = None,
        min_words: int = 3,
        nms: bool = True,
        clean_boundaries: bool = True,
        **kwargs,
    ) -> List[Span]:
        """Extrahiert und filtert Argumentations-Spans aus einem Text.

        Args:
            text (str): Zu analysierender Ausgangstext.
            threshold (Optional[float]): Minimaler Schwellenwert (überschreibt self.threshold).
            min_words (int): Minimale Wortanzahl je Span.
            nms (bool): Non-Maximum Suppression zur Entfernung von Überlappungen aktivieren.
            clean_boundaries (bool): Ränder von Satzzeichen und Anführungszeichen befreien.
            **kwargs: Zusätzliche Filter- oder Überschreib-Argumente.

        Returns:
            List[Span]: Gefilterte und bereinigte Spans.
        """
        if threshold is None and "threshold" in kwargs:
            threshold = kwargs["threshold"]

        if self.is_loaded and self._nlp is not None:
            return self._predict_model(
                text, 
                threshold=threshold,
                min_words=min_words,
                nms=nms,
                clean_boundaries=clean_boundaries,
                **kwargs,
            )
        return []

    def _predict_model(
        self, 
        text: str,
        threshold: Optional[float] = None,
        min_words: int = 3,
        nms: bool = True,
        clean_boundaries: bool = True,
        **kwargs,
    ) -> List[Span]:
        """Führt satzweise Inferenz und anschließendes Postprocessing aus.

        Args:
            text (str): Zu analysierender Ausgangstext.
            threshold (Optional[float]): Mindest-Konfidenz (Standard: self.threshold).
            min_words (int): Mindest-Wortanzahl.
            nms (bool): NMS aktivieren.
            clean_boundaries (bool): Ränder bereinigen.
            **kwargs: Zusätzliche Argumente.

        Returns:
            List[Span]: Sortierte und gefilterte Modellvorhersagen.
        """
        eff_threshold = threshold if threshold is not None else self.threshold

        spancat = self._nlp.get_pipe("spancat")
        if "threshold" in spancat.cfg:
            spancat.cfg["threshold"] = eff_threshold

        sentence_re = re.compile(r'(?<=[.!?])\s+')
        raw_sents   = sentence_re.split(text.strip())

        sents, cursor = [], 0
        for s in raw_sents:
            s = s.strip()
            if not s:
                continue

            pos = text.find(s, cursor)
            if pos != -1:
                sents.append((s, pos))
                cursor = pos + len(s)

        raw_spans = []
        for sent_text, sent_start in sents:
            try:
                doc = self._nlp(sent_text)
            except Exception:
                continue

            for sp in doc.spans.get("sc", []):
                score = getattr(sp._, "score", 0.0)
                raw_spans.append(Span(
                    start = sent_start + sp.start_char,
                    end   = sent_start + sp.end_char,
                    label = sp.label_,
                    text  = sp.text,
                    score = round(score, 3) if score > 0 else round(eff_threshold, 3),
                ))

        # Postprocessing anwenden
        postproc = SpanPostprocessor(
            threshold=eff_threshold,
            min_words=min_words,
            filter_duplicates=nms,
            clean_boundaries=clean_boundaries,
        )
        filtered_spans = postproc.process(raw_spans, full_text=text)
        return filtered_spans


# ── AppState ──────────────────────────────────────────────────────────────────

class AppState:
    """
    Typisierter Wrapper um ``st.session_state`` für den UI-Zustand.

    Attributes:
        spans (List[Span]): Aktuelle Annotations-Spans der Sitzung.
        text (str): Aktuell analysierter Text.
        file (str): Pfad der aktuell analysierten Datei.
        analyzed (bool): Gibt an, ob für den aktuellen Text Ergebnisse vorliegen.
        active_span_idx (int): Index des im Editor ausgewählten Spans im
            Streamlit-Sitzungszustand.
        model_load_error (str): Fehlermeldung des letzten Modellladeversuchs im
            Streamlit-Sitzungszustand.
        _DEFAULTS (Dict[str, object]): Standardwerte, mit denen fehlende
            Sitzungseinträge initialisiert werden.
    """

    _DEFAULTS = {
        "current_spans": [],
        "current_text": "",
        "current_file": "",
        "analyzed": False,
        "active_span_idx": 0,
        "model_load_error": "",
    }

    def __init__(self):
        """Initialisiert fehlende Werte im Streamlit-Sitzungszustand.

        Raises:
            RuntimeError: Wenn die Methode außerhalb eines Streamlit-Kontexts läuft.
        """
        for key, default in self._DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = default

    # Spans
    @property
    def spans(self) -> List[Span]:
        """Gibt die aktuell gespeicherten Annotations-Spans zurück.

        Returns:
            List[Span]: Veränderbare Liste der Spans in der aktiven Sitzung.
        """
        return st.session_state.current_spans

    @spans.setter
    def spans(self, value: List[Span]) -> None:
        """Speichert Annotations-Spans in der aktiven Sitzung.

        Args:
            value (List[Span]): Ersatzliste der Annotations-Spans.
        """
        st.session_state.current_spans = value

    # Text
    @property
    def text(self) -> str:
        """Gibt den aktuell analysierten Ausgangstext zurück.

        Returns:
            str: In der aktiven Sitzung gespeicherter Text.
        """
        return st.session_state.current_text

    @text.setter
    def text(self, value: str) -> None:
        """Speichert Ausgangstext in der aktiven Sitzung.

        Args:
            value (str): Text der aktuellen Analyse.
        """
        st.session_state.current_text = value

    # Datei-Pfad
    @property
    def file(self) -> str:
        """Gibt den Pfad der aktuell analysierten Datei zurück.

        Returns:
            str: Gespeicherter Dateipfad bzw. leerer String, wenn nicht gesetzt.
        """
        return st.session_state.current_file

    @file.setter
    def file(self, value: str) -> None:
        """Speichert den Pfad der aktuell analysierten Datei.

        Args:
            value (str): Dateipfad der aktuellen Analyse.
        """
        st.session_state.current_file = value

    # Analyse-Flag
    @property
    def analyzed(self) -> bool:
        """Gibt zurück, ob die aktuelle Sitzung Analyseergebnisse enthält.

        Returns:
            bool: ``True``, wenn die Analyse abgeschlossen ist.
        """
        return st.session_state.analyzed

    @analyzed.setter
    def analyzed(self, value: bool) -> None:
        """Setzt das Kennzeichen für die abgeschlossene Analyse.

        Args:
            value (bool): Ob die Sitzung aktuelle Ergebnisse enthält.
        """
        st.session_state.analyzed = value

    def reset(self) -> None:
        """Leert Analysedaten, ohne die Dateiauswahl-Widgets zurückzusetzen.

        Raises:
            RuntimeError: If called outside an active Streamlit context.
        """
        self.spans    = []
        self.text     = ""
        self.file     = ""
        self.analyzed = False
        st.session_state.active_span_idx = 0


# ── Modell (gecached, einmalig geladen) ───────────────────────────────────────

@st.cache_resource(show_spinner="Lade AM-Modell...")
def get_model() -> AMModelAdapter:
    """Gibt den zwischengespeicherten Argumentation-Mining-Modelladapter zurück.

    Returns:
        AMModelAdapter: Geladener Adapter oder ein Adapter im Fallback-Modus.

    Raises:
        RuntimeError: Wenn der Ressourcen-Cache von Streamlit nicht verfügbar ist.
    """
    return AMModelAdapter(MODEL_PATH).load()


# ── Visualisierung ────────────────────────────────────────────────────────────

def render_displacy(text: str, spans: List[Span]) -> None:
    """Visualisiert vorhergesagte Spans mit spaCys displaCy-Komponente.

    Args:
        text (str): Vollständiger Text für die Zeichenoffsets.
        spans (List[Span]): Zu visualisierende Annotations-Spans.

    Raises:
        AttributeError: Wenn der geladene Adapter keine spaCy-Pipeline enthält.
        ValueError: Wenn ein Span-Offset für ``text`` ungültig ist.
    """
    nlp = get_model()._nlp
    doc = nlp.make_doc(text)
    
    span_objs = []
    for s in spans:
        span = doc.char_span(s.start, s.end, label=s.label, alignment_mode="contract")
        if span:
            span_objs.append(span)
    
    doc.spans["sc"] = span_objs

    # Visualisierung mit spacy-streamlit (nutzt displaCy)
    spacy_streamlit.visualize_spans(
        doc,
        title=None,
        spans_key="sc",
        displacy_options={"colors": TAP_COLORS},
        show_table=False
    )


def render_span_preview(
    text: str,
    start: int,
    end: int,
    color: str,
    context_chars: int,
    max_height: Optional[int] = None,
) -> None:
    """Rendert einen hervorgehobenen Kontext für einen Span.

    Args:
        text (str): Vollständiger Text, auf den sich die Offsets beziehen.
        start (int): Inklusiver Startoffset des hervorgehobenen Bereichs.
        end (int): Exklusiver Endoffset des hervorgehobenen Bereichs.
        color (str): CSS-Farbe für Hervorhebung und Rahmen.
        context_chars (int): Anzahl der zusätzlich angezeigten Zeichen je Seite.
        max_height (Optional[int]): Maximale Fensterhöhe in Pixeln. Bei ``None``
            wird keine vertikale Begrenzung gesetzt.
    """
    context_start = max(0, start - context_chars)
    context_end = min(len(text), end + context_chars)
    before = escape(text[context_start:start]).replace("\r", " ").replace("\n", " ")
    selected_text = escape(text[start:end]).replace("\r", " ").replace("\n", " ")
    after = escape(text[end:context_end]).replace("\r", " ").replace("\n", " ")
    height_style = ""
    if max_height is not None:
        height_style = f"max-height:{max_height}px;overflow-y:auto;"

    st.markdown(
        f"<div style='font-size:0.95rem;color:#333;margin-top:8px;"
        f"font-family:Georgia,serif;padding:12px 14px;background:#fafafa;"
        f"border:1px solid {color}66;border-radius:6px;line-height:1.7;"
        f"white-space:normal;{height_style}'>"
        f"{before}<mark style='background:{color}33;border-left:3px solid {color};"
        f"border-right:3px solid {color};padding:1px 2px'>{selected_text}</mark>{after}"
        f"</div>"
        f"<div style='font-size:0.8rem;color:#666;margin-top:4px'>"
        f"Vorschau: Zeichen {start}–{end}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_exact_offset_editor(
    text: str,
    start: int,
    end: int,
    key_prefix: str,
    submit_label: str,
) -> Optional[tuple[int, int]]:
    """Rendert die erweiterte Eingabe exakter Zeichenoffsets.

    Args:
        text (str): Vollständiger Text, der die zulässigen Offsets bestimmt.
        start (int): Vorbelegter Startoffset.
        end (int): Vorbelegter Endoffset.
        key_prefix (str): Eindeutiger Präfix für die Streamlit-Widget-Schlüssel.
        submit_label (str): Beschriftung der Schaltfläche zum Übernehmen.

    Returns:
        Optional[tuple[int, int]]: Gültige Start- und Endoffsets nach einem
            Klick auf die Schaltfläche, sonst ``None``.
    """
    with st.expander("Erweiterte Bearbeitung: exakte Zeichenoffsets"):
        advanced_start, advanced_end = st.columns(2)

        with advanced_start:
            exact_start = st.number_input(
                "Exakter Startoffset",
                min_value=0,
                max_value=max(len(text) - 1, 0),
                value=min(start, max(len(text) - 1, 0)),
                key=f"{key_prefix}_start",
            )

        with advanced_end:
            exact_end = st.number_input(
                "Exakter Endoffset",
                min_value=1,
                max_value=len(text),
                value=max(1, min(end, len(text))),
                key=f"{key_prefix}_end",
            )

        if st.button(submit_label, key=f"{key_prefix}_apply"):
            if int(exact_start) < int(exact_end):
                return int(exact_start), int(exact_end)
            st.error("Start muss kleiner als Ende sein.")

    return None


def render_annotation_editor(text: str, spans: List[Span]) -> List[Span]:
    """Rendert Steuerelemente zum Bearbeiten und Löschen vorhandener Spans.

    Args:
        text (str): Vollständiger Text für die Span-Grenzen.
        spans (List[Span]): Veränderbare Liste der zu bearbeitenden Spans.

    Returns:
        List[Span]: Aktualisierte und nach Startoffset sortierte Span-Liste.

    Raises:
        RuntimeError: Wenn die Funktion außerhalb eines Streamlit-Kontexts läuft.
    """
    st.markdown("### Spans bearbeiten")
    st.markdown(f"**Span auswählen** ({len(spans)} erkannt)")

    if spans:
        options = [_option_label(i, s) for i, s in enumerate(spans)]

        st.session_state.active_span_idx = min(
            st.session_state.active_span_idx, len(spans) - 1
        )

        # Auswahlbox zu bearbeitende Span
        selected = st.selectbox(
            "Span:", options,
            index=st.session_state.active_span_idx,
            key="span_selector",
            label_visibility="collapsed",
        )
        
        idx = options.index(selected)
        st.session_state.active_span_idx = idx
        span = spans[idx]
        color = TAP_COLORS.get(span.label, "#999")

        col_lbl, col_del = st.columns([5, 2])

        # Auswahlbox zum Span-Label ändern        
        with col_lbl:
            st.caption(
                "Ändere das Span-Label"
            )
            new_lbl = st.selectbox(
                "Label ändern", TAP_LABELS,
                index=TAP_LABELS.index(span.label),
                key=f"edit_label_{idx}_{span.start}",
                label_visibility="collapsed",
            )

            if new_lbl != span.label:
                spans[idx].label = new_lbl
                st.rerun()

        # Auswahlbox zum Span löschen
        with col_del:
            st.caption(
                "Oder lösche die ausgewählte Span"
            )
            if st.button("Span löschen", key="edit_del", help="Span löschen"):
                spans.pop(idx)
                st.session_state.active_span_idx = max(0, idx - 1)
                st.rerun()

        st.caption(
            "Lege die neue Auswahl über die Grenzen fest und gleiche sie mit dem Preview ab."
        )

        # Anpassen der Slider Range für nachfolgende Slider
        with st.expander("Slider-Reichweite anpassen"):
            slider_range = st.select_slider(
                "Slider-Reichweite (Zeichen)",
                options=list(range(50, 501, 50)),
                value=100,
                key=f"edit_slider_range_{idx}_{span.start}_{span.end}",
            )

        col_s, col_e = st.columns(2)

        # Slider zum Verschieben der Start- und Endoffsets 
        with col_s:
            start_delta = st.slider(
                "Start verschieben (Zeichen)",
                min_value=-slider_range,
                max_value=slider_range,
                value=0,
                key=f"start_delta_{idx}_{span.start}_{span.end}",
            )

        with col_e:
            end_delta = st.slider(
                "Ende verschieben (Zeichen)",
                min_value=-slider_range,
                max_value=slider_range,
                value=0,
                key=f"end_delta_{idx}_{span.start}_{span.end}",
            )

        proposed_start = min(max(0, span.start + start_delta), len(text))
        proposed_end = min(max(0, span.end + end_delta), len(text))
        boundaries_valid = proposed_start < proposed_end

        context_start = max(0, proposed_start - 150)
        context_end = min(len(text), proposed_end + 150)

        before = (
            escape(text[context_start:proposed_start])
            .replace("\r", " ")
            .replace("\n", " ")
        )

        selected_text = (
            escape(text[proposed_start:proposed_end])
            .replace("\r", " ")
            .replace("\n", " ")
        )

        after = (
            escape(text[proposed_end:context_end])
            .replace("\r", " ")
            .replace("\n", " ")
        )

        # Markdown-Vorschau des hervorgehobenen Textes mit CSS-Hervorhebung
        st.markdown(
            f"<div style='font-size:0.95rem;color:#333;margin-top:8px;"
            f"font-family:Georgia,serif;padding:12px 14px;background:#fafafa;"
            f"border:1px solid {color}66;border-radius:6px;line-height:1.7;"
            f"white-space:normal'>"
            f"{before}<mark style='background:{color}33;border-left:3px solid {color};"
            f"border-right:3px solid {color};padding:1px 2px'>{selected_text}</mark>{after}"
            f"</div>"
            f"<div style='font-size:0.8rem;color:#666;margin-top:4px'>"
            f"Vorschau: Zeichen {proposed_start}–{proposed_end}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if not boundaries_valid:
            st.error("Start muss kleiner als Ende sein.")

        apply_col, reset_col = st.columns(2)

        # Button zum Übernehmen der relativen Offsets
        with apply_col:
            if st.button(
                "Grenzen übernehmen",
                key=f"apply_relative_{idx}_{span.start}_{span.end}",
                use_container_width=True, type= "primary",
                disabled=not boundaries_valid,
            ):
                spans[idx].start = proposed_start
                spans[idx].end = proposed_end
                spans[idx].text = text[proposed_start:proposed_end]
                spans.sort(key=lambda s: s.start)
                st.rerun()

        # Button zum Zurücksetzen der relativen Offsets
        with reset_col:
            st.button(
                "Zurücksetzen",
                key=f"reset_relative_{idx}_{span.start}_{span.end}",
                use_container_width=True,
                on_click=lambda: _reset_editor_controls(idx, span),
            )
    
        # Erwerterte Bearbeitung innerhalb eines Expanders
        with st.expander("Erweiterte Bearbeitung: exakte Zeichenoffsets"):
            advanced_start, advanced_end = st.columns(2)

            with advanced_start:
                exact_start = st.number_input(
                    "Exakter Startoffset",
                    min_value=0,
                    max_value=max(len(text) - 1, 0),
                    value=proposed_start,
                    key=(
                        f"exact_start_{idx}_{span.start}_{span.end}_"
                        f"{start_delta}_{end_delta}"
                    ),
                )

            with advanced_end:
                exact_end = st.number_input(
                    "Exakter Endoffset",
                    min_value=1,
                    max_value=len(text),
                    value=proposed_end,
                    key=(
                        f"exact_end_{idx}_{span.start}_{span.end}_"
                        f"{start_delta}_{end_delta}"
                    ),
                )

            if st.button(
                "Exakte Offsets übernehmen",
                key=f"apply_exact_{idx}_{span.start}_{span.end}",
            ):
                if int(exact_start) < int(exact_end):
                    spans[idx].start = int(exact_start)
                    spans[idx].end = int(exact_end)
                    spans[idx].text = text[int(exact_start):int(exact_end)]
                    spans.sort(key=lambda s: s.start)
                    st.rerun()
                else:
                    st.error("Start muss kleiner als Ende sein.")
    else:
        st.info("Noch keine Spans vorhanden.")

    return spans

def _reset_editor_controls(idx: int, span: Span) -> None:
    """Setzt die temporären Steuerelemente des aktuellen Spans zurück.

    Alle relativen Verschiebungen und exakten Offset-Eingaben werden
    entfernt, sodass die Vorschau wieder die gespeicherten Grenzen zeigt.
    """
    key_prefixes = (
        f"exact_start_{idx}_{span.start}_{span.end}_",
        f"exact_end_{idx}_{span.start}_{span.end}_",
    )

    keys_to_reset = {
        f"edit_slider_range_{idx}_{span.start}_{span.end}",
        f"start_delta_{idx}_{span.start}_{span.end}",
        f"end_delta_{idx}_{span.start}_{span.end}",
    }

    keys_to_reset.update(
        key for key in st.session_state
        if key.startswith(key_prefixes)
    )

    for widget_key in keys_to_reset:
        st.session_state.pop(widget_key, None)

def render_new_span_editor(text: str, spans: List[Span]) -> List[Span]:
    """Rendert Steuerelemente zum Hinzufügen eines neuen Spans.

    Args:
        text (str): Vollständiger Text für die Span-Grenzen.
        spans (List[Span]): Veränderbare Liste, der ein Span hinzugefügt wird.

    Returns:
        List[Span]: Aktualisierte und nach Startoffset sortierte Span-Liste.

    Raises:
        RuntimeError: Wenn die Funktion außerhalb eines Streamlit-Kontexts läuft.
    """
    st.markdown("### Neue Span hinzufügen")
    st.caption(f"Dokumentlänge: **{len(text)} Zeichen**")

    new_label = st.selectbox("Span-Label wählen", TAP_LABELS, key="new_label")

    st.caption(
        "Lege die neue Auswahl über die Grenzen fest und gleiche sie mit dem Preview ab."
    )

    # Slider zum anpassen der Slieder-RAnge fuer die End- und Startoffset-Slider
    with st.expander("Slider-Reichweite anpassen"):
        new_slider_range = st.select_slider(
            "Slider-Reichweite für neue Span (Zeichen)",
            options=list(range(50, 501, 50)),
            value=100,
            key="new_slider_range",
        )

    # Buttons zum Verschieben des Vorschaufensters
    new_window_size = 150
    max_window_start = max(0, len(text) - new_window_size)

    if "new_span_window_start" not in st.session_state:
        st.session_state.new_span_window_start = 0

    previous_col, range_col, next_col = st.columns([1, 2, 1])
    with previous_col:
        st.button(
            "← 50 Zeichen",
            key="move_new_span_window_back",
            use_container_width=True,
            disabled=st.session_state.new_span_window_start == 0,
            on_click=_move_new_span_window,
            args=(max_window_start, -50),
        )

    with range_col:
        window_start = st.session_state.new_span_window_start
        window_end = min(len(text), window_start + new_window_size)
        st.caption(f"Ausgangsbereich: Zeichen {window_start}–{window_end}")

    with next_col:
        st.button(
            "50 Zeichen →",
            key="move_new_span_window_forward",
            use_container_width=True,
            disabled=st.session_state.new_span_window_start >= max_window_start,
            on_click=_move_new_span_window,
            args=(max_window_start, 50),
        )

    # Slider zum Verschieben der Start- und Endoffsets innerhalb des Ausgangsbereichs
    new_default_start = st.session_state.new_span_window_start
    new_default_end = min(len(text), new_default_start + new_window_size)
    col_s, col_e = st.columns(2)

    with col_s:
        new_start_delta = st.slider(
            "Start verschieben (Zeichen)",
            min_value=-new_slider_range,
            max_value=new_slider_range,
            value=0,
            key="new_start_delta",
        )

    with col_e:
        new_end_delta = st.slider(
            "Ende verschieben (Zeichen)",
            min_value=-new_slider_range,
            max_value=new_slider_range,
            value=0,
            key="new_end_delta",
        )

    new_start = min(max(0, new_default_start + new_start_delta), len(text))
    new_end = min(max(0, new_default_end + new_end_delta), len(text))
    new_boundaries_valid = new_start < new_end

    new_context_start = max(0, new_start - 300)
    new_context_end = min(len(text), new_end + 300)
    new_before = (
        escape(text[new_context_start:new_start])
        .replace("\r", " ")
        .replace("\n", " ")
    )

    new_selected_text = (
        escape(text[new_start:new_end])
        .replace("\r", " ")
        .replace("\n", " ")
    )

    new_after = (
        escape(text[new_end:new_context_end])
        .replace("\r", " ")
        .replace("\n", " ")
    )

    new_color = TAP_COLORS[new_label]
    st.markdown(
        f"<div style='font-size:0.95rem;color:#333;margin-top:8px;"
        f"font-family:Georgia,serif;padding:12px 14px;background:#fafafa;"
        f"border:1px solid {new_color}66;border-radius:6px;line-height:1.7;"
        f"white-space:normal;max-height:260px;overflow-y:auto'>"
        f"{new_before}<mark style='background:{new_color}33;border-left:3px solid {new_color};"
        f"border-right:3px solid {new_color};padding:1px 2px'>{new_selected_text}</mark>{new_after}"
        f"</div>"
        f"<div style='font-size:0.8rem;color:#666;margin-top:4px'>"
        f"Vorschau: Zeichen {new_start}–{new_end}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not new_boundaries_valid:
        st.error("Start muss kleiner als Ende sein.")

    if st.button(
        "Span hinzufügen",
        type="primary",
        key="add_span",
        use_container_width=True,
        disabled=not new_boundaries_valid,
    ):
        spans.append(Span(
            start=new_start,
            end=new_end,
            label=new_label,
            text=text[new_start:new_end],
        ))
        spans.sort(key=lambda s: s.start)
        st.success(f"Span hinzugefügt: **{new_label}** "
                   f"({new_start}–{new_end})")
        st.rerun()

    with st.expander("Erweiterte Bearbeitung: exakte Zeichenoffsets"):
        advanced_start, advanced_end = st.columns(2)
        with advanced_start:
            exact_new_start = st.number_input(
                "Exakter Startoffset",
                min_value=0,
                max_value=max(len(text) - 1, 0),
                value=new_start,
                key=f"exact_new_start_{new_start_delta}_{new_end_delta}",
            )
        with advanced_end:
            exact_new_end = st.number_input(
                "Exakter Endoffset",
                min_value=1,
                max_value=len(text),
                value=new_end,
                key=f"exact_new_end_{new_start_delta}_{new_end_delta}",
            )
        if st.button("Span mit exakten Offsets hinzufügen", key="add_exact_span"):
            if int(exact_new_start) < int(exact_new_end):
                spans.append(Span(
                    start=int(exact_new_start),
                    end=int(exact_new_end),
                    label=new_label,
                    text=text[int(exact_new_start):int(exact_new_end)],
                ))
                spans.sort(key=lambda s: s.start)
                st.rerun()
            else:
                st.error("Start muss kleiner als Ende sein.")

    return spans

def _move_new_span_window(max_window_start: int, amount: int) -> None:
        """Verschiebt den Ausgangsbereich für eine neue Span.

        Args:
            max_window_start (int): Der maximale Startoffset des Fensters.
            amount (int): Zeichenanzahl, um die der Bereich verschoben wird.
        """
        st.session_state.new_span_window_start = min(
            max(0, st.session_state.new_span_window_start + amount),
            max_window_start,
        )

        keys_to_reset = {
            "new_start_delta",
            "new_end_delta",
            "new_slider_range",
        }

        keys_to_reset.update(
            key for key in st.session_state if key.startswith("exact_new_")
        )

        for widget_key in keys_to_reset:
            st.session_state.pop(widget_key, None)

def _option_label(i: int, s: Span) -> str:
        """Erstellt eine kurze Beschriftung für eine Span-Auswahloption.

        Args:
            i (int): Nullbasierte Position des Spans.
            s (Span): Durch die Option repräsentierter Span.

        Returns:
            str: Nummerierte Beschriftung mit Typ, Farbmarkierung und Vorschau.
        """

        dot = {"CLAIM": "🔵", "DATA": "🟢", "WARRANT": "🟠", "REBUTTAL": "🔴"}
        preview = s.text[:150].replace("\n", " ")

        if len(s.text) > 150:
            preview += "..."

        return f"#{i+1} {dot.get(s.label,'⚪')} {s.label} — {preview}"

# ── Hauptlayout ───────────────────────────────────────────────────────────────

def main() -> None:
    """Konfiguriert und rendert die vollständige Streamlit-Evaluierungsanwendung.

    Raises:
        RuntimeError: Wenn die Funktion außerhalb einer Streamlit-App läuft.
    """
    st.set_page_config(
        page_title="AM-Evaluierungsumgebung",
        page_icon="🔍",
        layout="wide",
    )

    state = AppState()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.caption("Prototypische Evaluierungsumgebung für ein deutsches Argumentation-Mining-Modell")
        st.divider()

        # TAP-Legende
        st.markdown("**TAP-Elemente**")
        for label, desc in TAP_DESCRIPTIONS.items():
            color = TAP_COLORS[label]
            st.markdown(
                f"<span style='background:{color}33;border-left:3px solid {color};"
                f"padding:4px 8px;border-radius:3px;display:block;margin-bottom:6px;"
                f"font-size:0.85rem'><b>{label}</b><br>"
                f"<span style='font-weight:normal'>{desc}</span></span>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Modell-Status
        st.markdown("**Modell-Status**")
        model = get_model()
        if model.is_loaded:
            st.success("✅ AM-Modell geladen")
        else:
            err = st.session_state.get("model_load_error", "")
            if err:
                st.error(f"❌ Ladefehler:\n{err}")
            else:
                st.warning(
                    "Kein trainiertes Modell gefunden.\n\n"
                    f"Erwartet unter:\n`{MODEL_PATH}`"
                )

        st.divider()

        # Klassifikation & Postprocessing Filter
        st.markdown("**Klassifikation & Filterung**")
        selected_threshold = st.slider(
            "Mindest-Konfidenz (Threshold):",
            min_value=0.30,
            max_value=0.95,
            value=0.65,
            step=0.05,
            help="Höhere Werte filtern unsichere und falsch klassifizierte Spans heraus. Standard: 0.65",
        )

        with st.expander("Postprocessing-Filter", expanded=False):
            min_words = st.slider(
                "Min. Wörter pro Span:",
                min_value=1,
                max_value=10,
                value=3,
                help="Filtert kurze Fragmente mit weniger Wörtern heraus.",
            )
            nms_enabled = st.checkbox(
                "Überlappungen filtern (NMS)",
                value=True,
                help="Entfernt überflüssige überlappende n-Gramm-Spans und behält den besten Span.",
            )
            clean_bounds = st.checkbox(
                "Ränder bereinigen",
                value=True,
                help="Entfernt führende/nachfolgende Satzzeichen und Anführungszeichen.",
            )

    # ── Hauptbereich ──────────────────────────────────────────────────────────
    st.title("Argumentation Mining - Test & Evaluation")
    st.markdown(
        "Analysiere Texte auf argumentative Strukturen nach dem "
        "**Toulmin-Argumentation-Pattern** und bewerte die Ergebnisse."
    )

    # Textauswahl
    st.markdown("---")
    st.markdown("### Textdatei auswählen")

    col_dir, col_reload = st.columns([5, 1])
    with col_dir:
        directory = st.text_input(
            "Verzeichnis mit Textdateien:",
            value=str(Path("data/ohi").resolve()),
            placeholder="z.B. C:\\Dokumente\\OHI-Texte",
        )
    with col_reload:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔄", help="Verzeichnis neu einlesen")

    scanner = FileScanner(directory)
    files   = scanner.scan()
    input_text = ""
    file_path  = ""

    if files:
        options = ["— Datei wählen —"] + list(files.keys())
        selected = st.selectbox(f"{len(files)} Textdatei(en) gefunden:", options)

        if selected != "— Datei wählen —":
            text_file = scanner.load(files[selected])
            file_path = str(text_file.filepath.resolve())
            st.subheader(f"Inhaltsvorschau: {selected}")
            st.code(body=text_file.content, language="plaintext", height=150)
            input_text = text_file.content
    else:
        if directory and not Path(directory).is_dir():
            st.warning(f"Verzeichnis nicht gefunden: `{directory}`")
        elif directory:
            st.info("Keine Textdateien (.txt, .md, .text) im Verzeichnis gefunden.")

    # Analysieren / Zurücksetzen
    col_a, col_r = st.columns(2)
    with col_a:
        analyze_btn = st.button(
            "Analysieren", type="primary", use_container_width=True
        )
    with col_r:
        if state.analyzed:
            if st.button("Zurücksetzen", use_container_width=True):
                state.reset()
                st.rerun()

    if analyze_btn:
        if not input_text.strip():
            st.warning("Bitte zuerst eine Textdatei auswählen.")
        else:
            with st.spinner("Analysiere Text mit Postprocessing..."):
                spans = model.predict(
                    input_text,
                    threshold=selected_threshold,
                    min_words=min_words,
                    nms=nms_enabled,
                    clean_boundaries=clean_bounds,
                )
            state.spans = spans
            state.text = input_text
            state.file = file_path
            state.analyzed = True
            st.rerun()

    # Ergebnisbereich
    if state.analyzed and state.text:
        text  = state.text
        spans = state.spans

        st.markdown("---")

        col_vis, col_stat = st.columns([3, 1])

        with col_vis:
            st.markdown(
                f"### Erkannte TAP-Elemente "
                f"<span style='font-size:0.85rem;font-weight:normal;color:#888'>"
                f"({text.count(chr(10))+1} Zeilen · {len(spans)} Spans)</span>",
                unsafe_allow_html=True,
            )
            
            render_displacy(text, spans)

            edit_tab, add_tab = st.tabs(["Span bearbeiten", "Neue Span hinzufügen"])
            with edit_tab:
                spans = render_annotation_editor(text, spans)

            with add_tab:
                spans = render_new_span_editor(text, spans)

            state.spans = spans

        with col_stat:
            st.markdown("### Übersicht")
            counts: Dict[str, int] = {}
            scores: Dict[str, list] = {}

            for s in spans:
                counts[s.label] = counts.get(s.label, 0) + 1
                scores.setdefault(s.label, []).append(s.score)

            for label in TAP_LABELS:
                count = counts.get(label, 0)
                color = TAP_COLORS[label]
                avg   = (sum(scores.get(label, [0])) /
                         max(len(scores.get(label, [1])), 1))
                score_str = f" · ⌀ {avg:.0%}" if count > 0 and avg > 0 else ""

                st.markdown(
                    f"<div style='background:{color}22;border-left:3px solid {color};"
                    f"padding:6px 10px;border-radius:4px;margin-bottom:6px'>"
                    f"<b>{label}</b>: {count}{score_str}</div>",
                    unsafe_allow_html=True,
                )

        # Export
        st.markdown("---")
        st.markdown("### Export")
        export = AnnotationExport(text=text, spans=state.spans)

        st.download_button(
            label="Als JSON exportieren",
            data=export.to_json(),
            file_name=f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.caption(
            "exportierte JSON's können dem Trainings-Datensatz später "
            "für verbesserte zukünftige Ergebnisse hinzugefügt werden."
        )

    # Footer
    st.markdown("---")
    st.caption(
        "Fachpraktikum NLP-IER - FernUniversität in Hagen - "
        "Felix Drescher"
    )


if __name__ == "__main__":
    main()
