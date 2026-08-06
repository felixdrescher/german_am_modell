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
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import spacy_streamlit
import streamlit as st

# Pipeline-Imports ermöglichen (Projektstruktur: app/ liegt unter Projektroot)
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    Kapselt das trainierte spaCy-Modell mit regelbasiertem Fallback.

    Attributes:
        model_path (Path): Verzeichnis des trainierten spaCy-Modells.
        threshold (float): Minimale Konfidenz für übernommene Modell-Spans.
        is_loaded (bool): Gibt an, ob das trainierte Modell geladen wurde.
        _nlp (Optional[Language]): Geladene spaCy-Pipeline oder ``None`` im
            Fallback-Modus.
    """

    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = 0.5):
        """Initialisiert einen Adapter für ein trainiertes spaCy-Modell.

        Args:
            model_path (Path): Verzeichnis mit dem trainierten Modell.
            threshold (float): Minimale Konfidenz für übernommene Vorhersagen.
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

            self.is_loaded = True

        except Exception as e:
            # Meldung in der sidebar anzeigen
            st.session_state["model_load_error"] = str(e)

        return self

    def predict(self, text: str) -> List[Span]:
        """Extrahiert Argumentations-Spans aus einem Text.

        Args:
            text (str): Zu analysierender Ausgangstext.

        Returns:
            List[Span]: Vorhersagen oberhalb der Konfidenzschwelle oder
                heuristische Fallback-Spans, wenn kein Modell geladen ist.
        """
        if self.is_loaded and self._nlp is not None:
            return self._predict_model(text)

    def _predict_model(self, text: str) -> List[Span]:
        """Führt eine satzweise Inferenz mit dem geladenen spaCy-Modell aus.

        Args:
            text (str): Zu analysierender Ausgangstext.

        Returns:
            List[Span]: Sortierte Modellvorhersagen oberhalb der Schwelle.

        Raises:
            AttributeError: Wenn dem konfigurierten Modell die erwartete API fehlt.
        """
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

        spans = []
        for sent_text, sent_start in sents:
            try:
                doc = self._nlp(sent_text)

            except Exception:
                continue

            for sp in doc.spans.get("sc", []):
                score = getattr(sp._, "score", 1.0)

                if score < self.threshold:
                    continue

                spans.append(Span(
                    start = sent_start + sp.start_char,
                    end   = sent_start + sp.end_char,
                    label = sp.label_,
                    text  = sp.text,
                    score = round(score, 3),
                ))

        spans.sort(key=lambda s: s.start)
        return spans


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
        title=f"🎨 Erkannte TAP-Elemente",
        spans_key="sc",
        displacy_options={"colors": TAP_COLORS},
        show_table=False
    )


def render_annotation_editor(text: str, spans: List[Span]) -> List[Span]:
    """Rendert Steuerelemente zum Bearbeiten, Löschen und Hinzufügen von Spans.

    Args:
        text (str): Vollständiger Text für die Span-Grenzen.
        spans (List[Span]): Veränderbare Liste der zu bearbeitenden Spans.

    Returns:
        List[Span]: Aktualisierte und nach Startoffset sortierte Span-Liste.

    Raises:
        RuntimeError: Wenn die Funktion außerhalb eines Streamlit-Kontexts läuft.
    """
    st.markdown("###Spans bearbeiten")
    st.markdown(f"**Span auswählen** ({len(spans)} erkannt)")

    if spans:
        dot = {"CLAIM": "🔵", "DATA": "🟢", "WARRANT": "🟠", "REBUTTAL": "🔴"}

        def option_label(i: int, s: Span) -> str:
            """Erstellt eine kurze Beschriftung für eine Span-Auswahloption.

            Args:
                i (int): Nullbasierte Position des Spans.
                s (Span): Durch die Option repräsentierter Span.

            Returns:
                str: Nummerierte Beschriftung mit Typ, Farbmarkierung und Vorschau.
            """
            preview = s.text[:60].replace("\n", " ")

            if len(s.text) > 60:
                preview += "..."

            return f"#{i+1} {dot.get(s.label,'⚪')} {s.label} — {preview}"

        options = [option_label(i, s) for i, s in enumerate(spans)]

        st.session_state.active_span_idx = min(
            st.session_state.active_span_idx, len(spans) - 1
        )

        selected = st.selectbox(
            "Span:", options,
            index=st.session_state.active_span_idx,
            key="span_selector",
            label_visibility="collapsed",
        )
        
        idx   = options.index(selected)
        st.session_state.active_span_idx = idx
        span  = spans[idx]
        color = TAP_COLORS.get(span.label, "#999")

        # Karte
        st.markdown(
            f"<div style='background:{color}11;border:1px solid {color}55;"
            f"border-radius:8px;padding:14px 16px;margin-top:6px'>",
            unsafe_allow_html=True,
        )

        col_lbl, col_del = st.columns([6, 1])
        with col_lbl:
            new_lbl = st.selectbox(
                "Label", TAP_LABELS,
                index=TAP_LABELS.index(span.label),
                key=f"edit_label_{idx}_{span.start}",
                label_visibility="collapsed",
            )

            if new_lbl != span.label:
                spans[idx].label = new_lbl
                st.rerun()

        with col_del:
            if st.button("✕", key="edit_del", help="Span löschen"):
                spans.pop(idx)
                st.session_state.active_span_idx = max(0, idx - 1)
                st.rerun()

        col_s, col_e, col_apply = st.columns([2, 2, 2])
        with col_s:
            new_s = st.number_input(
                "Start", min_value=0, max_value=max(len(text) - 1, 0),
                value=span.start,
                key=f"edit_start_{idx}_{span.start}",
            )

        with col_e:
            new_e = st.number_input(
                "End", min_value=1, max_value=len(text),
                value=span.end,
                key=f"edit_end_{idx}_{span.start}",
            )

        with col_apply:
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Übernehmen (Enter)", key="edit_apply", use_container_width=True):

                if int(new_s) < int(new_e):
                    spans[idx].start = int(new_s)
                    spans[idx].end   = int(new_e)
                    spans[idx].text  = text[int(new_s):int(new_e)]
                    spans.sort(key=lambda s: s.start)
                    st.rerun()

                else:
                    st.error("Start muss kleiner als End sein.")

        fragment = text[span.start:span.end]
        preview  = fragment[:160] + ("…" if len(fragment) > 160 else "")
        st.markdown(
            f"<div style='font-size:0.88rem;color:#444;margin-top:8px;"
            f"font-family:Georgia,serif;padding:6px 10px;background:white;"
            f"border-radius:4px;border-left:3px solid {color};line-height:1.6'>"
            f"{preview}</div>", unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Noch keine Spans vorhanden.")

    # Neue Span hinzufügen
    st.markdown("---")
    st.markdown("**Neue Span hinzufügen**")
    st.caption(f"Dokumentlänge: **{len(text)} Zeichen**")

    col_s, col_e, col_l, col_btn = st.columns([2, 2, 2, 1])
    with col_s:
        new_start = st.number_input(
            "Start", min_value=0, max_value=max(len(text) - 1, 0),
            value=0, key="new_start",
        )
    with col_e:
        new_end = st.number_input(
            "End", min_value=1, max_value=len(text),
            value=min(50, len(text)), key="new_end",
        )
    with col_l:
        new_label = st.selectbox("Label", TAP_LABELS, key="new_label")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        add_btn = st.button("➕", key="add_span", use_container_width=True)

    if int(new_start) < int(new_end):
        preview_txt = text[int(new_start):int(new_end)]
        st.markdown(
            f"**Vorschau:** `{preview_txt[:120]}{'…' if len(preview_txt) > 120 else ''}`"
        )

    if add_btn:
        if int(new_start) >= int(new_end):
            st.error("Start muss kleiner als End sein.")
        else:
            spans.append(Span(
                start=int(new_start), end=int(new_end),
                label=new_label,
                text=text[int(new_start):int(new_end)],
            ))
            spans.sort(key=lambda s: s.start)
            st.success(f"Span hinzugefügt: **{new_label}** "
                       f"({int(new_start)}–{int(new_end)})")
            st.rerun()

    return spans


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
        st.title("🔍 Argumentation Mining")
        st.caption("Prototypische Evaluierungsumgebung · FernUni Hagen")
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
            st.subheader(f"Inhalt: {selected}")
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
            "🔍 Analysieren", type="primary", use_container_width=True
        )
    with col_r:
        if state.analyzed:
            if st.button("🗑️ Zurücksetzen", use_container_width=True):
                state.reset()
                st.rerun()

    if analyze_btn:
        if not input_text.strip():
            st.warning("Bitte zuerst eine Textdatei auswählen.")
        else:
            with st.spinner("Analysiere Text..."):
                spans = model.predict(input_text)
            state.spans    = spans
            state.text     = input_text
            state.file     = file_path
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

        # Annotation-Editor
        st.markdown("---")
        state.spans = render_annotation_editor(text, spans)

        # Export
        st.markdown("---")
        st.markdown("### ⬇️ Export")
        export = AnnotationExport(text=text, spans=state.spans)
        st.download_button(
            label="Als JSON exportieren (Fine-Tuning)",
            data=export.to_json(),
            file_name=f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption(
            "💡 Exportierte JSONs können dem Trainings-Datensatz später "
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
