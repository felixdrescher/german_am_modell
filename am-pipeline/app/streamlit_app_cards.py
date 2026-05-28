"""
streamlit_app.py
----------------
Prototypische Test- und Evaluierungsumgebung für das Argumentation-Mining-Modell.
Forschende können:
  1. Textdateien aus einem lokalen Verzeichnis laden (schreibgeschützt)
  2. Die erkannten TAP-Elemente in einer scrollbaren Box mit Zeilennummern sehen
  3. Spans manuell annotieren: Text markieren → Label zuweisen, Spans löschen
  4. Arbeitsfortschritt in SQLite speichern und später fortsetzen
  5. Feedback als spaCy-kompatibles JSON exportieren (späteres Fine-Tuning)

Starten mit:
  streamlit run app/streamlit_app.py
"""

import json
import re
import sqlite3
import sys
import ptvsd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Konfiguration ─────────────────────────────────────────────────────────────

TAP_LABELS = ["CLAIM", "DATA", "WARRANT", "REBUTTAL"]

TAP_COLORS = {
    "CLAIM":    "#4A90D9",
    "DATA":     "#27AE60",
    "WARRANT":  "#E67E22",
    "REBUTTAL": "#E74C3C",
}

TAP_DESCRIPTIONS = {
    "CLAIM":    "Behauptung – die zentrale These oder Position",
    "DATA":     "Daten – stützende Fakten oder Belege",
    "WARRANT":  "Warrant – Begründung, warum die Daten die Behauptung stützen",
    "REBUTTAL": "Rebuttal – Einschränkung oder Gegenargument",
}

DB_PATH = Path(__file__).parent / "progress.db"


# ── Datenbank ─────────────────────────────────────────────────────────────────

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    """Erstellt Tabellen falls noch nicht vorhanden."""
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                filepath    TEXT NOT NULL,
                text        TEXT NOT NULL,
                spans_json  TEXT NOT NULL DEFAULT '[]',
                status      TEXT NOT NULL DEFAULT 'in_progress',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.commit()


def db_save_session(filename: str, filepath: str, text: str,
                    spans: list, status: str = "in_progress") -> int:
    now = datetime.now().isoformat()
    with db_connect() as conn:
        # Bestehende Session für diese Datei aktualisieren oder neu anlegen
        row = conn.execute(
            "SELECT id FROM sessions WHERE filepath = ?", (filepath,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE sessions SET spans_json=?, status=?, updated_at=? WHERE id=?",
                (json.dumps(spans, ensure_ascii=False), status, now, row["id"])
            )
            conn.commit()
            return row["id"]
        else:
            cur = conn.execute(
                "INSERT INTO sessions (filename, filepath, text, spans_json, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (filename, filepath, text,
                 json.dumps(spans, ensure_ascii=False), status, now, now)
            )
            conn.commit()
            return cur.lastrowid


def db_load_session(filepath: str) -> Optional[dict]:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE filepath = ?", (filepath,)
        ).fetchone()
        if row:
            d = dict(row)
            d["spans"] = json.loads(d["spans_json"])
            return d
    return None


def db_list_sessions() -> list:
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT filename, filepath, status, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def db_delete_session(filepath: str):
    with db_connect() as conn:
        conn.execute("DELETE FROM sessions WHERE filepath = ?", (filepath,))
        conn.commit()


# ── Datei-Hilfsfunktionen ─────────────────────────────────────────────────────

def scan_text_directory(directory: str) -> Dict[str, Path]:
    supported = {".txt", ".md", ".text"}
    result = {}
    try:
        p = Path(directory)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in supported:
                    result[f.name] = f
    except PermissionError:
        pass
    return result


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# ── Modell-Stub ───────────────────────────────────────────────────────────────

def run_model_stub(text: str) -> list:
    spans = []
    patterns = {
        "CLAIM":    [r"Ich (denke|meine|glaube|finde)[^.]*\.", r"sollte[^.]*\.", r"bin ich[^.]*\."],
        "DATA":     [r"Mit [^.]*\d+%[^.]*\.", r"Die \w+ war[^.]*\.", r"\d+[^.]*\."],
        "WARRANT":  [r"bedeutet[^.]*\.", r"weil[^.]*\.", r"Verantwortung[^.]*\."],
        "REBUTTAL": [r"Obwohl[^.]*\.", r"obwohl[^.]*\.", r"stimmt schon[^.]*\."],
    }
    for label, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, text):
                spans.append({"start": match.start(), "end": match.end(),
                               "label": label, "text": match.group()})
    spans.sort(key=lambda x: x["start"])
    filtered, last_end = [], -1
    for span in spans:
        if span["start"] >= last_end:
            filtered.append(span)
            last_end = span["end"]
    return filtered


# ── Visualisierung via displaCy ───────────────────────────────────────────────

def spans_to_displacy_manual(text: str, spans: list) -> str:
    """
    Konvertiert unsere Span-Liste in displaCy's Manual-Format und rendert
    das Ergebnis als HTML-String (style='span').

    Das Manual-Format erlaubt displaCy ohne ein geladenes spaCy-Modell zu
    nutzen. Sobald das echte GBERT-Modell vorliegt, kann diese Funktion durch
    einen direkten displacy.render(doc, style='span') Aufruf ersetzt werden.
    """
    import spacy
    from spacy import displacy

    # Tokenisierung mit dem deutschen Blank-Modell (kein ML nötig)
    nlp = spacy.blank("de")
    doc = nlp(text)

    # Span-Dicts → spaCy Span-Objekte unter doc.spans["sc"]
    spacy_spans = []
    for s in spans:
        span = doc.char_span(s["start"], s["end"],
                             label=s["label"], alignment_mode="expand")
        if span is not None:
            spacy_spans.append(span).append(" ")
    doc.spans["sc"] = spacy_spans

    html = displacy.render(
        doc,
        style="span",
        options={
            "colors": TAP_COLORS,
            "spans_key": "sc",
        },
        page=False,
        minify=True,
    )
    return html


def render_displacy_view(text: str, spans: list) -> None:
    """
    Rendert die displaCy-Span-Visualisierung in einer scrollbaren Box.
    Lange Texte werden satzweise aufgeteilt für bessere Lesbarkeit.
    """
    import spacy
    from spacy import displacy

    nlp = spacy.blank("de")

    # Satzweise aufteilen: displaCy wird bei sehr langen Texten unlesbar
    # wenn alles in einer Zeile dargestellt wird.
    # Wir splitten an Absätzen (Leerzeilen) oder alle ~800 Zeichen.
    breakpoint()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    # Zeichenoffsets der Absätze berechnen
    para_offsets = []
    pos = 0
    for para in text.split("\n\n"):
        para_offsets.append(pos)
        pos += len(para) + 2   # +2 für \n\n

    all_html = ""
    for para_idx, para in enumerate(paragraphs):
        para_start = para_offsets[para_idx] if para_idx < len(para_offsets) else 0

        doc = nlp(para)

        # Spans die in diesen Absatz fallen herausfiltern und Offsets anpassen
        local_spans = []
        for s in spans:
            # Span muss sich mit dem Absatz überlappen
            s_end_adj   = s["end"]
            s_start_adj = s["start"]
            para_end    = para_start + len(para)

            if s_start_adj >= para_end or s_end_adj <= para_start:
                continue

            local_start = max(s_start_adj - para_start, 0)
            local_end   = min(s_end_adj   - para_start, len(para))

            sp = doc.char_span(local_start, local_end,
                               label=s["label"], alignment_mode="expand")
            if sp is not None:
                local_spans.append(sp)

        doc.spans["sc"] = local_spans

        para_html = displacy.render(
            doc,
            style="span",
            options={"colors": TAP_COLORS, "spans_key": "sc"},
            page=False,
            minify=True,
        )
        # Absatz-Trennlinie zwischen Abschnitten
        if para_idx > 0:
            all_html += "<hr style='border:none;border-top:1px solid #eee;margin:8px 0'>"
        all_html += para_html

    # Scrollbare Box
    wrapped = f"""
    <div style="
        height: 520px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 16px 20px;
        background: #ffffff;
        font-family: 'Georgia', serif;
        font-size: 0.92rem;
        line-height: 1.8;
    ">
        {all_html}
    </div>
    """
    components.html(wrapped, height=540, scrolling=False)


# ── Annotation-Editor (editierbare Karten mit Zeichengrenzen) ────────────────

def render_annotation_editor(text: str, spans: list) -> list:
    """
    Variante: Jede Span als vollständig editierbare Karte.
    - Start/End-Zeichen direkt anpassbar → Span wächst/schrumpft live
    - Label per Dropdown änderbar
    - Neue Span per Zeichenbereich oder Texteingabe
    - Span löschen per ✕-Button
    """
    st.markdown("### ✏️ Spans bearbeiten")
    st.caption(
        "Jede Span kann direkt über ihre **Zeichenpositionen** (Start/End) "
        "vergrößert oder verkleinert werden. Die Vorschau aktualisiert sich sofort."
    )

    # ── Neue Span hinzufügen ──────────────────────────────────────────────────
    with st.expander("➕ Neue Span hinzufügen", expanded=False):
        tab_pos, tab_txt = st.tabs(["📍 Per Zeichenposition", "🔤 Per Textsuche"])

        with tab_pos:
            st.caption(
                "Trage Start- und Endposition (Zeichenindex) ein. "
                f"Dokumentlänge: **{len(text)} Zeichen**."
            )
            col_s, col_e, col_l, col_btn = st.columns([2, 2, 2, 1])
            with col_s:
                new_start = st.number_input(
                    "Start", min_value=0, max_value=len(text) - 1,
                    value=0, key="new_start", label_visibility="visible"
                )
            with col_e:
                new_end = st.number_input(
                    "End", min_value=1, max_value=len(text),
                    value=min(50, len(text)), key="new_end", label_visibility="visible"
                )
            with col_l:
                new_label_pos = st.selectbox(
                    "Label", TAP_LABELS, key="new_label_pos"
                )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                add_pos_btn = st.button("➕", key="add_pos", use_container_width=True)

            if new_start < new_end:
                preview_pos = text[int(new_start):int(new_end)]
                st.markdown(
                    f"**Vorschau:** `{preview_pos[:120]}{'…' if len(preview_pos)>120 else ''}`"
                )

            if add_pos_btn:
                if new_start >= new_end:
                    st.error("Start muss kleiner als End sein.")
                else:
                    fragment = text[int(new_start):int(new_end)]
                    spans.append({
                        "start": int(new_start),
                        "end":   int(new_end),
                        "label": new_label_pos,
                        "text":  fragment,
                    })
                    spans.sort(key=lambda x: x["start"])
                    st.success(
                        f"Span hinzugefügt: **{new_label_pos}** "
                        f"({int(new_start)}–{int(new_end)}): `{fragment[:60]}`"
                    )
                    st.rerun()

        with tab_txt:
            st.caption("Textstelle eintippen oder einkopieren — Position wird automatisch ermittelt.")
            col_t, col_l2, col_b2 = st.columns([4, 2, 1])
            with col_t:
                new_txt = st.text_input(
                    "Textausschnitt", key="new_span_txt",
                    placeholder="Exakter Text aus dem Dokument...",
                    label_visibility="collapsed",
                )
            with col_l2:
                new_label_txt = st.selectbox(
                    "Label", TAP_LABELS, key="new_label_txt",
                    label_visibility="collapsed",
                )
            with col_b2:
                add_txt_btn = st.button("➕", key="add_txt", use_container_width=True)

            if add_txt_btn and new_txt.strip():
                matches = list(re.finditer(re.escape(new_txt.strip()), text))
                if not matches:
                    st.error("Textausschnitt nicht im Dokument gefunden.")
                else:
                    existing_starts = {s["start"] for s in spans}
                    added = False
                    for m in matches:
                        if m.start() not in existing_starts:
                            spans.append({
                                "start": m.start(), "end": m.end(),
                                "label": new_label_txt, "text": new_txt.strip(),
                            })
                            spans.sort(key=lambda x: x["start"])
                            st.success(
                                f"Span hinzugefügt: **{new_label_txt}** "
                                f"({m.start()}–{m.end()})"
                            )
                            added = True
                            break
                    if not added:
                        st.warning("Dieser Textausschnitt ist bereits annotiert.")
                    st.rerun()

    # ── Editierbare Span-Karten ───────────────────────────────────────────────
    st.markdown(f"**Erkannte / annotierte Spans** ({len(spans)})")

    if not spans:
        st.info("Noch keine Spans vorhanden.")
        return spans

    to_delete = None
    changed   = False

    for i, span in enumerate(spans):
        color = TAP_COLORS.get(span["label"], "#999")

        st.markdown(
            f"<div style='background:{color}11;border:1px solid {color}44;"
            f"border-radius:6px;padding:10px 14px;margin-bottom:8px'>",
            unsafe_allow_html=True,
        )

        # Zeile 1: Label + Löschen
        col_lbl, col_del = st.columns([6, 1])
        with col_lbl:
            new_lbl = st.selectbox(
                "Label",
                TAP_LABELS,
                index=TAP_LABELS.index(span["label"]),
                key=f"lbl_{i}",
                label_visibility="collapsed",
            )
            if new_lbl != span["label"]:
                spans[i]["label"] = new_lbl
                changed = True
        with col_del:
            if st.button("✕", key=f"del_{i}", help="Span löschen"):
                to_delete = i

        # Zeile 2: Start / End als editierbare Zahlenfelder
        col_s, col_e, col_apply = st.columns([2, 2, 2])
        with col_s:
            new_s = st.number_input(
                "Start", min_value=0, max_value=len(text) - 1,
                value=span["start"], key=f"start_{i}",
            )
        with col_e:
            new_e = st.number_input(
                "End", min_value=1, max_value=len(text),
                value=span["end"], key=f"end_{i}",
            )
        with col_apply:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↩ Übernehmen", key=f"apply_{i}", use_container_width=True):
                if int(new_s) < int(new_e):
                    spans[i]["start"] = int(new_s)
                    spans[i]["end"]   = int(new_e)
                    spans[i]["text"]  = text[int(new_s):int(new_e)]
                    spans.sort(key=lambda x: x["start"])
                    changed = True
                else:
                    st.error("Start muss kleiner als End sein.")

        # Zeile 3: Textvorschau des aktuellen Span-Inhalts
        current_fragment = text[span["start"]:span["end"]]
        preview = current_fragment[:120] + ("…" if len(current_fragment) > 120 else "")
        st.markdown(
            f"<div style='font-size:0.85rem;color:#555;margin-top:4px;"
            f"font-family:Georgia,serif;padding:4px 6px;background:white;"
            f"border-radius:3px;border-left:3px solid {color}'>"
            f"{preview}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    if to_delete is not None:
        spans.pop(to_delete)
        st.rerun()

    if changed:
        st.rerun()

    return spans


# ── Export ────────────────────────────────────────────────────────────────────

def export_feedback(text: str, spans: list) -> dict:
    return {
        "text": text,
        "spans": spans,
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "source": "streamlit_eval",
            "span_count": len(spans),
        }
    }


# ── Hauptlayout ───────────────────────────────────────────────────────────────

def main():
    db_init()

    st.set_page_config(
        page_title="AM-Evaluierungsumgebung",
        page_icon="🔍",
        layout="wide",
    )

    # Session-State initialisieren
    if "current_spans" not in st.session_state:
        st.session_state.current_spans = []
    if "current_text" not in st.session_state:
        st.session_state.current_text = ""
    if "current_file" not in st.session_state:
        st.session_state.current_file = ""
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🔍 Argumentation Mining")
        st.caption("Prototypische Evaluierungsumgebung · FernUni Hagen")
        st.divider()

        st.markdown("**TAP-Elemente**")
        for label, desc in TAP_DESCRIPTIONS.items():
            color = TAP_COLORS[label]
            st.markdown(
                f"<span style='background:{color}33;border-left:3px solid {color};"
                f"padding:4px 8px;border-radius:3px;display:block;margin-bottom:6px;"
                f"font-size:0.85rem'>"
                f"**{label}**<br><span style='font-weight:normal'>{desc}</span></span>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Modell-Status**")
        st.info("🔧 Demo-Modus\n\nEchtes GBERT-Modell wird nach dem Training eingebunden.")

        st.divider()
        st.markdown("**💾 Gespeicherte Sessions**")
        sessions = db_list_sessions()
        if sessions:
            for s in sessions[:8]:
                status_icon = "✅" if s["status"] == "done" else "🔄"
                updated = s["updated_at"][:16].replace("T", " ")
                st.markdown(
                    f"<div style='font-size:0.8rem;padding:4px 0;border-bottom:1px solid #eee'>"
                    f"{status_icon} <b>{s['filename']}</b><br>"
                    f"<span style='color:#999'>{updated}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Noch keine Sessions gespeichert.")

    # ── Hauptbereich ──────────────────────────────────────────────────────────
    st.title("Argumentation Mining · Test & Evaluation")
    st.markdown(
        "Analysiere Texte auf argumentative Strukturen nach dem "
        "**Toulmin-Argumentation-Pattern** und bewerte die Ergebnisse."
    )

    # ── Textauswahl ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 Text auswählen oder eingeben")

    col_dir, col_reload = st.columns([5, 1])
    with col_dir:
        directory = st.text_input(
            "Verzeichnis mit Textdateien:",
            value=str(Path("data/ohi").resolve()),
            placeholder="z.B. C:\\Users\\Felix\\Dokumente\\OHI",
        )
    with col_reload:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔄", help="Verzeichnis neu einlesen")

    files = scan_text_directory(directory)
    input_text = ""

    if files:
        file_options = ["— Datei wählen —"] + list(files.keys())
        selected_file = st.selectbox(
            f"{len(files)} Textdatei(en) gefunden:",
            file_options,
        )

        if selected_file != "— Datei wählen —":
            file_path = str(files[selected_file].resolve())
            file_content = read_text_file(files[selected_file])

            # Gespeicherten Fortschritt prüfen
            saved = db_load_session(file_path)
            if saved and not st.session_state.get("resume_checked_" + selected_file):
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.info(
                        f"💾 Gespeicherter Fortschritt gefunden "
                        f"({len(saved['spans'])} Spans, "
                        f"zuletzt: {saved['updated_at'][:16].replace('T',' ')})"
                    )
                with col_r2:
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("▶️ Fortsetzen", use_container_width=True):
                            st.session_state.current_spans = saved["spans"]
                            st.session_state.current_text  = file_content
                            st.session_state.current_file  = file_path
                            st.session_state.analyzed      = True
                            st.session_state["resume_checked_" + selected_file] = True
                            st.rerun()
                    with col_no:
                        if st.button("🔄 Neu starten", use_container_width=True):
                            st.session_state["resume_checked_" + selected_file] = True
                            st.rerun()

            st.text_area(
                f"Inhalt: {selected_file} (schreibgeschützt)",
                value=file_content,
                height=150,
                disabled=True,
            )
            input_text = file_content

        else:
            input_text = st.text_area(
                "Oder Text direkt eingeben:",
                height=150,
                placeholder="Gib hier einen deutschen Text ein...",
            )
    else:
        if directory and not Path(directory).is_dir():
            st.warning(f"Verzeichnis nicht gefunden: `{directory}`")
        elif directory:
            st.info("Keine Textdateien (.txt, .md, .text) im Verzeichnis gefunden.")
        input_text = st.text_area(
            "Text direkt eingeben:",
            height=150,
            placeholder="Gib hier einen deutschen Text ein...",
        )

    # ── Analysieren ───────────────────────────────────────────────────────────
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_btn = st.button("🔍 Analysieren", type="primary", use_container_width=True)
    with col_btn2:
        if st.session_state.analyzed and st.session_state.current_text:
            if st.button("🗑️ Analyse zurücksetzen", use_container_width=True):
                st.session_state.analyzed      = False
                st.session_state.current_spans = []
                st.session_state.current_text  = ""
                st.session_state.current_file  = ""
                st.rerun()

    if analyze_btn and input_text.strip():
        with st.spinner("Analysiere Text..."):
            spans = run_model_stub(input_text)
        st.session_state.current_spans = spans
        st.session_state.current_text  = input_text
        st.session_state.current_file  = (
            file_path if "file_path" in dir() and file_path else ""
        )
        st.session_state.analyzed = True
        st.rerun()

    elif analyze_btn:
        st.warning("Bitte gib zuerst einen Text ein.")

    # ── Ergebnisbereich ───────────────────────────────────────────────────────
    if st.session_state.analyzed and st.session_state.current_text:
        text  = st.session_state.current_text
        spans = st.session_state.current_spans

        st.markdown("---")

        # Visualisierung + Statistik
        col_vis, col_stat = st.columns([3, 1])

        with col_vis:
            lines = text.splitlines()
            st.markdown(
                f"### 🎨 Erkannte TAP-Elemente "
                f"<span style='font-size:0.85rem;font-weight:normal;color:#888'>"
                f"({len(lines)} Zeilen, {len(spans)} Spans)</span>",
                unsafe_allow_html=True,
            )
            render_displacy_view(text, spans)

        with col_stat:
            st.markdown("### 📊 Übersicht")
            counts = {}
            for s in spans:
                counts[s["label"]] = counts.get(s["label"], 0) + 1
            for label in TAP_LABELS:
                count = counts.get(label, 0)
                color = TAP_COLORS[label]
                st.markdown(
                    f"<div style='background:{color}22;border-left:3px solid {color};"
                    f"padding:6px 10px;border-radius:4px;margin-bottom:6px'>"
                    f"<b>{label}</b>: {count}</div>",
                    unsafe_allow_html=True,
                )

        # ── Annotation-Editor ─────────────────────────────────────────────────
        st.markdown("---")
        updated_spans = render_annotation_editor(text, spans)
        if updated_spans != spans:
            st.session_state.current_spans = updated_spans

        # ── Speichern & Export ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💾 Speichern & Export")

        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            if st.button("💾 Fortschritt speichern", use_container_width=True):
                fname = (Path(st.session_state.current_file).name
                         if st.session_state.current_file else "manuell")
                fpath = st.session_state.current_file or "manual_input"
                db_save_session(fname, fpath, text,
                                st.session_state.current_spans, "in_progress")
                st.success("Fortschritt gespeichert ✓")

        with col_s2:
            if st.button("✅ Als fertig markieren", use_container_width=True):
                fname = (Path(st.session_state.current_file).name
                         if st.session_state.current_file else "manuell")
                fpath = st.session_state.current_file or "manual_input"
                db_save_session(fname, fpath, text,
                                st.session_state.current_spans, "done")
                st.success("Session als fertig markiert ✓")

        with col_s3:
            export = export_feedback(text, st.session_state.current_spans)
            export_json = json.dumps(export, ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Als JSON exportieren",
                data=export_json,
                file_name=f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.caption(
            "💡 Gespeicherte JSONs können dem Trainings-Datensatz "
            "hinzugefügt werden (Fine-Tuning)."
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Fachpraktikum NLP-IER · FernUniversität in Hagen · "
        "Entwicklung: Felix Drescher · Betreuer: Dr. Nawroth, Prof. Hemmje"
    )


if __name__ == "__main__":
    main()