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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

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


# ── Visualisierung ────────────────────────────────────────────────────────────

def highlight_text(text: str, spans: list) -> str:
    """HTML mit Zeilennummern und farbigen Spans für scrollbare Box."""

    lines = text.splitlines()

    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    sorted_spans = sorted(spans, key=lambda x: x["start"])

    rows = ""

    for line_idx, line in enumerate(lines):
        line_start = line_starts[line_idx]
        line_end = line_start + len(line)

        relevant = [
            s for s in sorted_spans
            if s["start"] < line_end and s["end"] > line_start
        ]

        if not relevant:
            rows += (
                f"<tr style='border:none'>"
                f"<td style='border:none;padding-bottom:4px'>"
                f"{line or '&nbsp;'}"
                f"</td></tr>"
            )
            continue

        cell = ""
        cursor = line_start

        for span in relevant:
            s_start = max(span["start"], line_start)
            s_end = min(span["end"], line_end)
            label = span["label"]
            color = TAP_COLORS.get(label, "#999")

            if cursor < s_start:
                cell += line[cursor - line_start: s_start - line_start]

            fragment = line[s_start - line_start: s_end - line_start]

            cell += (
                f"<mark style='background:{color}22;"
                f"border-bottom:2.5px solid {color};"
                f"border-radius:3px;padding:1px 3px;margin:0 1px;'>"
                f"<span style='color:{color};font-weight:700;"
                f"font-size:0.75rem;vertical-align:super;margin-right:2px'>"
                f"{label}</span>"
                f"{fragment}</mark>"
            )

            cursor = s_end

        if cursor < line_end:
            cell += line[cursor - line_start:]

        rows += (
            f"<tr style='border:none'>"
            f"<td style='border:none;padding-bottom:4px;font-size:1.2rem'>"
            f"{cell or '&nbsp;'}"
            f"</td></tr>"
        )

    return f"""
    <table style="
        border-collapse: separate;
        border-spacing: 0 6px;
        border: none;
        width: 100%;
        line-height: 2.7;
    ">
        {rows}
    </table>
    """


# ── Annotation-Editor (Einzelkarte mit Span-Auswahl) ─────────────────────────

def render_annotation_editor(text: str, spans: list) -> list:
    """
    Zeigt immer nur eine editierbare Karte für die aktuell ausgewählte Span.
    Neue Spans per Zeichenposition werden unterhalb der Karte angezeigt.
    """
    st.markdown("### ✏️ Spans bearbeiten")

    # ── Span-Auswahl + Einzelkarte ────────────────────────────────────────────
    st.markdown(f"**Span auswählen** ({len(spans)} erkannt)")

    if spans:
        def span_label(i: int, s: dict) -> str:
            color_dot = "🔵" if s["label"] == "CLAIM"   else \
                        "🟢" if s["label"] == "DATA"    else \
                        "🟠" if s["label"] == "WARRANT" else "🔴"
            preview = s["text"][:60].replace("\n", " ")
            if len(s["text"]) > 60:
                preview += "…"
            return f"#{i+1} {color_dot} {s['label']} — {preview}"

        options = [span_label(i, s) for i, s in enumerate(spans)]

        if "active_span_idx" not in st.session_state:
            st.session_state.active_span_idx = 0
        st.session_state.active_span_idx = min(
            st.session_state.active_span_idx, len(spans) - 1
        )

        selected = st.selectbox(
            "Span:",
            options,
            index=st.session_state.active_span_idx,
            key="span_selector",
            label_visibility="collapsed",
        )
        idx   = options.index(selected)
        st.session_state.active_span_idx = idx
        span  = spans[idx]
        color = TAP_COLORS.get(span["label"], "#999")

        # Editierbare Einzelkarte
        st.markdown(
            f"<div style='background:{color}11;border:1px solid {color}55;"
            f"border-radius:8px;padding:14px 16px;margin-top:6px'>",
            unsafe_allow_html=True,
        )

        col_lbl, col_del = st.columns([6, 1])
        with col_lbl:
            new_lbl = st.selectbox(
                "Label", TAP_LABELS,
                index=TAP_LABELS.index(span["label"]),
                key=f"edit_label_{idx}_{span['start']}",
                label_visibility="collapsed",
            )
            if new_lbl != span["label"]:
                spans[idx]["label"] = new_lbl
                st.rerun()
        with col_del:
            if st.button("✕", key="edit_del", help="Span löschen"):
                spans.pop(idx)
                st.session_state.active_span_idx = max(0, idx - 1)
                st.rerun()

        col_s, col_e, col_apply = st.columns([2, 2, 2])
        with col_s:
            new_s = st.number_input(
                "Start", min_value=0, max_value=len(text) - 1,
                value=span["start"], key=f"edit_start_{idx}_{span['start']}",
            )
        with col_e:
            new_e = st.number_input(
                "End", min_value=1, max_value=len(text),
                value=span["end"], key=f"edit_end_{idx}_{span['start']}",
            )
        with col_apply:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("↩ Übernehmen", key="edit_apply", use_container_width=True):
                if int(new_s) < int(new_e):
                    spans[idx]["start"] = int(new_s)
                    spans[idx]["end"]   = int(new_e)
                    spans[idx]["text"]  = text[int(new_s):int(new_e)]
                    spans.sort(key=lambda x: x["start"])
                    st.rerun()
                else:
                    st.error("Start muss kleiner als End sein.")

        current_fragment = text[span["start"]:span["end"]]
        preview = current_fragment[:160] + ("…" if len(current_fragment) > 160 else "")
        st.markdown(
            f"<div style='font-size:0.88rem;color:#444;margin-top:8px;"
            f"font-family:Georgia,serif;padding:6px 10px;background:white;"
            f"border-radius:4px;border-left:3px solid {color};line-height:1.6'>"
            f"{preview}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Noch keine Spans vorhanden.")

    # ── Neue Span hinzufügen (unterhalb der Karte) ────────────────────────────
    st.markdown("---")
    st.markdown("**➕ Neue Span hinzufügen**")
    st.caption(f"Dokumentlänge: **{len(text)} Zeichen**")

    col_s, col_e, col_l, col_btn = st.columns([2, 2, 2, 1])
    with col_s:
        new_start = st.number_input(
            "Start", min_value=0, max_value=len(text) - 1,
            value=0, key="new_start",
        )
    with col_e:
        new_end = st.number_input(
            "End", min_value=1, max_value=len(text),
            value=min(50, len(text)), key="new_end",
        )
    with col_l:
        new_label = st.selectbox("Label", TAP_LABELS, key="new_label_pos")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        add_btn = st.button("➕", key="add_pos", use_container_width=True)

    if int(new_start) < int(new_end):
        preview_pos = text[int(new_start):int(new_end)]
        st.markdown(
            f"**Vorschau:** `{preview_pos[:120]}{'…' if len(preview_pos) > 120 else ''}`"
        )

    if add_btn:
        if int(new_start) >= int(new_end):
            st.error("Start muss kleiner als End sein.")
        else:
            spans.append({
                "start": int(new_start),
                "end":   int(new_end),
                "label": new_label,
                "text":  text[int(new_start):int(new_end)],
            })
            spans.sort(key=lambda x: x["start"])
            st.success(f"Span hinzugefügt: **{new_label}** ({int(new_start)}–{int(new_end)})")
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
            highlighted = highlight_text(text, spans)
            st.markdown(
                f"""<div style="height:520px;overflow-y:auto;border:1px solid #e0e0e0;
                border-radius:6px;padding:16px 20px;background:#fafafa;
                font-family:'Georgia',serif;font-size:0.95rem;line-height:2.2;
                white-space:pre-wrap;word-break:break-word;">{highlighted}</div>""",
                unsafe_allow_html=True,
            )

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