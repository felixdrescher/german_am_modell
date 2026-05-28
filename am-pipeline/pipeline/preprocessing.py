"""
preprocessing.py
----------------
Liest WebAnno TSV 3.3 Dateien (DARIUS-Format) und konvertiert sie
in spaCy-kompatibles Trainingsformat (DocBin).

DARIUS TAP-Labels:
  claim    – Behauptung
  data     – stützende Daten/Fakten
  warrant  – Begründung/Schlussregel
  rebuttal – Gegenargument
  O        – nicht-argumentativ (outside)
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

import spacy
from spacy.tokens import DocBin, Doc, Span
from spacy.training import Example


# Mapping von DARIUS-Labels auf interne Span-Labels
LABEL_MAP = {
    "claim":    "CLAIM",
    "data":     "DATA",
    "warrant":  "WARRANT",
    "rebuttal": "REBUTTAL",
    "O":        None,   # außerhalb eines Arguments → wird nicht annotiert
}


def parse_webanno_tsv(filepath: str) -> List[Dict]:
    """
    Liest eine WebAnno TSV 3.3 Datei und gibt eine Liste von Sätzen zurück.
    Jeder Satz ist ein Dict mit:
      - 'text': str          (Satztext)
      - 'tokens': List[str]  (Token-Liste)
      - 'labels': List[str]  (Label pro Token, z.B. 'claim', 'O')
      - 'spans': List[Tuple] (start_char, end_char, label) für jede Span
    """
    sentences = []
    current_sentence = None

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Kommentarzeilen überspringen (außer #Text=)
            if line.startswith("#Text="):
                # Neuen Satz beginnen
                text = line[len("#Text="):]
                current_sentence = {
                    "text": text,
                    "tokens": [],
                    "labels": [],
                    "offsets": [],   # (start_char, end_char) pro Token
                }
                sentences.append(current_sentence)
                continue

            if line.startswith("#") or line == "":
                continue

            # Token-Zeile: SatzID-TokenID \t start-end \t token \t label
            parts = line.split("\t")
            if len(parts) < 4:
                continue

            token_id   = parts[0]   # z.B. "1-3"
            char_range = parts[1]   # z.B. "11-12"
            token_text = parts[2]
            raw_label  = parts[3].strip()

            # Label normalisieren (DARIUS nutzt direkt den Label-String)
            label = raw_label if raw_label in LABEL_MAP else "O"

            if current_sentence is not None:
                start, end = map(int, char_range.split("-"))
                # Offset relativ zum Satzanfang berechnen
                # (im echten DARIUS sind Offsets dokumentweit –
                #  hier im Sample sind sie es auch, daher subtrahieren wir
                #  den Offset des ersten Tokens im Satz)
                current_sentence["tokens"].append(token_text)
                current_sentence["labels"].append(label)
                current_sentence["offsets"].append((start, end))

    # Offsets auf Satz-relativen Raum normalisieren
    for sent in sentences:
        if not sent["offsets"]:
            continue
        base = sent["offsets"][0][0]
        sent["offsets"] = [(s - base, e - base) for s, e in sent["offsets"]]

    return sentences


def sentences_to_spacy_spans(
    sentences: List[Dict],
    nlp: spacy.Language
) -> List[Tuple[Doc, List[Tuple[int, int, str]]]]:
    """
    Konvertiert geparste Sätze in (Doc, spans)-Tupel für spaCy.
    Spans werden als (start_char, end_char, label) zurückgegeben.
    """
    results = []

    for sent in sentences:
        text = sent["text"]
        doc = nlp.make_doc(text)

        # Zusammenhängende Token mit gleichem Label zu Spans gruppieren
        spans = []
        current_label = None
        span_start = None
        span_end = None

        for i, (label, (start, end)) in enumerate(
            zip(sent["labels"], sent["offsets"])
        ):
            mapped = LABEL_MAP.get(label)

            if mapped is None:
                # Laufende Span abschließen
                if current_label is not None:
                    spans.append((span_start, span_end, current_label))
                    current_label = None
                continue

            if mapped != current_label:
                # Alte Span abschließen
                if current_label is not None:
                    spans.append((span_start, span_end, current_label))
                # Neue Span beginnen
                current_label = mapped
                span_start = start
                span_end = end
            else:
                # Span verlängern
                span_end = end

        # Letzte offene Span abschließen
        if current_label is not None:
            spans.append((span_start, span_end, current_label))

        results.append((doc, spans))

    return results


def create_docbin(
    spacy_data: List[Tuple[Doc, List[Tuple[int, int, str]]]],
    output_path: str
) -> None:
    """
    Erstellt eine spaCy DocBin-Datei aus den aufbereiteten Daten.
    Die Spans werden unter doc.spans["sc"] gespeichert
    (Span Categorization – das spaCy-Modell für Span-Erkennung).
    """
    db = DocBin()
    nlp = spacy.blank("de")

    for doc, spans in spacy_data:
        spacy_spans = []
        for start_char, end_char, label in spans:
            span = doc.char_span(start_char, end_char, label=label,
                                 alignment_mode="expand")
            if span is not None:
                spacy_spans.append(span)
            else:
                print(f"  [WARNUNG] Span konnte nicht erstellt werden: "
                      f"'{doc.text[start_char:end_char]}' ({label})")

        doc.spans["sc"] = spacy_spans
        db.add(doc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(output_path)
    print(f"DocBin gespeichert: {output_path} ({len(spacy_data)} Dokumente)")


def load_and_convert(
    input_dir: str,
    output_train: str,
    output_dev: str,
    dev_split: float = 0.2
) -> None:
    """
    Hauptfunktion: Liest alle TSV-Dateien aus input_dir,
    splittet in Train/Dev und speichert als spaCy DocBin.
    """
    nlp = spacy.blank("de")
    all_sentences = []

    tsv_files = list(Path(input_dir).glob("*.tsv"))
    if not tsv_files:
        print(f"Keine TSV-Dateien in {input_dir} gefunden.")
        return

    print(f"{len(tsv_files)} TSV-Datei(en) gefunden.")

    for tsv_file in tsv_files:
        print(f"  Lese: {tsv_file.name}")
        sentences = parse_webanno_tsv(str(tsv_file))
        all_sentences.extend(sentences)

    print(f"  Gesamt: {len(all_sentences)} Sätze")

    # Label-Statistik ausgeben
    label_counts: Dict[str, int] = {}
    for sent in all_sentences:
        for label in sent["labels"]:
            mapped = LABEL_MAP.get(label, "O")
            if mapped:
                label_counts[mapped] = label_counts.get(mapped, 0) + 1
    print("  Label-Verteilung:", label_counts)

    # In spaCy-Format konvertieren
    spacy_data = sentences_to_spacy_spans(all_sentences, nlp)

    # Train/Dev-Split
    split_idx = int(len(spacy_data) * (1 - dev_split))
    train_data = spacy_data[:split_idx]
    dev_data   = spacy_data[split_idx:]

    print(f"  Train: {len(train_data)} | Dev: {len(dev_data)}")

    create_docbin(train_data, output_train)
    create_docbin(dev_data,   output_dev)


# ── Direktaufruf zum Testen ──────────────────────────────────────────────────
if __name__ == "__main__":
    load_and_convert(
        input_dir="data/darius",
        output_train="data/darius/train.spacy",
        output_dev="data/darius/dev.spacy",
    )
