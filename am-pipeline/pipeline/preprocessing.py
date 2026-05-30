"""
preprocessing.py
----------------
Liest WebAnno TSV 3.3 Dateien im echten DARIUS-Format und konvertiert sie
in spaCy-kompatibles Trainingsformat (DocBin).

Besonderheiten des echten DARIUS-Formats:
  - Keine #Text= Kommentarzeilen — Sätze durch Leerzeilen getrennt
  - Offsets dokumentweit (nicht satzweise)
  - 21 Spalten, letzte leer (trailing Tab) → TAP-Label fix in Spalte 19
  - Windows-Zeilenenden (\r\n)

TAP-Labels (Spalte 19):
  1. Claim[N]            → CLAIM
  2. Data[N]             → DATA
  3. Warrant[N]          → WARRANT
  4. Rebuttal[N]         → REBUTTAL
  5. Nicht zutreffend[N] → O (nicht annotieren)
  _                      → O
"""

import re
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


TAP_LABEL_MAP = {
    "1. Claim":            "CLAIM",
    "2. Data":             "DATA",
    "3. Warrant":          "WARRANT",
    "4. Rebuttal":         "REBUTTAL",
    "5. Nicht zutreffend": None,
}

TAP_RE       = re.compile(r"^(\d+\.\s*.+?)\[(\d+)\]$")
TAP_COL      = 19   # fester Spaltenindex für TAPArgumente


def parse_tap_label(raw: str) -> Tuple[Optional[str], Optional[str]]:
    raw = raw.strip()
    if not raw or raw == "_":
        return None, None
    m = TAP_RE.match(raw)
    if not m:
        return None, None
    return TAP_LABEL_MAP.get(m.group(1).strip()), m.group(2)


def parse_webanno_tsv(filepath: str) -> List[Dict]:
    """
    Parst eine DARIUS TSV-Datei.
    Gibt Liste von Satz-Dicts zurück:
      text    – rekonstruierter Text
      tokens  – Token-Texte
      offsets – (start, end) dokumentweit
      tap     – (label|None, arg_id|None) pro Token
    """
    sentences    = []
    current_toks = []

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    def flush(token_lines):
        if not token_lines:
            return None
        tokens, offsets, tap = [], [], []
        for parts in token_lines:
            try:
                start, end = map(int, parts[1].split("-"))
            except (ValueError, IndexError):
                continue
            tap_raw = parts[TAP_COL].strip() if len(parts) > TAP_COL else "_"
            label, arg_id = parse_tap_label(tap_raw)
            tokens.append(parts[2])
            offsets.append((start, end))
            tap.append((label, arg_id))
        if not offsets:
            return None
        # Text aus Token + Offsets rekonstruieren
        base  = offsets[0][0]
        total = offsets[-1][1] - base
        chars = [" "] * total
        for tok, (s, e) in zip(tokens, offsets):
            for i, ch in enumerate(tok):
                if s - base + i < total:
                    chars[s - base + i] = ch
        return {
            "text":    "".join(chars),
            "tokens":  tokens,
            "offsets": offsets,
            "tap":     tap,
        }

    for raw_line in content.splitlines():
        line = raw_line.rstrip("\r")   # Windows CR
        if line.startswith("#"):
            continue
        if line.strip() == "":
            sent = flush(current_toks)
            if sent:
                sentences.append(sent)
            current_toks = []
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and "-" in parts[0]:
            current_toks.append(parts)

    sent = flush(current_toks)
    if sent:
        sentences.append(sent)

    return sentences


def extract_spans(sent: Dict) -> List[Tuple[int, int, str]]:
    """
    Extrahiert TAP-Spans, Offsets satz-relativ normalisiert.
    Neue Span bei Label- oder Argument-ID-Wechsel.
    """
    if not sent["offsets"]:
        return []
    base = sent["offsets"][0][0]
    spans                          = []
    cur_label = cur_id = None
    cur_start = cur_end = None

    for (label, arg_id), (abs_s, abs_e) in zip(sent["tap"], sent["offsets"]):
        rs, re_ = abs_s - base, abs_e - base
        if label is None:
            if cur_label:
                spans.append((cur_start, cur_end, cur_label))
            cur_label = cur_id = cur_start = cur_end = None
            continue
        if label != cur_label or arg_id != cur_id:
            if cur_label:
                spans.append((cur_start, cur_end, cur_label))
            cur_label, cur_id, cur_start, cur_end = label, arg_id, rs, re_
        else:
            cur_end = re_

    if cur_label:
        spans.append((cur_start, cur_end, cur_label))
    return spans


def sentences_to_docbin(sentences: List[Dict], output_path: str) -> None:
    import spacy
    from spacy.tokens import DocBin

    nlp    = spacy.blank("de")
    db     = DocBin()
    counts = defaultdict(int)
    warns  = 0

    for sent in sentences:
        doc   = nlp.make_doc(sent["text"])
        spans = extract_spans(sent)
        spacy_spans = []
        for s, e, lbl in spans:
            sp = doc.char_span(s, e, label=lbl, alignment_mode="expand")
            if sp is not None:
                spacy_spans.append(sp)
                counts[lbl] += 1
            else:
                warns += 1
        doc.spans["sc"] = spacy_spans
        db.add(doc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    db.to_disk(output_path)
    print(f"  → {Path(output_path).name}: {len(sentences)} Sätze, "
          f"{sum(counts.values())} Spans")
    for lbl in ["CLAIM", "DATA", "WARRANT", "REBUTTAL"]:
        if counts[lbl]:
            print(f"     {lbl:10s}: {counts[lbl]:4d}")
    if warns:
        print(f"  ⚠️  {warns} Spans nicht zuordenbar")


def load_and_convert(
    input_dir:    str,
    output_train: str,
    output_dev:   str,
    dev_split:    float = 0.2,
    shuffle:      bool  = True,
    seed:         int   = 42,
) -> None:
    tsv_files = sorted(Path(input_dir).glob("*.tsv"))
    if not tsv_files:
        print(os.getcwd())
        print(f"❌ Keine TSV-Dateien in '{input_dir}'.")
        return

    print(f"📂 {len(tsv_files)} TSV-Datei(en) gefunden")
    essays = []
    for f in tsv_files:
        sents = parse_webanno_tsv(str(f))
        if sents:
            essays.append(sents)

    print(f"   {len(essays)} Essays, "
          f"{sum(len(e) for e in essays)} Sätze gesamt\n")

    if shuffle:
        random.seed(seed)
        random.shuffle(essays)

    split       = int(len(essays) * (1 - dev_split))
    train_sents = [s for e in essays[:split] for s in e]
    dev_sents   = [s for e in essays[split:] for s in e]

    print(f"📊 Split (Essay-Ebene):")
    print(f"   Train: {len(essays[:split])} Essays, {len(train_sents)} Sätze")
    print(f"   Dev:   {len(essays[split:])} Essays, {len(dev_sents)} Sätze\n")

    print("💾 Train:")
    sentences_to_docbin(train_sents, output_train)
    print("\n💾 Dev:")
    sentences_to_docbin(dev_sents, output_dev)
    print("\n✅ Fertig.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        print(f"\n🔍 Debug: {sys.argv[1]}\n")
        sents = parse_webanno_tsv(sys.argv[1])
        print(f"Sätze: {len(sents)}\n")
        for i, s in enumerate(sents):
            spans = extract_spans(s)
            if not spans:
                continue
            print(f"Satz {i+1}: \"{s['text'][:65]}\"")
            for start, end, lbl in spans:
                print(f"  [{lbl}] \"{s['text'][start:end]}\"")
    else:
        load_and_convert(
            input_dir    = "data/darius/tsv",
            output_train = "data/darius/train.spacy",
            output_dev   = "data/darius/dev.spacy",
        )
