"""
preprocessor.py
---------------
Kapselt die DARIUS-Vorverarbeitungslogik als Klassen.

Verwendung:
    p = Preprocessor("data/darius", "data/darius/train.spacy", "data/darius/dev.spacy")
    p.run()
"""

import re
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
TAP_RE  = re.compile(r"^(\d+\.\s*.+?)\[(\d+)\]$")
TAP_COL = 19


# ── Datenklassen ──────────────────────────────────────────────────────────────

class TAPAnnotation:
    """Repräsentiert eine TAP-Annotation für ein einzelnes Token.

    Attributes:
        label (Optional[str]): TAP-Label oder ``None`` ohne Annotation.
        arg_id (Optional[str]): DARIUS-Argument-ID oder ``None``.
    """
    def __init__(self, label: Optional[str], arg_id: Optional[str]):
        """Initialisiert eine Annotation für ein Token.

        Args:
            label (Optional[str]): TAP-Label oder ``None`` ohne Annotation.
            arg_id (Optional[str]): DARIUS-Argument-ID oder ``None``.
        """
        self.label  = label    # "CLAIM", "DATA", ... oder None
        self.arg_id = arg_id   # Argument-ID aus DARIUS, z.B. "10"


class Sentence:
    """Repräsentiert einen Satz aus einem DARIUS-Essay.

    Attributes:
        text (str): Zusammengesetzter Text des Satzes.
        tokens (List[str]): Token in ihrer ursprünglichen Reihenfolge.
        offsets (List[Tuple[int, int]]): Dokumentweite Start- und Endoffsets
            der Token.
        tap (List[TAPAnnotation]): Tokenweise TAP-Annotationen.
    """
    def __init__(self, text: str, tokens: List[str],
                 offsets: List[Tuple[int, int]],
                 tap: List[TAPAnnotation]):
        """Initialisiert einen Satz mit Token- und Annotationsdaten.

        Args:
            text (str): Zusammengesetzter Satztext.
            tokens (List[str]): Token des Satzes.
            offsets (List[Tuple[int, int]]): Dokumentweite Token-Offsets.
            tap (List[TAPAnnotation]): Annotationen zu den Token.
        """
        self.text    = text
        self.tokens  = tokens
        self.offsets = offsets   # dokumentweite Zeichenpositionen
        self.tap     = tap


class DARIUSEssay:
    """Repräsentiert ein vollständiges Essay aus dem DARIUS-Korpus.

    Attributes:
        filepath (Path): Pfad der ursprünglichen TSV-Datei.
        filename (str): Aus ``filepath`` abgeleiteter Dateiname.
        sentences (List[Sentence]): Geparste Sätze des Essays.
        sentence_count (int): Berechnete Anzahl der enthaltenen Sätze.
    """
    def __init__(self, filepath: Path, sentences: List[Sentence]):
        """Initialisiert ein DARIUS-Essay.

        Args:
            filepath (Path): Ursprünglicher Pfad der TSV-Datei.
            sentences (List[Sentence]): Geparste Sätze des Essays.
        """
        self.filepath  = filepath
        self.filename  = filepath.name
        self.sentences = sentences

    @property
    def sentence_count(self) -> int:
        """Gibt die Anzahl der Sätze im Essay zurück.

        Returns:
            int: Anzahl der enthaltenen Sätze.
        """
        return len(self.sentences)


class Preprocessor:
    """
    Liest DARIUS WebAnno TSV 3.3 Dateien, extrahiert TAP-Annotationen
    und konvertiert sie in spaCy DocBin-Format für das Modelltraining.

    Attributes:
        input_dir (Path): Verzeichnis mit DARIUS-TSV-Dateien.
        output_train (Path): Ausgabepfad für die Trainingsdaten im spaCy-Format.
        output_dev (Path): Ausgabepfad für die Entwicklungsdaten im spaCy-Format.
        dev_split (float): Anteil der Daten für die Entwicklungsmenge.
        seed (int): Zufallsstartwert für einen reproduzierbaren Datensplit.
        essays (List[DARIUSEssay]): Geladene und geparste Essays.
        label_counts (Dict[str, int]): Anzahl erfolgreich gespeicherter Spans
            je TAP-Label.
    """

    def __init__(self, 
                 input_dir: str, 
                 output_train: str,
                 output_dev: str, 
                 dev_split: float = 0.05, 
                 seed: int = 42):
        """Initialisiert die Vorverarbeitung des DARIUS-Korpus.

        Args:
            input_dir (str): Verzeichnis mit DARIUS-TSV-Dateien.
            output_train (str): Ausgabepfad für die Trainingsdaten.
            output_dev (str): Ausgabepfad für die Entwicklungsdaten.
            dev_split (float): Anteil der Entwicklungsdaten.
            seed (int): Zufallsstartwert für einen reproduzierbaren Split.
        """
        self.input_dir    = Path(input_dir)
        self.output_train = Path(output_train)
        self.output_dev   = Path(output_dev)
        self.dev_split    = dev_split
        self.seed         = seed
        self.essays:        List[DARIUSEssay] = []
        self.label_counts:  Dict[str, int] = defaultdict(int)

    def run(self):
        """Führt die vollständige Vorverarbeitung aus.

        Raises:
            FileNotFoundError: Wenn das Eingabeverzeichnis keine TSV-Dateien enthält.
            ImportError: Wenn spaCy beim Speichern nicht verfügbar ist.
        """
        self._load_essays()
        self._split()
        self._save()

    def _load_essays(self) -> None:
        """Lädt alle .tsv-Dateien aus ``input_dir``.

        Raises:
            FileNotFoundError: Wenn keine TSV-Dateien gefunden werden.
        """
        tsv_files = sorted(self.input_dir.glob("*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"Keine TSV-Dateien in '{self.input_dir}'.")
        print(f"{len(tsv_files)} TSV-Datei(en) gefunden")
        self.essays = []
        for path in tsv_files:
            sentences = self._parse_tsv(path)
            if sentences:
                self.essays.append(DARIUSEssay(path, sentences))
        total = sum(e.sentence_count for e in self.essays)
        print(f"{len(self.essays)} Essays, {total} Sätze geladen")

    def _parse_tsv(self, filepath: Path) -> List[Sentence]:
        """Parst eine DARIUS-WebAnno-TSV-Datei in Sätze.

        Args:
            filepath (Path): Pfad zur zu parsenden TSV-Datei.

        Returns:
            List[Sentence]: Aus der Datei extrahierte Sätze.

        Raises:
            FileNotFoundError: Wenn ``filepath`` nicht existiert.
            UnicodeDecodeError: Wenn die Datei nicht als UTF-8 dekodierbar ist.
        """
        sentences, current_toks = [], []
        content = filepath.read_text(encoding="utf-8")

        def flush(token_lines) -> Optional[Sentence]:
            """Erzeugt aus gesammelten Token-Zeilen einen Satz.

            Args:
                token_lines (List[List[str]]): Spalten der TSV-Tokenzeilen.

            Returns:
                Optional[Sentence]: Erzeugter Satz oder ``None`` ohne Tokens.
            """
            if not token_lines:
                return None
            
            tokens, offsets, tap = [], [], []
            for parts in token_lines:
                try:
                    start, end = map(int, parts[1].split("-"))
                except (ValueError, IndexError):
                    continue

                raw = parts[TAP_COL].strip() if len(parts) > TAP_COL else "_"
                label, arg_id = self._parse_tap_label(raw)
                tokens.append(parts[2])
                offsets.append((start, end))
                tap.append(TAPAnnotation(label, arg_id))

            if not offsets:
                return None
            
            base  = offsets[0][0]
            total = offsets[-1][1] - base
            chars = [" "] * total

            for tok, (s, e) in zip(tokens, offsets):
                for i, ch in enumerate(tok):
                    if s - base + i < total:
                        chars[s - base + i] = ch

            return Sentence("".join(chars), tokens, offsets, tap)

        for raw_line in content.splitlines():
            line = raw_line.rstrip("\r")

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

    def _parse_tap_label(self, raw: str) -> Tuple[Optional[str], Optional[str]]:
        """Extrahiert TAP-Label und Argument-ID aus einem TSV-Feld.

        Args:
            raw (str): Roher Annotationswert aus der TAP-Spalte.

        Returns:
            Tuple[Optional[str], Optional[str]]: Label und Argument-ID oder
                zweimal ``None`` bei keiner gültigen Annotation.
        """
        raw = raw.strip()
        if not raw or raw == "_":
            return None, None
        
        m = TAP_RE.match(raw)
        if not m:
            return None, None
        
        return TAP_LABEL_MAP.get(m.group(1).strip()), m.group(2)

    def _extract_spans(self, sent: Sentence) -> List[Tuple[int, int, str]]:
        """Fasst Token-Annotationen zu Zeichen-Spans zusammen.

        Args:
            sent (Sentence): Satz, dessen Annotationen verarbeitet werden.

        Returns:
            List[Tuple[int, int, str]]: Relative Offsets und zugehörige Labels.
        """
        if not sent.offsets:
            return []
        
        base = sent.offsets[0][0]
        spans = []
        cur_label = cur_id = cur_start = cur_end = None

        for ann, (abs_s, abs_e) in zip(sent.tap, sent.offsets):
            rs, re_ = abs_s - base, abs_e - base

            if ann.label is None:
                if cur_label:
                    spans.append((cur_start, cur_end, cur_label))

                cur_label = cur_id = cur_start = cur_end = None
                continue

            if ann.label != cur_label or ann.arg_id != cur_id:
                if cur_label:
                    spans.append((cur_start, cur_end, cur_label))

                cur_label, cur_id = ann.label, ann.arg_id
                cur_start, cur_end = rs, re_

            else:
                cur_end = re_

        if cur_label:
            spans.append((cur_start, cur_end, cur_label))

        return spans

    def _split(self) -> None:
        """Teilt Essays zufällig in Trainings- und Entwicklungsdaten auf.

        Raises:
            AttributeError: Wenn der erwartete Trainingsdaten-Container fehlt.
        """
        random.seed(self.seed)
        shuffled = self.essays[:]
        random.shuffle(shuffled)
        split = int(len(shuffled) * (1 - self.dev_split))

        train_essays = shuffled[:split]
        dev_essays = shuffled[split:]

        train_sents = [s for e in train_essays for s in e.sentences]
        dev_sents = [s for e in dev_essays   for s in e.sentences]

        print(f"Split: Train={len(train_sents)} Sätze | Dev={len(dev_sents)} Sätze")

    def _save(self) -> None:
        """Speichert Sätze als spaCy-DocBin-Dateien.

        Raises:
            ImportError: Wenn spaCy nicht installiert ist.
            AttributeError: Wenn der erwartete Trainingsdaten-Container fehlt.
            OSError: Wenn die Ausgabedateien nicht geschrieben werden können.
        """
        import spacy
        from spacy.tokens import DocBin

        nlp = spacy.blank("de")
        for sents, path, name in [
            (self.training_data.train_sentences, self.output_train, "Train"),
            (self.training_data.dev_sentences,   self.output_dev,   "Dev"),
        ]:
            db, counts, warns = DocBin(), defaultdict(int), 0
            for sent in sents:
                doc = nlp.make_doc(sent.text)
                spans = self._extract_spans(sent)
                spacy_spans = []

                for s, e, lbl in spans:
                    sp = doc.char_span(s, e, label=lbl, alignment_mode="expand")

                    if sp is not None:
                        spacy_spans.append(sp); counts[lbl] += 1
                        self.label_counts[lbl] += 1

                    else:
                        warns += 1

                doc.spans["sc"] = spacy_spans
                db.add(doc)

            path.parent.mkdir(parents=True, exist_ok=True)
            db.to_disk(str(path))

            print(f"{name}: {sum(counts.values())} Spans → {path.name}")

            if warns:
                print(f"{warns} Spans nicht zuordenbar")


if __name__ == "__main__":
    Preprocessor(
        input_dir = "am-pipeline/data/darius/tsv",
        output_train = "am-pipeline/data/darius/train.spacy",
        output_dev = "am-pipeline/data/darius/dev.spacy",
    ).run()
    print("Preprocessing erledigt.")
