"""
convert_interviews.py
---------------------
Konvertiert Interview-Dateien (.odt und .csv/.tsv) in strukturierte .txt-Dateien
für das Argumentation-Mining-Modell und die Streamlit-Evaluierungsumgebung.

Verwendung als Python-Modul:
    from data.convert_interviews import convert_file, convert_directory

    # Einzelne Datei konvertieren (erzeugt automatisch eine .txt-Datei)
    txt_path = convert_file("data/ohi/ADG3149_01_01.odt")
    txt_path = convert_file("data/ohi/ADG3149_01_01_de_speaker.csv")

    # Gesamtes Verzeichnis konvertieren:
    converted = convert_directory("data/ohi")

Verwendung über CLI:
    python data/convert_interviews.py
    python data/convert_interviews.py --path data/ohi
"""

import csv
import sys
import zipfile
import argparse
from pathlib import Path
from typing import List, Optional, Union
import xml.etree.ElementTree as ET


# ── ODT-Konvertierung ──────────────────────────────────────────────────────────

def convert_odt(filepath: Union[str, Path]) -> str:
    """Extrahiert den Fließtext aus einer OpenDocument-Textdatei (.odt).

    Berücksichtigt ODF-spezifische Elemente wie Leerzeichen (<text:s>),
    Tabulatoren (<text:tab>) und Zeilenumbrüche (<text:line-break>),
    um korrekte Wort- und Satzabstände sicherzustellen.

    Args:
        filepath: Pfad zur .odt-Datei.

    Returns:
        Extrahierter und absatzweise formatierter Text als String.
    """
    filepath = Path(filepath)
    ns = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"

    def _extract_text_node(elem: ET.Element) -> str:
        """Rekursive Extraktion aller Texte und ODF-Text-Tags."""
        parts = []
        if elem.text:
            parts.append(elem.text)

        for child in elem:
            tag = child.tag
            if tag == f"{{{ns}}}s":
                # text:s = Whitespace, optional mit text:c = Anzahl
                count = int(child.attrib.get(f"{{{ns}}}c", "1"))
                parts.append(" " * count)
            elif tag == f"{{{ns}}}tab":
                parts.append("\t")
            elif tag == f"{{{ns}}}line-break":
                parts.append("\n")
            else:
                parts.append(_extract_text_node(child))

            if child.tail:
                parts.append(child.tail)

        return "".join(parts)

    paragraphs = []
    with zipfile.ZipFile(filepath, "r") as z:
        content_xml = z.read("content.xml")
        root = ET.fromstring(content_xml)

        for elem in root.iter():
            # Paragraphen (<text:p>) und Überschriften (<text:h>)
            if elem.tag in (f"{{{ns}}}p", f"{{{ns}}}h"):
                p_text = _extract_text_node(elem).strip()
                if p_text:
                    paragraphs.append(p_text)

    return "\n\n".join(paragraphs)


# ── CSV/TSV-Konvertierung ─────────────────────────────────────────────────────

def convert_csv(
    filepath: Union[str, Path],
    include_speaker: bool = True,
    merge_consecutive: bool = True,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
) -> str:
    """Konvertiert eine Transkript-CSV/TSV-Datei in formatierten Text.

    Erkennt automatisch typische Spaltennamen wie TRANSCRIPT/TEXT und
    SPEAKER/PERSON und fasst zusammenhängende Sprecherbeiträge übersichtlich
    zusammen.

    Args:
        filepath: Pfad zur CSV/TSV-Datei.
        include_speaker: Wenn True, wird der Sprechername vorangestellt
            (z.B. "SPEAKER_02: ...").
        merge_consecutive: Wenn True, werden aufeinanderfolgende Segmente
            desselben Sprechers zu einem Absatz zusammengeführt.
        delimiter: Optionales Trennzeichen. Bei None wird es automatisch
            ermittelt (Tabulator, Komma oder Semikolon).
        encoding: Zeichenkodierung der CSV-Datei (Standard: 'utf-8').

    Returns:
        Formatierter Transkript-Text als String.
    """
    filepath = Path(filepath)

    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        # Delimiter-Erkennung
        if delimiter is None:
            sample = f.read(4096)
            f.seek(0)
            if "\t" in sample:
                delimiter = "\t"
            elif ";" in sample:
                delimiter = ";"
            else:
                delimiter = ","

        reader = csv.DictReader(f, delimiter=delimiter)
        raw_fieldnames = reader.fieldnames or []
        fieldnames = [col.strip() for col in raw_fieldnames]

        # Spalte für Text/Transkript ermitteln
        transcript_col = None
        for candidate in ["TRANSCRIPT", "transcript", "TEXT", "text", "Content", "content", "utterance"]:
            if candidate in fieldnames:
                transcript_col = candidate
                break
        if not transcript_col and fieldnames:
            transcript_col = fieldnames[-1]

        # Spalte für Sprecher ermitteln
        speaker_col = None
        for candidate in ["SPEAKER", "speaker", "SPEAKER_ID", "speaker_id", "Person", "person"]:
            if candidate in fieldnames:
                speaker_col = candidate
                break

        lines = []
        current_speaker: Optional[str] = None
        current_utterances: List[str] = []

        def _flush_current():
            if current_utterances:
                joined = " ".join(current_utterances).strip()
                if joined:
                    if include_speaker and current_speaker:
                        lines.append(f"{current_speaker}: {joined}")
                    else:
                        lines.append(joined)

        for row in reader:
            text = (row.get(transcript_col) or "").strip()
            if not text:
                continue

            speaker = (row.get(speaker_col) or "").strip() if speaker_col else ""

            if merge_consecutive and speaker:
                if speaker == current_speaker:
                    current_utterances.append(text)
                else:
                    _flush_current()
                    current_speaker = speaker
                    current_utterances = [text]
            else:
                if include_speaker and speaker:
                    lines.append(f"{speaker}: {text}")
                else:
                    lines.append(text)

        if merge_consecutive:
            _flush_current()

    return "\n\n".join(lines)


# ── Universelle Konvertierungsmethode ─────────────────────────────────────────

def convert_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    include_speaker: bool = True,
    merge_consecutive: bool = True,
    encoding: str = "utf-8",
) -> Path:
    """Konvertiert eine .odt- oder .csv/.tsv-Datei in eine .txt-Datei.

    Args:
        input_path: Pfad zur Eingabedatei (.odt, .csv, .tsv).
        output_path: Optionaler Zielpfad der .txt-Datei. Wenn nicht angegeben,
            wird die Endung durch .txt ersetzt.
        include_speaker: Bei CSV: Sprecherbezeichnung beibehalten.
        merge_consecutive: Bei CSV: Aufeinanderfolgende Sätze desselben Sprechers
            zusammenfassen.
        encoding: Ziel-Zeichenkodierung (Standard: 'utf-8').

    Returns:
        Path-Objekt der geschriebenen .txt-Datei.

    Raises:
        ValueError: Falls das Dateiformat nicht unterstützt wird (.odt, .csv, .tsv).
        FileNotFoundError: Falls die Eingabedatei nicht existiert.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".odt":
        text_content = convert_odt(input_path)
    elif suffix in (".csv", ".tsv"):
        text_content = convert_csv(
            input_path,
            include_speaker=include_speaker,
            merge_consecutive=merge_consecutive,
        )
    else:
        raise ValueError(f"Nicht unterstütztes Dateiformat '{suffix}'. Unterstützt: .odt, .csv, .tsv")

    if output_path is None:
        target_path = input_path.with_suffix(".txt")
    else:
        target_path = Path(output_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text_content, encoding=encoding)
    return target_path


def convert_directory(
    dir_path: Union[str, Path] = ".",
    output_dir: Optional[Union[str, Path]] = None,
    formats: tuple = (".odt", ".csv", ".tsv"),
    recursive: bool = True,
    include_speaker: bool = True,
    merge_consecutive: bool = True,
) -> List[Path]:
    """Konvertiert alle unterstützten Interviewdateien in einem Verzeichnis.

    Args:
        dir_path: Quellverzeichnis mit den Eingabedateien.
        output_dir: Optionales Zielverzeichnis. Bei None werden .txt-Dateien
            im selben Verzeichnis wie die Quelldatei abgelegt.
        formats: Tuple der zu konvertierenden Dateiendungen.
        recursive: Ob Unterverzeichnisse rekursiv durchsucht werden sollen.
        include_speaker: Bei CSV: Sprecherbezeichnung beibehalten.
        merge_consecutive: Bei CSV: Sprechersegmente zusammenfassen.

    Returns:
        Liste aller erzeugten .txt-Pfade.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Verzeichnis nicht gefunden: {dir_path}")

    converted_files: List[Path] = []
    pattern = "**/*" if recursive else "*"

    for file_path in sorted(dir_path.glob(pattern)):
        if file_path.is_file() and file_path.suffix.lower() in formats:
            if output_dir is not None:
                out_path = Path(output_dir) / f"{file_path.stem}.txt"
            else:
                out_path = None

            try:
                res = convert_file(
                    input_path=file_path,
                    output_path=out_path,
                    include_speaker=include_speaker,
                    merge_consecutive=merge_consecutive,
                )
                converted_files.append(res)
            except Exception as e:
                print(f"Fehler bei {file_path.name}: {e}", file=sys.stderr)

    return converted_files


# ── CLI-Einstiegspunkt ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Konvertiert .odt- und .csv-Interviews in .txt-Dateien."
    )
    parser.add_argument(
        "--path",
        "-p",
        default=str(Path(__file__).parent / "ohi"),
        help="Pfad zu einer Datei oder einem Verzeichnis (Standard: data/ohi)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optionaler Zielpfad oder Zielverzeichnis",
    )
    parser.add_argument(
        "--no-speaker",
        action="store_true",
        help="Sprecherbezeichnungen bei CSV nicht mit in die Textdatei schreiben",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Aufeinanderfolgende Sprechersegmente nicht zusammenführen",
    )

    args = parser.parse_args()
    target_path = Path(args.path)

    if not target_path.exists():
        # Falls relativ zum data-Ordner
        fallback = Path(__file__).parent / args.path
        if fallback.exists():
            target_path = fallback
        else:
            print(f"Pfad nicht gefunden: {args.path}", file=sys.stderr)
            sys.exit(1)

    if target_path.is_file():
        out = convert_file(
            target_path,
            output_path=args.output,
            include_speaker=not args.no_speaker,
            merge_consecutive=not args.no_merge,
        )
        print(f"Erfolgreich konvertiert: {out}")
    else:
        results = convert_directory(
            target_path,
            output_dir=args.output,
            include_speaker=not args.no_speaker,
            merge_consecutive=not args.no_merge,
        )
        print(f"{len(results)} Datei(en) erfolgreich konvertiert in '{target_path}':")
        for res in results:
            print(f"  - {res.name}")


if __name__ == "__main__":
    main()
