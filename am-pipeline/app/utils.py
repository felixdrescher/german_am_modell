from datetime import datetime
from typing import Dict
from pathlib import Path


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