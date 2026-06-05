"""
training.py
-----------
Trainiert die AM-Pipeline (einstufig) auf den DARIUS-Daten.

Design-Entscheidungen:
  - Einstufig: DistilBERT klassifiziert alle TAP-Elemente direkt
    (CLAIM, DATA, WARRANT, REBUTTAL). Eine zweite Stufe ist unnötig,
    da der Transformer Erkennung und Klassifikation gleichzeitig löst.
  - Kein Resume: spaCy stellt Optimizer-State nicht her → F1 fällt auf 0.
  - Dev-Split 5%: spaCy evaluiert Dev als einen GPU-Batch → OOM bei >5%.
  - eval_frequency=400: seltener evaluieren reduziert OOM-Risiko.
  - PYTORCH_ALLOC_CONF=expandable_segments:True: weniger Fragmentierung.
  - ZIP-Export: model-best wird nach dem Training gezippt für einfachen
    Download von Kaggle/Colab.

Aufruf:
  python pipeline/training.py         # Training + Evaluation + ZIP
  python pipeline/training.py --eval  # nur Evaluation
  python pipeline/training.py --zip   # nur ZIP (Modell bereits trainiert)
  python pipeline/training.py --cpu   # CPU erzwingen
"""

import os
import sys
import zipfile
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


# ── Pfade ─────────────────────────────────────────────────────────────────────

TRAIN_DATA = Path("data/darius/train.spacy")
DEV_DATA   = Path("data/darius/dev.spacy")
CONFIG     = Path("configs/stage1_claim.cfg")
MODEL_DIR  = Path("models/stage1_claim")
MODEL_BEST = MODEL_DIR / "model-best"
ZIP_DIR    = Path("models")


# ── ZIP-Export ────────────────────────────────────────────────────────────────

def zip_model(model_dir: Path = MODEL_BEST) -> Path | None:
    """
    Zippt model-best in eine einzelne Datei für einfachen Download.
    Enthält nur das Modell selbst — keine Trainingsdaten.

    Gibt den ZIP-Pfad zurück.
    """
    if not model_dir.exists():
        print(f"⚠️  Kein Modell zum Zippen: {model_dir}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_path  = ZIP_DIR / f"am_model_best_{timestamp}.zip"
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Erstelle ZIP: {zip_path}")
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in model_dir.rglob("*"):
            if file.is_file():
                # Pfad im ZIP relativ zu models/ — so kann man direkt
                # nach models/stage1_claim/model-best/ entpacken
                arcname = file.relative_to(ZIP_DIR)
                zf.write(file, arcname)
                file_count += 1

    size_mb = zip_path.stat().st_size / 1e6
    print(f"   {file_count} Dateien · {size_mb:.1f} MB")
    print(f"   → {zip_path}")
    print(f"\n   Lokal entpacken:")
    print(f"   Unzip nach: models/  (erzeugt stage1_claim/model-best/)")

    return zip_path


# ── Voraussetzungen ───────────────────────────────────────────────────────────

def check_prerequisites() -> bool:
    ok = True
    print("🔍 Voraussetzungen prüfen...\n")

    for p in [TRAIN_DATA, DEV_DATA, CONFIG]:
        status = "✅" if p.exists() else "❌"
        print(f"  {status} {p}")
        if not p.exists():
            ok = False

    print()
    for pkg in ["spacy", "spacy_transformers", "torch"]:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}  → pip install {pkg}")
            ok = False

    print()
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  🎮 GPU: {name} ({vram:.1f} GB VRAM)")
        else:
            print("  ⚠️  Keine GPU — CPU-Modus (sehr langsam)")
    except ImportError:
        pass

    if not ok:
        print("\n❌ Bitte fehlende Voraussetzungen installieren.")
    return ok


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(use_gpu: bool = True) -> bool:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(CONFIG),
        "--output", str(MODEL_DIR),
        "--paths.train", str(TRAIN_DATA),
        "--paths.dev",   str(DEV_DATA),
        "--gpu-id", "0" if use_gpu else "-1",
    ]

    print(f"\n{'='*60}")
    print(f"🚀 Training: TAP-Element Detection (einstufig)")
    print(f"   Modell:  distilbert-base-german-cased")
    print(f"   Labels:  CLAIM · DATA · WARRANT · REBUTTAL")
    print(f"   Config:  {CONFIG}")
    print(f"   Output:  {MODEL_DIR}")
    print(f"   GPU:     {'ja' if use_gpu else 'nein (CPU)'}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(Path.cwd()), env=env)
    return result.returncode == 0


def run_evaluation() -> None:
    if not MODEL_BEST.exists():
        print(f"⚠️  Kein Modell: {MODEL_BEST}")
        return

    print(f"\n📊 Evaluation auf Dev-Daten (CPU)")
    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        str(MODEL_BEST),
        str(DEV_DATA),
        "--output", str(MODEL_DIR / "eval_results.json"),
        "--gpu-id", "-1",
    ]
    subprocess.run(cmd, cwd=str(Path.cwd()))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AM-Pipeline Training (einstufig)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--eval", action="store_true",
                        help="Nur Evaluation des vorhandenen Modells")
    parser.add_argument("--zip",  action="store_true",
                        help="Nur ZIP-Export des vorhandenen Modells")
    parser.add_argument("--cpu",  action="store_true",
                        help="CPU erzwingen")
    args = parser.parse_args()

    # Nur ZIP
    if args.zip:
        zip_model()
        return

    # Nur Evaluation
    if args.eval:
        run_evaluation()
        return

    try:
        import torch
        use_gpu = torch.cuda.is_available() and not args.cpu
    except ImportError:
        use_gpu = False

    if not check_prerequisites():
        sys.exit(1)

    # Training
    ok = run_training(use_gpu)
    if not ok:
        print("❌ Training fehlgeschlagen.")
        sys.exit(1)

    # Evaluation
    run_evaluation()

    # ZIP-Export
    zip_path = zip_model()

    print(f"\n{'='*60}")
    print(f"✅ Fertig.")
    print(f"   Modell:    {MODEL_BEST}")
    if zip_path:
        print(f"   ZIP:       {zip_path}  ← dieser Download reicht")
    print(f"\n   Lokal einbinden:")
    print(f"   1. ZIP herunterladen")
    print(f"   2. In Projektordner entpacken → models/stage1_claim/model-best/")
    print(f"   3. streamlit run app/streamlit_app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
