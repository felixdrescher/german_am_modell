"""
training.py
-----------
Trainiert die AM-Pipeline (einstufig) auf den DARIUS-Daten.

Design-Entscheidungen:
  - Einstufig: DistilBERT klassifiziert alle TAP-Elemente direkt
    (CLAIM, DATA, WARRANT, REBUTTAL).
  - ZIP-Export: model-best wird nach dem Training gezippt für einfachen
    Download von Kaggle/Colab.

CLI:
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


TRAIN_DATA = Path("data/darius/train.spacy")
DEV_DATA   = Path("data/darius/dev.spacy")
CONFIG     = Path("configs/model.cfg")
MODEL_DIR  = Path("models/spacy_output")
MODEL_BEST = MODEL_DIR / "model-best"
ZIP_DIR    = Path("models")


class Training:
    """
    Kapselt statische Hilfsmethoden für das Modelltraining.

    Ein vollständiger Durchlauf prüft Voraussetzungen, trainiert das Modell,
    evaluiert ``model-best`` und erstellt anschließend ein ZIP-Archiv.

    Attributes:
        Keine: Die Utility-Klasse verwaltet keinen Instanz- oder Klassenzustand.
            Sie verwendet die auf Modulebene definierten Pfade und Konfigurationen.
    """

    @staticmethod
    def zip_model(model_dir: Path = MODEL_BEST) -> Path | None:
        """Packt ein trainiertes Modell als ZIP-Archiv.

        Args:
            model_dir (Path): Verzeichnis des zu archivierenden Modells.

        Returns:
            Path | None: Pfad zum ZIP-Archiv oder ``None`` ohne Modellverzeichnis.

        Raises:
            OSError: Wenn das Archiv nicht erstellt oder geschrieben werden kann.
        """
        if not model_dir.exists():
            print(f"kein Modell zum zippen gefunden unter: {model_dir}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        zip_path  = ZIP_DIR / f"am_model_best_{timestamp}.zip"
        ZIP_DIR.mkdir(parents=True, exist_ok=True)

        print(f"erstelle .zip: {zip_path}")
        file_count = 0

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in model_dir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(ZIP_DIR)
                    zf.write(file, arcname)
                    file_count += 1

        print(f"{file_count} Dateien gezippt")
        print(f"{zip_path}")

        return zip_path


    def check_prerequisites() -> bool:
        """Prüft Dateien, Python-Pakete und die GPU-Verfügbarkeit.

        Returns:
            bool: ``True``, wenn alle notwendigen Dateien und Pakete vorhanden sind.
        """
        ok = True
        print("Voraussetzungen prüfen...")

        for p in [TRAIN_DATA, DEV_DATA, CONFIG]:
            status = "erfolgreich" if p.exists() else "fehlerhaft"
            print(f"{status} {p}")

            if not p.exists():
                ok = False

        print()

        for pkg in ["spacy", "spacy_transformers", "torch"]:
            try:
                __import__(pkg)
                print(f"{pkg} vorhanden")
            except ImportError:
                print(f"{pkg} nicht vorhanden -> pip install {pkg}")
                ok = False

        print()

        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU: {name} ({vram:.1f} GB VRAM)")

        except ImportError:
            pass

        if not ok:
            print("Bitte fehlende Voraussetzungen installieren.")

        return ok


    @staticmethod
    def train(use_gpu: bool = True) -> bool:
        """Startet ein spaCy-Training mit den konfigurierten Daten.

        Args:
            use_gpu (bool): Ob die GPU mit der ID 0 verwendet werden soll.

        Returns:
            bool: ``True``, wenn der Trainingsprozess erfolgreich endet.

        Raises:
            OSError: Wenn der Trainingsprozess nicht gestartet werden kann.
        """
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

        print(f"Config: {CONFIG}")
        print(f"Output: {MODEL_DIR}")
        print(f"GPU: {'ja' if use_gpu else 'nein (CPU)'}")

        result = subprocess.run(cmd, cwd=str(Path.cwd()), env=env)
        return result.returncode == 0

    @staticmethod
    def evaluate() -> None:
        """Führt die spaCy-Evaluation für das beste Modell aus.

        Raises:
            OSError: Wenn der Evaluationsprozess nicht gestartet werden kann.
        """
        if not MODEL_BEST.exists():
            print(f"Kein Modell: {MODEL_BEST}")
            return

        print(f"Evaluation auf Dev-Daten")
        cmd = [
            sys.executable, "-m", "spacy", "evaluate",
            str(MODEL_BEST),
            str(DEV_DATA),
            "--output", str(MODEL_DIR / "eval_results.json"),
            "--gpu-id", "-1",
        ]
        subprocess.run(cmd, cwd=str(Path.cwd()))


def run_training_workflow() -> None:
    """Führt abhängig von den CLI-Argumenten Training, Evaluation oder Export aus.

    Raises:
        SystemExit: Wenn Voraussetzungen fehlen oder das Training fehlschlägt.
        OSError: Wenn ein externer spaCy-Prozess nicht gestartet werden kann.
    """
    parser = argparse.ArgumentParser(
        description="AM-Pipeline Training",
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
        Training.zip_model()
        return

    # Nur Evaluation
    if args.eval:
        Training.evaluate()
        return

    try:
        import torch
        use_gpu = torch.cuda.is_available() and not args.cpu
    except ImportError:
        use_gpu = False

    if not Training.check_prerequisites():
        sys.exit(1)

    # Training
    ok = Training.train(use_gpu)
    if not ok:
        print("Training fehlgeschlagen.")
        sys.exit(1)

    # Evaluation
    Training.evaluate()

    # ZIP-Export
    zip_path = Training.zip_model()

    print(f"\n{'='*60}")
    print(f"Training fertig.")
    print(f"Modell unter: {MODEL_BEST}")

    if zip_path:
        print(f"Modell-ZIP unter: {zip_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    run_training_workflow()
