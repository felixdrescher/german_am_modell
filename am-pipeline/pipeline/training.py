"""
training.py
-----------
Trainiert beide Stufen der AM-Pipeline auf den DARIUS-Daten.

Ablauf:
  1. Voraussetzungen prüfen (Daten, Packages, GPU)
  2. Stufe 1 trainieren: Claim Detection
  3. Stufe 2 trainieren: TAP Component Detection
  4. Evaluation beider Modelle auf Dev-Daten

Aufruf:
  python pipeline/training.py              # beide Stufen
  python pipeline/training.py --stage 1   # nur Stufe 1
  python pipeline/training.py --stage 2   # nur Stufe 2
  python pipeline/training.py --eval      # nur Evaluation

Voraussetzungen (einmalig installieren):
  pip install spacy spacy-transformers torch
  pip install "spacy[transformers]"
  python -m spacy download de_core_news_sm
"""

import sys
import subprocess
import argparse
from pathlib import Path


# ── Pfade ─────────────────────────────────────────────────────────────────────

TRAIN_DATA   = Path("data/darius/train.spacy")
DEV_DATA     = Path("data/darius/dev.spacy")
CONFIG_S1    = Path("configs/stage1_claim.cfg")
CONFIG_S2    = Path("configs/stage2_tap.cfg")
MODEL_DIR_S1 = Path("models/stage1_claim")
MODEL_DIR_S2 = Path("models/stage2_tap")


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def check_prerequisites() -> bool:
    """Prüft ob alle nötigen Dateien und Packages vorhanden sind."""
    ok = True

    print("🔍 Voraussetzungen prüfen...\n")

    # Daten
    for p in [TRAIN_DATA, DEV_DATA, CONFIG_S1, CONFIG_S2]:
        status = "✅" if p.exists() else "❌"
        print(f"  {status} {p}")
        if not p.exists():
            ok = False

    # Packages
    print()
    for pkg in ["spacy", "spacy_transformers", "torch"]:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}  → pip install {pkg}")
            ok = False

    # GPU
    print()
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  🎮 GPU: {gpu} ({vram:.1f} GB VRAM)")
            print(f"     GBERT-large benötigt ~6 GB VRAM.")
            print(f"     Falls nicht genug VRAM: in config.cfg"
                  f" 'deepset/gbert-large' → 'deepset/gbert-base' ändern.")
        else:
            print(f"  ⚠️  Keine GPU gefunden — Training auf CPU.")
            print(f"     Das kann bei GBERT-large sehr lange dauern (Stunden).")
            print(f"     Empfehlung: in config.cfg auf 'deepset/gbert-base' wechseln.")
    except ImportError:
        pass

    if not ok:
        print("\n❌ Bitte fehlende Voraussetzungen installieren.")
    return ok


def run_spacy_train(
    config_path: Path,
    output_dir:  Path,
    stage_name:  str,
    use_gpu:     bool = True,
) -> bool:
    """
    Startet spaCy-Training via subprocess.
    Gibt True zurück wenn erfolgreich.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_flag = "--gpu-id 0" if use_gpu else "--gpu-id -1"

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(TRAIN_DATA),
        "--paths.dev",   str(DEV_DATA),
    ]
    if use_gpu:
        cmd += ["--gpu-id", "0"]
    else:
        cmd += ["--gpu-id", "-1"]

    print(f"\n{'='*60}")
    print(f"🚀 Starte Training: {stage_name}")
    print(f"   Config:  {config_path}")
    print(f"   Output:  {output_dir}")
    print(f"   GPU:     {'ja' if use_gpu else 'nein (CPU)'}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(Path.cwd()))
    return result.returncode == 0


def evaluate_model(model_dir: Path, stage_name: str) -> None:
    """Evaluiert ein trainiertes Modell auf den Dev-Daten."""
    best_model = model_dir / "model-best"
    if not best_model.exists():
        print(f"⚠️  Kein trainiertes Modell in {best_model}")
        return

    print(f"\n📊 Evaluation: {stage_name}")
    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        str(best_model),
        str(DEV_DATA),
        "--output", str(model_dir / "eval_results.json"),
    ]
    subprocess.run(cmd, cwd=str(Path.cwd()))


# ── Haupt-Training ────────────────────────────────────────────────────────────

def train_stage1(use_gpu: bool = True) -> bool:
    """Stufe 1: Claim Detection."""
    return run_spacy_train(CONFIG_S1, MODEL_DIR_S1, "Stufe 1 — Claim Detection", use_gpu)


def train_stage2(use_gpu: bool = True) -> bool:
    """Stufe 2: TAP Component Detection."""
    return run_spacy_train(CONFIG_S2, MODEL_DIR_S2, "Stufe 2 — TAP Components", use_gpu)


def evaluate_all() -> None:
    evaluate_model(MODEL_DIR_S1, "Stufe 1 — Claim Detection")
    evaluate_model(MODEL_DIR_S2, "Stufe 2 — TAP Components")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AM-Pipeline Training auf DARIUS-Daten"
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2],
        help="Nur eine Stufe trainieren (1 oder 2). Standard: beide."
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Nur Evaluation der vorhandenen Modelle."
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Training auf CPU erzwingen (langsamer, kein VRAM nötig)."
    )
    args = parser.parse_args()

    # GPU-Verfügbarkeit prüfen
    try:
        import torch
        use_gpu = torch.cuda.is_available() and not args.cpu
    except ImportError:
        use_gpu = False

    if args.eval:
        evaluate_all()
        return

    if not check_prerequisites():
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"AM-Pipeline Training")
    print(f"Modus: {'GPU' if use_gpu else 'CPU'}")
    print(f"{'='*60}")

    if args.stage == 1:
        try:
            success = train_stage1(use_gpu)
            if success:
                evaluate_model(MODEL_DIR_S1, "Stufe 1")
        except Exception as e:
            print(f"❌ Fehler beim Training von Stufe 1: {e}")
    elif args.stage == 2:
        try:
            success = train_stage2(use_gpu)
            if success:
                
                evaluate_model(MODEL_DIR_S2, "Stufe 2")
        except Exception as e:
            print(f"❌ Fehler beim Training von Stufe 2: {e}")
    else:
        # Beide Stufen
        print("\n⚠️  Hinweis: Stufe 1 und Stufe 2 sind voneinander unabhängig")
        print("   und können parallel trainiert werden (zwei Terminals):\n")
        print("   Terminal 1:  python pipeline/training.py --stage 1")
        print("   Terminal 2:  python pipeline/training.py --stage 2\n")

        s1_ok = train_stage1(use_gpu)
        if s1_ok:
            evaluate_model(MODEL_DIR_S1, "Stufe 1")
        else:
            print("❌ Stufe 1 fehlgeschlagen — breche ab.")
            sys.exit(1)

        s2_ok = train_stage2(use_gpu)
        if s2_ok:
            evaluate_model(MODEL_DIR_S2, "Stufe 2")
        else:
            print("❌ Stufe 2 fehlgeschlagen.")
            sys.exit(1)

    print("\n✅ Training abgeschlossen.")
    print(f"   Modelle gespeichert in:")
    print(f"   {MODEL_DIR_S1}/model-best")
    print(f"   {MODEL_DIR_S2}/model-best")
    print(f"\n   Streamlit-App starten:")
    print(f"   streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()