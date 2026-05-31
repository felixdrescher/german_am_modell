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
        else:
            print(f"  ⚠️  Keine GPU gefunden — Training auf CPU.")
    except ImportError:
        pass

    if not ok:
        print("\n❌ Bitte fehlende Voraussetzungen installieren.")
    return ok


def run_spacy_train(
    config_path: Path,
    output_dir: Path,
    stage_name: str,
    use_gpu: bool = True,
    resume: bool = False,
) -> bool:

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(TRAIN_DATA),
        "--paths.dev", str(DEV_DATA),
    ]

    if use_gpu:
        cmd += ["--gpu-id", "0"]
    else:
        cmd += ["--gpu-id", "-1"]

    # 👉 RESUME LOGIC
    if resume:
        resume_path = output_dir / "model-last"
        if resume_path.exists():
            cmd += ["--resume-path", str(resume_path)]
            print(f"🔁 Resume from: {resume_path}")
        else:
            print("⚠️ Kein Checkpoint gefunden – starte neu")

    print("\n" + "=" * 60)
    print(f"🚀 Training: {stage_name}")
    print(f"   Resume: {'ja' if resume else 'nein'}")
    print("=" * 60 + "\n")

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
        help="Training auf CPU erzwingen."
    )

    parser.add_argument(
        "--resume", action="store_true",
        help="Training aus model-last fortsetzen."
    )

    args = parser.parse_args()

    # GPU check
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

    print("\n" + "=" * 60)
    print("AM-Pipeline Training")
    print(f"Modus: {'GPU' if use_gpu else 'CPU'}")
    print(f"Resume: {'ja' if args.resume else 'nein'}")
    print("=" * 60)

    def run_stage(stage_fn, model_dir, stage_name):
        try:
            success = stage_fn(use_gpu, resume=args.resume)
            if success:
                evaluate_model(model_dir, stage_name)
            return success
        except Exception as e:
            print(f"❌ Fehler bei {stage_name}: {e}")
            return False

    # -------------------
    # Stage selection
    # -------------------
    if args.stage == 1:
        success = run_stage(train_stage1, MODEL_DIR_S1, "Stufe 1")
        if not success:
            sys.exit(1)

    elif args.stage == 2:
        success = run_stage(train_stage2, MODEL_DIR_S2, "Stufe 2")
        if not success:
            sys.exit(1)

    else:
        print("\n⚠️ Beide Stufen laufen sequentiell\n")

        s1_ok = run_stage(train_stage1, MODEL_DIR_S1, "Stufe 1")
        if not s1_ok:
            print("❌ Stufe 1 fehlgeschlagen — Abbruch.")
            sys.exit(1)

        s2_ok = run_stage(train_stage2, MODEL_DIR_S2, "Stufe 2")
        if not s2_ok:
            print("❌ Stufe 2 fehlgeschlagen.")
            sys.exit(1)

    print("\n✅ Training abgeschlossen.")
    print("Modelle gespeichert in:")
    print(f"   {MODEL_DIR_S1}/model-best")
    print(f"   {MODEL_DIR_S2}/model-best")


if __name__ == "__main__":
    main()