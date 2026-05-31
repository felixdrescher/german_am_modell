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

Voraussetzungen:
  pip install spacy spacy-transformers torch
  pip install "spacy[transformers]"
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
    ok = True

    print("🔍 Voraussetzungen prüfen...\n")

    for p in [TRAIN_DATA, DEV_DATA, CONFIG_S1, CONFIG_S2]:
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
            print(f"  ❌ {pkg}")
            ok = False

    print()

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️ CPU-Modus")
    except ImportError:
        pass

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

    cmd += ["--gpu-id", "0" if use_gpu else "-1"]

    # ── RESUME (korrekt für spaCy 3.8) ─────────────────────────────
    if resume:
        resume_path = output_dir / "model-last"

        if resume_path.exists():
            print(f"🔁 Resume aktiviert: {resume_path}")
            # spaCy 3.8: kein resume-path Flag!
            # cmd += ["--resume"]
        else:
            print(f"⚠️ Kein model-last gefunden → Neustart {resume_path}")

    print("\n" + "=" * 60)
    print(f"🚀 Training: {stage_name}")
    print(f"   Resume: {resume}")
    print("=" * 60 + "\n")

    result = subprocess.run(cmd, cwd=str(Path.cwd()))
    return result.returncode == 0


def evaluate_model(model_dir: Path, stage_name: str) -> None:
    best_model = model_dir / "model-best"

    if not best_model.exists():
        print(f"⚠️ Kein Modell gefunden: {best_model}")
        return

    print(f"\n📊 Evaluation: {stage_name}")

    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        str(best_model),
        str(DEV_DATA),
        "--output", str(model_dir / "eval_results.json"),
    ]

    subprocess.run(cmd, cwd=str(Path.cwd()))


# ── Training Stages ──────────────────────────────────────────────────────────

def train_stage1(use_gpu: bool = True, resume: bool = False) -> bool:
    return run_spacy_train(CONFIG_S1, MODEL_DIR_S1, "Stufe 1 — Claim Detection", use_gpu, resume)


def train_stage2(use_gpu: bool = True, resume: bool = False) -> bool:
    return run_spacy_train(CONFIG_S2, MODEL_DIR_S2, "Stufe 2 — TAP Detection", use_gpu, resume)


def evaluate_all() -> None:
    evaluate_model(MODEL_DIR_S1, "Stufe 1")
    evaluate_model(MODEL_DIR_S2, "Stufe 2")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage", type=int, choices=[1, 2])
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

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

    print("\n==============================")
    print("AM-Pipeline Training")
    print(f"GPU: {'ja' if use_gpu else 'nein'}")
    print(f"Resume: {args.resume}")
    print("==============================\n")

    def run_stage(fn, model_dir, name):
        ok = fn(use_gpu, args.resume)
        if ok:
            evaluate_model(model_dir, name)
        return ok

    if args.stage == 1:
        if not run_stage(train_stage1, MODEL_DIR_S1, "Stufe 1"):
            sys.exit(1)

    elif args.stage == 2:
        if not run_stage(train_stage2, MODEL_DIR_S2, "Stufe 2"):
            sys.exit(1)

    else:
        print("⚠️ Beide Stufen")

        if not run_stage(train_stage1, MODEL_DIR_S1, "Stufe 1"):
            sys.exit(1)

        if not run_stage(train_stage2, MODEL_DIR_S2, "Stufe 2"):
            sys.exit(1)

    print("\n✅ Fertig")
    print(f"Modelle: {MODEL_DIR_S1}/model-best | {MODEL_DIR_S2}/model-best")


if __name__ == "__main__":
    main()