"""
training.py
-----------
Trainiert beide Stufen der AM-Pipeline auf den DARIUS-Daten.

Wichtige Design-Entscheidungen:
  - Kein Resume: spaCy setzt Optimizer-State nicht fort → F1 fällt auf 0.
    Stattdessen: Training läuft in einem Durchgang durch.
  - Dev-Split 5% statt 20%: spaCy evaluiert das gesamte Dev-Set als einen
    Batch → OOM bei großem Dev-Set. 5% reicht für Fortschrittsmessung.
  - eval_frequency=400: seltener evaluieren reduziert OOM-Risiko weiter.
  - PYTORCH_ALLOC_CONF=expandable_segments:True: reduziert Fragmentierung.

Aufruf:
  python pipeline/training.py              # beide Stufen
  python pipeline/training.py --stage 1   # nur Stufe 1
  python pipeline/training.py --stage 2   # nur Stufe 2
  python pipeline/training.py --eval      # nur Evaluation
  python pipeline/training.py --cpu       # CPU erzwingen
"""

import os
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


# ── Voraussetzungen ───────────────────────────────────────────────────────────

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

def run_spacy_train(
    config_path: Path,
    output_dir:  Path,
    stage_name:  str,
    use_gpu:     bool = True,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reduziert GPU-Speicherfragmentierung (empfohlen von PyTorch für T4)
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(TRAIN_DATA),
        "--paths.dev",   str(DEV_DATA),
        "--gpu-id", "0" if use_gpu else "-1",
    ]

    print(f"\n{'='*60}")
    print(f"🚀 Training: {stage_name}")
    print(f"   Config:  {config_path}")
    print(f"   Output:  {output_dir}")
    print(f"   GPU:     {'ja' if use_gpu else 'nein (CPU)'}")
    print(f"\n   Hinweis: kein --resume, da spaCy den Optimizer-State")
    print(f"   nicht wiederherstellt → Training immer von Anfang.")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(Path.cwd()), env=env)
    return result.returncode == 0


def evaluate_model(model_dir: Path, stage_name: str) -> None:
    best = model_dir / "model-best"
    if not best.exists():
        print(f"⚠️  Kein Modell: {best}")
        return

    print(f"\n📊 Evaluation: {stage_name}")

    # Evaluation auf kleinem Subset um OOM zu vermeiden
    # (spaCy evaluiert alles auf einmal im GPU-Speicher)
    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        str(best),
        str(DEV_DATA),
        "--output", str(model_dir / "eval_results.json"),
        "--gpu-id", "-1",   # CPU für Evaluation — vermeidet OOM
    ]
    subprocess.run(cmd, cwd=str(Path.cwd()))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AM-Pipeline Training",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--stage", type=int, choices=[1, 2],
                        help="Nur Stufe 1 oder 2 trainieren")
    parser.add_argument("--eval",  action="store_true",
                        help="Nur Evaluation der vorhandenen Modelle")
    parser.add_argument("--cpu",   action="store_true",
                        help="CPU erzwingen")
    args = parser.parse_args()

    try:
        import torch
        use_gpu = torch.cuda.is_available() and not args.cpu
    except ImportError:
        use_gpu = False

    if args.eval:
        evaluate_model(MODEL_DIR_S1, "Stufe 1 — Claim Detection")
        evaluate_model(MODEL_DIR_S2, "Stufe 2 — TAP Components")
        return

    if not check_prerequisites():
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"AM-Pipeline Training · Modus: {'GPU' if use_gpu else 'CPU'}")
    print(f"{'='*60}")

    def run(config, model_dir, name):
        ok = run_spacy_train(config, model_dir, name, use_gpu)
        if ok:
            evaluate_model(model_dir, name)
        else:
            print(f"❌ {name} fehlgeschlagen.")
            sys.exit(1)

    if args.stage == 1:
        run(CONFIG_S1, MODEL_DIR_S1, "Stufe 1 — Claim Detection")
    elif args.stage == 2:
        run(CONFIG_S2, MODEL_DIR_S2, "Stufe 2 — TAP Components")
    else:
        print("\n💡 Tipp: Beide Stufen können parallel laufen:")
        print("   Terminal 1: python pipeline/training.py --stage 1")
        print("   Terminal 2: python pipeline/training.py --stage 2\n")
        run(CONFIG_S1, MODEL_DIR_S1, "Stufe 1 — Claim Detection")
        run(CONFIG_S2, MODEL_DIR_S2, "Stufe 2 — TAP Components")

    print("\n✅ Training abgeschlossen.")
    print(f"   {MODEL_DIR_S1}/model-best")
    print(f"   {MODEL_DIR_S2}/model-best")


if __name__ == "__main__":
    main()
