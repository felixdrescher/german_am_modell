"""
model.py
--------
Definiert die zweistufige AM-Pipeline:

  Stufe 1 — Claim Detection (SpanCategorizer)
  Stufe 2 — TAP Component Detection (SpanCategorizer)

Beide nutzen Transformer Backbone.

Zusätzlich:
- Support für "resume-last" über model-last als Initialisierung
"""

from pathlib import Path
from typing import List, Dict, Tuple
import spacy
from spacy.language import Language

# ── Konstanten ────────────────────────────────────────────────────────────────

MODEL_DIR_STAGE1 = Path("models/stage1_claim")
MODEL_DIR_STAGE2 = Path("models/stage2_tap")

CONFIG_DIR = Path("configs")

# ── CONFIG WRITING ────────────────────────────────────────────────────────────

def create_stage1_config() -> str:
    return """
[nlp]
lang = "de"
pipeline = ["transformer", "spancat"]
batch_size = 4
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers": "spacy.Tokenizer.v1"}

[components]

[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "distilbert/distilbert-base-german-cased"
tokenizer_config = {"use_fast": true}
mixed_precision = false

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.spancat]
factory = "spancat"
spans_key = "sc"
threshold = 0.5

[components.spancat.model]
@architectures = "spacy.SpanCategorizer.v1"

[components.spancat.model.reducer]
@layers = "spacy.mean_max_reducer.v1"
hidden_size = 128

[components.spancat.model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO = null
nI = null

[components.spancat.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[components.spancat.model.tok2vec.pooling]
@layers = "reduce_mean.v1"

[components.spancat.suggester]
@misc = "spacy.ngram_range_suggester.v1"
min_size = 1
max_size = 40

[training]
train_corpus = "corpora.train"
dev_corpus = "corpora.dev"
seed = 42
gpu_allocator = "pytorch"
patience = 1600
max_steps = 20000
max_epochs = 30
eval_frequency = 200
dropout = 0.1
accumulate_gradient = 6
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 5e-6

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 50
stop = 500
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = true

[training.score_weights]
spans_sc_f = 1.0
spans_sc_p = 0.0
spans_sc_r = 0.0

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[paths]
train = "data/darius/train.spacy"
dev   = "data/darius/dev.spacy"

[system]
gpu_allocator = "pytorch"
seed = 42

[initialize]
vectors = null
init_tok2vec = null
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.components.spancat]
"""

def create_stage2_config() -> str:
    return """
[nlp]
lang = "de"
pipeline = ["transformer", "spancat"]
batch_size = 4
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers": "spacy.Tokenizer.v1"}

[components]

[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "distilbert/distilbert-base-german-cased"
tokenizer_config = {"use_fast": true}
mixed_precision = true

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 64
stride = 32

[components.spancat]
factory = "spancat"
spans_key = "sc"
threshold = 0.5

[components.spancat.model]
@architectures = "spacy.SpanCategorizer.v1"

[components.spancat.model.reducer]
@layers = "spacy.mean_max_reducer.v1"
hidden_size = 128

[components.spancat.model.scorer]
@layers = "spacy.LinearLogistic.v1"
nO = null
nI = null

[components.spancat.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[components.spancat.model.tok2vec.pooling]
@layers = "reduce_mean.v1"

[components.spancat.suggester]
@misc = "spacy.ngram_suggester.v1"
sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

[training]
train_corpus = "corpora.train"
dev_corpus = "corpora.dev"
seed = 42
gpu_allocator = "pytorch"
patience = 1600
max_steps = 20000
max_epochs = 30
eval_frequency = 200
dropout = 0.1
accumulate_gradient = 3
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 5e-5

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = true

[training.score_weights]
spans_sc_f = 1.0
spans_sc_p = 0.0
spans_sc_r = 0.0

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[paths]
train = "data/darius/train.spacy"
dev   = "data/darius/dev.spacy"

[system]
gpu_allocator = "pytorch"
seed = 42

[initialize]
vectors = null
init_tok2vec = null
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.components.spancat]
"""


def write_configs() -> None:
    CONFIG_DIR.mkdir(exist_ok=True)

    (CONFIG_DIR / "stage1_claim.cfg").write_text(create_stage1_config())
    (CONFIG_DIR / "stage2_tap.cfg").write_text(create_stage2_config())

    print("Standard-Configs geschrieben.")


# ── RESUME CONFIG GENERATION ──────────────────────────────────────────────────

def _inject_resume_path(cfg: str, model_last: Path) -> str:
    """
    spaCy-konforme Warm-Start Injection über init_tok2vec.
    """

    return cfg.replace(
        "init_tok2vec = null",
        f'init_tok2vec = "{model_last.as_posix()}"'
    )


def write_resume_configs() -> None:
    """
    Erzeugt Resume-Configs basierend auf model-last (falls vorhanden).
    """

    CONFIG_DIR.mkdir(exist_ok=True)

    s1_last = MODEL_DIR_STAGE1 / "model-last"
    s2_last = MODEL_DIR_STAGE2 / "model-last"

    if s1_last.exists():
        cfg = _inject_resume_path(create_stage1_config(), s1_last)
        (CONFIG_DIR / "stage1_claim_resume.cfg").write_text(cfg)
        print("Resume-Config Stufe 1 geschrieben")

    if s2_last.exists():
        cfg = _inject_resume_path(create_stage2_config(), s2_last)
        (CONFIG_DIR / "stage2_tap_resume.cfg").write_text(cfg)
        print("Resume-Config Stufe 2 geschrieben")


# ── PIPELINE LOADING ──────────────────────────────────────────────────────────

def load_pipeline(
    stage1_path: Path = MODEL_DIR_STAGE1,
    stage2_path: Path = MODEL_DIR_STAGE2,
) -> Tuple[Language, Language]:

    if not stage1_path.exists():
        raise FileNotFoundError(f"Stufe 1 fehlt: {stage1_path}")
    if not stage2_path.exists():
        raise FileNotFoundError(f"Stufe 2 fehlt: {stage2_path}")

    print(f"Lade Stage 1: {stage1_path}")
    nlp1 = spacy.load(str(stage1_path))

    print(f"Lade Stage 2: {stage2_path}")
    nlp2 = spacy.load(str(stage2_path))

    return nlp1, nlp2


# ── INFERENCE ────────────────────────────────────────────────────────────────

def predict(
    text: str,
    nlp_stage1: Language,
    nlp_stage2: Language,
    threshold1: float = 0.5,
    threshold2: float = 0.5,
) -> List[Dict]:

    result_spans = []

    doc1 = nlp_stage1(text)

    claim_sentences = []

    for sent in doc1.sents:
        for span in doc1.spans.get("sc", []):
            if span.label_ == "CLAIM" and span.start >= sent.start and span.end <= sent.end:
                claim_sentences.append({
                    "text": sent.text,
                    "start_char": sent.start_char
                })
                break

    if not claim_sentences:
        claim_sentences = [
            {"text": s.text, "start_char": s.start_char}
            for s in doc1.sents
        ]

    for sent in claim_sentences:
        doc2 = nlp_stage2(sent["text"])

        for span in doc2.spans.get("sc", []):
            score = getattr(span._, "score", 1.0)

            if score < threshold2:
                continue

            result_spans.append({
                "start": sent["start_char"] + span.start_char,
                "end": sent["start_char"] + span.end_char,
                "label": span.label_,
                "text": span.text,
                "score": round(score, 3),
            })

    return sorted(result_spans, key=lambda x: x["start"])


# ── STUB ─────────────────────────────────────────────────────────────────────

def predict_stub(text: str) -> List[Dict]:
    import re

    patterns = {
        "CLAIM": [r"Ich .*?\.", r"sollte .*?\."] ,
        "DATA": [r"\d+%.*?\."] ,
        "WARRANT": [r"weil .*?\."] ,
        "REBUTTAL": [r"Obwohl .*?\."] ,
    }

    spans = []

    for label, pats in patterns.items():
        for p in pats:
            for m in re.finditer(p, text):
                spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "label": label,
                    "text": m.group(),
                    "score": 0.0
                })

    return sorted(spans, key=lambda x: x["start"])


# ── CLI ENTRY ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--write-configs", action="store_true")
    parser.add_argument("--write-resume-configs", action="store_true")

    args = parser.parse_args()

    if args.write_configs:
        write_configs()

    if args.write_resume_configs:
        write_resume_configs()

    print("model.py fertig ausgeführt")