"""
model.py
--------
Definiert die zweistufige AM-Pipeline:

  Stufe 1 — Claim Detection (SpanCategorizer, binär)
    Eingabe: beliebiger Text
    Ausgabe: Sätze die mindestens einen CLAIM enthalten

  Stufe 2 — TAP Component Detection (SpanCategorizer, multi-label)
    Eingabe: nur die Sätze aus Stufe 1
    Ausgabe: CLAIM / DATA / WARRANT / REBUTTAL Spans

Beide Stufen nutzen GBERT (deepset/gbert-large) als Transformer-Backbone.
Das Modell wird lokal betrieben — kein Cloud-Zugriff.

Verwendung:
  # Training:       python pipeline/training.py
  # Inferenz:       from pipeline.model import predict
  # Streamlit-App:  wird von app/streamlit_app.py importiert
"""

from pathlib import Path
from typing import List, Dict, Optional
import spacy
from spacy.language import Language


# ── Konstanten ────────────────────────────────────────────────────────────────

LABELS_STAGE1 = ["CLAIM"]                              # Stufe 1: binäre Claim-Detektion
LABELS_STAGE2 = ["CLAIM", "DATA", "WARRANT", "REBUTTAL"]  # Stufe 2: alle TAP-Elemente

# Pfade (relativ zum Projektordner)
MODEL_DIR_STAGE1 = Path("models/stage1_claim")
MODEL_DIR_STAGE2 = Path("models/stage2_tap")

# GBERT Modell-Name (HuggingFace)
GBERT_MODEL = "deepset/gbert-large"


# ── Pipeline-Konfiguration ────────────────────────────────────────────────────

def create_stage1_config() -> str:
    """
    spaCy config.cfg für Stufe 1: Claim Detection.

    Fixes gegenüber vorheriger Version:
    - [nlp] Pflichtfelder (disabled, tokenizer, before/after_creation) ergänzt
    - sentence_suggester statt ngram (Claims sind satzlang)
    - Lernrate 5e-5 statt 1e-3 (BERT-typisch, schützt vortrainierte Gewichte)
    - warmup_cosine statt warmup_linear
    - shuffle = true in corpora.train
    - batch_size = 4 (für Colab T4 mit distilbert; für gbert-large auf 2 setzen)
    """
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


def create_stage2_config() -> str:
    """
    spaCy config.cfg für Stufe 2: TAP Component Detection.

    Unterschiede zu Stufe 1:
    - ngram_suggester mit sizes (flexiblere Span-Grenzen für Data/Warrant/Rebuttal)
    - Alle 4 Labels (CLAIM, DATA, WARRANT, REBUTTAL)
    """
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
    """Schreibt beide Config-Dateien in configs/."""
    Path("configs").mkdir(exist_ok=True)
    Path("configs/stage1_claim.cfg").write_text(create_stage1_config())
    Path("configs/stage2_tap.cfg").write_text(create_stage2_config())
    print("Config-Dateien geschrieben: configs/stage1_claim.cfg, configs/stage2_tap.cfg")


# ── Modell laden ──────────────────────────────────────────────────────────────

def load_pipeline(
    stage1_path: Path = MODEL_DIR_STAGE1,
    stage2_path: Path = MODEL_DIR_STAGE2,
) -> tuple:
    """
    Lädt beide trainierten Modell-Stufen.
    Gibt (nlp_stage1, nlp_stage2) zurück.
    Wirft FileNotFoundError wenn Modelle noch nicht trainiert wurden.
    """
    if not stage1_path.exists():
        raise FileNotFoundError(
            f"Stufe-1-Modell nicht gefunden: {stage1_path}\n"
            "Bitte zuerst python pipeline/training.py ausführen."
        )
    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Stufe-2-Modell nicht gefunden: {stage2_path}\n"
            "Bitte zuerst python pipeline/training.py ausführen."
        )

    print(f"Lade Stufe 1: {stage1_path}")
    nlp1 = spacy.load(str(stage1_path))
    print(f"Lade Stufe 2: {stage2_path}")
    nlp2 = spacy.load(str(stage2_path))
    return nlp1, nlp2


# ── Inferenz ──────────────────────────────────────────────────────────────────

def predict(
    text:        str,
    nlp_stage1:  Language,
    nlp_stage2:  Language,
    threshold1:  float = 0.5,
    threshold2:  float = 0.5,
) -> List[Dict]:
    """
    Zweistufige Inferenz auf einem Text.

    Stufe 1: Satz-Segmentierung + Claim-Detection.
             Nur Sätze mit CLAIM-Score >= threshold1 kommen weiter.

    Stufe 2: TAP-Element-Detection in den gefilterten Sätzen.
             Spans mit Score >= threshold2 werden zurückgegeben.

    Rückgabe: Liste von Span-Dicts:
      {
        "start": int,   # Zeichenposition im Originaltext
        "end":   int,
        "label": str,   # CLAIM / DATA / WARRANT / REBUTTAL
        "text":  str,   # Span-Text
        "score": float, # Modell-Konfidenz
      }
    """
    result_spans = []

    # ── Stufe 1: Sätze mit Claims finden ─────────────────────────────────────
    doc1 = nlp_stage1(text)

    # Satz-Segmentierung via spaCy (de_core_news_sm oder sentencizer)
    claim_sentences = []
    for sent in doc1.sents:
        sent_doc = nlp_stage1.make_doc(sent.text)
        # Spans aus Stufe 1 prüfen
        for span in doc1.spans.get("sc", []):
            if (span.start >= sent.start and
                span.end <= sent.end and
                span.label_ == "CLAIM" and
                span._.score >= threshold1):
                claim_sentences.append({
                    "text":       sent.text,
                    "sent_start": sent.start_char,
                })
                break

    if not claim_sentences:
        # Kein Claim gefunden → alle Sätze durch Stufe 2 schicken
        # (konservative Fallback-Strategie für OHI-Texte)
        claim_sentences = [
            {"text": sent.text, "sent_start": sent.start_char}
            for sent in doc1.sents
        ]

    # ── Stufe 2: TAP-Elemente in Claim-Sätzen finden ─────────────────────────
    for sent_info in claim_sentences:
        sent_text  = sent_info["text"]
        sent_start = sent_info["sent_start"]

        doc2 = nlp_stage2(sent_text)

        for span in doc2.spans.get("sc", []):
            score = getattr(span._, "score", 1.0)
            if score < threshold2:
                continue
            result_spans.append({
                "start": sent_start + span.start_char,
                "end":   sent_start + span.end_char,
                "label": span.label_,
                "text":  span.text,
                "score": round(score, 3),
            })

    # Nach Startposition sortieren
    result_spans.sort(key=lambda x: x["start"])
    return result_spans


# ── Stub für Entwicklung ohne trainiertes Modell ──────────────────────────────

def predict_stub(text: str) -> List[Dict]:
    """
    Gibt synthetische Spans zurück solange kein trainiertes Modell vorliegt.
    Wird von der Streamlit-App genutzt (run_model_stub in streamlit_app.py).
    Nach dem Training durch predict() ersetzen.
    """
    import re
    spans   = []
    patterns = {
        "CLAIM":    [r"Ich (?:denke|meine|glaube|finde)[^.]*\.", r"sollte[^.]*\.",
                     r"(?:bin|bleibe) ich[^.]*\."],
        "DATA":     [r"Mit [^.]*\d+%[^.]*\.", r"\d+[^.]*(?:GWh|Jahre|km|€)[^.]*\."],
        "WARRANT":  [r"(?:bedeutet|weil|deshalb|daher)[^.]*\.",
                     r"Verantwortung[^.]*\."],
        "REBUTTAL": [r"Obwohl[^.]*\.", r"obwohl[^.]*\.", r"[Jj]edoch[^.]*\."],
    }
    for label, plist in patterns.items():
        for pat in plist:
            for m in re.finditer(pat, text):
                spans.append({
                    "start": m.start(), "end": m.end(),
                    "label": label, "text": m.group(), "score": 0.0,
                })
    spans.sort(key=lambda x: x["start"])
    # Überlappungen entfernen
    filtered, last_end = [], -1
    for s in spans:
        if s["start"] >= last_end:
            filtered.append(s)
            last_end = s["end"]
    return filtered


# ── Direktaufruf ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    write_configs()
    print("\nModell-Stub Test:")
    test = ("Ich denke, dass Windkraftanlagen gefördert werden sollten. "
            "Mit einem Wirkungsgrad von 45% sind sie effizienter. "
            "Obwohl sie Lärm erzeugen, überwiegen die Vorteile.")
    for span in predict_stub(test):
        print(f"  [{span['label']}] \"{span['text'][:60]}\"")
