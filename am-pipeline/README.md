# Argumentation Mining Pipeline
### Fachpraktikum NLP-IER · FernUniversität in Hagen

Prototypische Umgebung zum deutschsprachigen Argumentation-Mining  
im Rahmen der historischen Biografie-Forschung (Oral-History.Digital).

---

## Projektstruktur

```
am-pipeline/
├── data/
│   ├── darius/          # DARIUS-Korpus (TSV) + konvertierte spaCy-Dateien
│   └── ohi/             # Oral-History-Interview Testdaten
├── pipeline/
│   ├── preprocessing.py  # TSV → spaCy DocBin Konvertierung
│   ├── model.py          # spaCy-Pipeline + GBERT Integration
│   └── training.py       # Training auf DARIUS
├── app/
│   └── streamlit_app.py  # Evaluierungsumgebung (UI)
├── evaluation/
│   └── metrics.py        # F1, Precision, Recall
├── configs/
│   └── config.cfg        # spaCy Trainingskonfig
└── requirements.txt
```

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

---

## Verwendung

### 1. Daten vorverarbeiten
```powershell
python pipeline/preprocessing.py
```
Konvertiert alle `.tsv`-Dateien in `data/darius/` zu `train.spacy` und `dev.spacy`.

### 2. Evaluierungsumgebung starten
```powershell
streamlit run app/streamlit_app.py
```

---

## TAP-Labels (Toulmin-Argumentation-Pattern)

| Label    | Bedeutung                              |
|----------|----------------------------------------|
| CLAIM    | Behauptung / zentrale These            |
| DATA     | Stützende Fakten / Belege              |
| WARRANT  | Begründung (warum Data → Claim gilt)   |
| REBUTTAL | Einschränkung / Gegenargument          |

---

## Technologie-Stack

- **NLP-Framework:** spaCy 3.x + spacy-transformers
- **Basismodell:** GBERT (deepset/gbert-large)
- **UI:** Streamlit + spacy-streamlit
- **Trainingskorpus:** DARIUS (Schaller et al., 2024)
- **Constraint:** Lokal, Open Source, kein Cloud-Zugriff
