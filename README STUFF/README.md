# CEP — Cross-lingual Emotion Prediction
## Speech Emotion Recognition Dataset & Preprocessing Pipeline

A complete ML data pipeline for **speech emotion recognition** across **2 languages** (English & Urdu) and **6 emotion classes** (calm · happy · sad · stressed · excited · angry).

---

## 📁 Project Structure

```
CEP/
├── dataset/
│   ├── raw/<language>/<emotion>/  ← raw user .webm recordings
│   ├── processed/<lang>/<emo>/    ← cleanly named .wav files
│   ├── eda/                       ← EDA charts, waveforms, and stats
│   └── processed_metadata.csv     ← training index
│
├── pipeline/
│   ├── config.py
│   ├── preprocess.py
│   ├── augment.py
│   ├── metadata.py
│   └── eda.py
│
├── main.py
└── requirements.txt
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> **ffmpeg required** for `.webm → .wav` conversion (Step 1).
> Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

---

## 🚀 Pipeline — Step by Step

Run from the `CEP/` root directory to execute all steps sequentially:

```bash
python main.py
```

### ✅ Step 1 — Preprocess
For each `.webm` in `dataset/raw`:
1. **Convert** → WAV (16 kHz, mono) via ffmpeg/pydub
2. **Normalize** amplitude to `[-1, 1]`
3. **Trim** leading/trailing silence (top_db = 30)
4. Saves explicitly to `dataset/processed/<language>/<emotion>/001.wav` allowing clean tracking without namespace collisions.

### ✅ Step 2 — Augmentation (Class Balancing)
Automatically balances under-represented classes using augmentation to match the majority class.
  | Technique | Parameter | Filename Suffix |
  |-----------|-----------|-----------------|
  | Pitch shift up | +2 semitones | `_ps+2` |
  | Pitch shift down | −2 semitones | `_ps-2` |
  | Time stretch fast | rate = 1.1× | `_tsf` |
  | Time stretch slow | rate = 0.9× | `_tss` |
  | Gaussian noise | σ = 0.005 | `_ns` |

### ✅ Step 3 — Build Metadata CSV
Scans `processed/` and writes `processed_metadata.csv` with granular tracking:
- `language, emotion, label (calm=0...angry=5)`
- `duration` (in seconds)
- `aug_type` (original, ps+2, tsf, etc.)

### ✅ Step 4 — Exploratory Data Analysis
Generates rich acoustic and statistical reports in `dataset/eda/`:
- `report.txt` with tables showing %, durations, and balance ratios.
- `waveforms.png` and `spectrograms.png` (acoustic grids covering all classes).
- `durations_boxplot.png` mapping durations by emotion.
- `class_distribution.png` and `orig_vs_aug.png` (augmentations stacked chart).

---

## 🏷️ Label Encoding

| Emotion | Label |
|---------|-------|
| calm | 0 |
| happy | 1 |
| sad | 2 |
| stressed | 3 |
| excited | 4 |
| angry | 5 |
