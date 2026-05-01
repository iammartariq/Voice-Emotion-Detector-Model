# pipeline/config.py
# Shared constants and paths for the CEP preprocessing pipeline.

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent   # CEP root
RAW_DIR       = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
EDA_DIR       = BASE_DIR / "dataset" / "eda"
METADATA_CSV  = BASE_DIR / "dataset" / "processed_metadata.csv"
EDA_REPORT    = EDA_DIR / "eda_report.txt"

# ── Audio settings ────────────────────────────────────────────────────────────
TARGET_SR = 16_000   # sample rate (Hz)
TOP_DB    = 30       # silence trim threshold (dB below peak)
MIN_SECS  = 0.5      # discard clips shorter than this after trimming

# ── Label encoding ────────────────────────────────────────────────────────────
LABEL_MAP = {
    "calm":     0,
    "happy":    1,
    "sad":      2,
    "stressed": 3,
    "excited":  4,
    "angry":    5,
}
EMOTIONS  = list(LABEL_MAP.keys())
LANGUAGES = ["english", "urdu"]
