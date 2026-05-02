# Shared constants and paths for the CEP preprocessing pipeline.

from pathlib import Path

# Root paths

BASE_DIR      = Path(__file__).resolve().parent.parent.parent.parent # Since config.py is inside 'Machine Learning CEP/version_02/CODE/pipeline/'

# DATASET Paths

DATASET_DIR   = BASE_DIR / "DATASET"
RAW_DIR       = DATASET_DIR / "raw dataset"
PROCESSED_DIR = DATASET_DIR / "processed dataset"

# METADATA Paths

METADATA_DIR  = DATASET_DIR / "metadata"
METADATA_FILE = METADATA_DIR / "processed_metadata.xlsx"

# EDA Paths

EDA_DIR       = BASE_DIR / "version 02" / "milestone 02 shots" / "eda"
EDA_REPORT    = EDA_DIR / "eda_report.txt"

# Audio settings

TARGET_SR = 16_000   # sample rate (Hz)
TOP_DB    = 30       # silence trim threshold (dB below peak)
MIN_SECS  = 0.5      # discard clips shorter than this after trimming

# Label encoding

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