# pipeline/metadata.py
# Step 3 — Scan processed/ and write processed_metadata.csv (training index).

import pandas as pd
import soundfile as sf
from pathlib import Path

from .config import (
    BASE_DIR, PROCESSED_DIR, METADATA_CSV,
    LANGUAGES, LABEL_MAP,
)


def run() -> pd.DataFrame:
    """Build and save processed_metadata.csv; return the DataFrame."""
    print("\n" + "─" * 65)
    print("  STEP 3 — Building Metadata CSV  (with duration tracking)")
    print("─" * 65)

    rows = []
    for wav in sorted(PROCESSED_DIR.rglob("*.wav")):
        parts = wav.relative_to(PROCESSED_DIR).parts
        if len(parts) < 3:
            continue
        lang, emo = parts[0], parts[1]
        if lang not in LANGUAGES or emo not in LABEL_MAP:
            continue
            
        duration = sf.info(str(wav)).duration
        aug_type = "original"
        if "_aug" in wav.stem:
            aug_type = wav.stem.split("_")[-1]

        rows.append({
            "file_path":    wav.relative_to(BASE_DIR).as_posix(),
            "language":     lang,
            "emotion":      emo,
            "label":        LABEL_MAP[emo],
            "duration":     round(duration, 3),
            "is_augmented": 1 if "_aug" in wav.stem else 0,
            "aug_type":     aug_type,
        })

    df = pd.DataFrame(rows, columns=[
        "file_path", "language", "emotion", "label", 
        "duration", "is_augmented", "aug_type"
    ])
    df.to_csv(METADATA_CSV, index=False)

    orig = df[df["is_augmented"] == 0]
    aug  = df[df["is_augmented"] == 1]
    print(f"\n  Total rows   : {len(df)}")
    print(f"  Originals    : {len(orig)}")
    print(f"  Augmented    : {len(aug)}")
    print(f"  ✓ Saved : {METADATA_CSV.name}")
    return df
