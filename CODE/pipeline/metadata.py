# Step 3 — Scan processed dataset/ and write processed_metadata.xlsx (training index).

import pandas as pd
import soundfile as sf
from pathlib import Path

from .config import (
    BASE_DIR, PROCESSED_DIR, METADATA_FILE, METADATA_DIR,
    LANGUAGES, LABEL_MAP,
)

def run() -> pd.DataFrame:
    
    """Build and save processed_metadata.xlsx; return the DataFrame."""

    print("  STEP 3 — Building Metadata Excel  (with duration tracking)")

    # Ensure the metadata directory exists
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for wav in sorted(PROCESSED_DIR.rglob("*.wav")):
        parts = wav.relative_to(PROCESSED_DIR).parts

        if len(parts) < 2:
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

    # Save to Excel to match project tree
    df.to_excel(METADATA_FILE, index=False)
    print(f"  ✓ Saved {len(df)} rows to {METADATA_FILE.name}")

    return df