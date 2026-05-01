"""
main.py
=======
CEP ML Preprocessing Pipeline — Entry Point

Runs all four steps in order:
  1. Preprocess  — .webm → 16kHz WAV, normalize, trim silence, rename (e.g. eng_calm_001.wav)
  2. Augment     — balance minority classes (pitch / stretch / noise)
  3. Metadata    — write processed_metadata.csv
  4. EDA         — statistics tables + acoustic charts

Usage:
    python main.py

Requirements:
    pip install -r requirements.txt
    ffmpeg must be on PATH (needed for .webm decoding)
"""

from pipeline import preprocess, augment, metadata, eda
from pipeline.config import LABEL_MAP, METADATA_CSV, EDA_DIR


def main():
    print("=" * 65)
    print("  CEP — ML PREPROCESSING PIPELINE")
    print("  Source : dataset/raw/<language>/<emotion>/*.webm")
    print("  Output : dataset/processed/  |  processed_metadata.csv  |  eda/")
    print("=" * 65)

    print("\n  Label encoding:")
    for emo, lbl in LABEL_MAP.items():
        print(f"    {emo:<12} → {lbl}")

    # ── Run pipeline steps ────────────────────────────────────────────────────
    preprocess.run()
    augment.run()
    df = metadata.run()
    eda.run(df)

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ✅  Pipeline complete!")
    print(f"  CSV : {METADATA_CSV}")
    print(f"  EDA : {EDA_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
