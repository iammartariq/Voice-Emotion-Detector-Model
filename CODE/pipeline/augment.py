# pipeline/augment.py
# Step 2 — Balance minority emotion classes via audio augmentation.
#
# Augmentation codes used in filenames:
#   ps+2  → pitch shift up   +2 semitones
#   ps-2  → pitch shift down −2 semitones
#   tsf   → time stretch fast (×1.1)
#   tss   → time stretch slow (×0.9)
#   ns    → add Gaussian noise

import random
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from .config import (
    PROCESSED_DIR, LANGUAGES, EMOTIONS, TARGET_SR,
)

random.seed(42)
np.random.seed(42)

# (code, transform_fn(y, sr) → y)
AUGMENTATIONS = [
    ("ps+2", lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=2)),
    ("ps-2", lambda y, sr: librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)),
    ("tsf",  lambda y, sr: librosa.effects.time_stretch(y, rate=1.1)),
    ("tss",  lambda y, sr: librosa.effects.time_stretch(y, rate=0.9)),
    ("ns",   lambda y, sr: np.clip(
        y + np.random.normal(0, 0.005, len(y)).astype(np.float32), -1.0, 1.0
    )),
]


def _normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def _save_wav(y: np.ndarray, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), y, TARGET_SR, subtype="PCM_16")


def _count_originals() -> dict:
    """Return {(lang, emo): [Path, ...]} for original (non-augmented) files."""
    inventory = {}
    for lang in LANGUAGES:
        for emo in EMOTIONS:
            folder = PROCESSED_DIR / lang / emo
            if folder.exists():
                files = [f for f in folder.glob("*.wav") if "_aug" not in f.stem]
                inventory[(lang, emo)] = files
    return inventory


def run():
    """Augment minority classes until all reach the max class size."""
    print("\n" + "─" * 65)
    print("  STEP 2 — Augmentation  (balancing minority classes)")
    print("─" * 65)

    inventory = _count_originals()
    if not inventory:
        print("  ⚠ No processed files found — run preprocess step first.")
        return

    counts = {k: len(v) for k, v in inventory.items()}

    # Print current distribution table
    print(f"\n  {'Language':<12} {'Emotion':<12} {'Count':>6}")
    print("  " + "─" * 32)
    min_cnt = min(counts.values())
    for (lang, emo), cnt in sorted(counts.items()):
        marker = " ←" if cnt == min_cnt else ""
        print(f"  {lang:<12} {emo:<12} {cnt:>6}{marker}")

    target = max(counts.values())
    print(f"\n  Target per class : {target}  (size of largest class)")

    total_new = 0

    for (lang, emo), files in sorted(inventory.items()):
        needed = target - len(files)
        if needed <= 0:
            continue

        print(f"\n  ↑ {lang}/{emo}: {len(files)} → {target}  (+{needed} files)")

        # Cycle through augmentation types and source files to fill the gap
        aug_cycle = (list(AUGMENTATIONS) * (needed // len(AUGMENTATIONS) + 1))[:needed]
        src_cycle = (files * (needed // len(files) + 2))[:needed]
        random.shuffle(aug_cycle)

        for i, (src_path, (code, aug_fn)) in enumerate(zip(src_cycle, aug_cycle)):
            dest = PROCESSED_DIR / lang / emo / f"{src_path.stem}_aug{i:03d}_{code}.wav"
            if dest.exists():
                continue
            try:
                y, _ = librosa.load(str(src_path), sr=TARGET_SR, mono=True)
                y = aug_fn(y.astype(np.float32), TARGET_SR).astype(np.float32)
                y = _normalize(y)
                _save_wav(y, dest)
                total_new += 1
            except Exception as e:
                print(f"\n    X {src_path.name} [{code}]: {e}")

    print(f"\n  Total new augmented files : {total_new}")
