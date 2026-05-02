# Step 1 — Convert .webm → 16kHz mono WAV, normalize amplitude, trim silence.

# Automatically clean renames the files: e.g. eng_calm_001.wav

import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm # for progress bar

from .config import (
    RAW_DIR, PROCESSED_DIR, LANGUAGES, LABEL_MAP,
    TARGET_SR, TOP_DB, MIN_SECS,
)

try:
    from pydub import AudioSegment
    from pydub.utils import which
    
except ImportError:
    sys.exit("ERROR: pydub not installed. Run: pip install pydub")

def _check_ffmpeg():
    
    if which("ffmpeg") is None:
        sys.exit(
            "\nERROR: ffmpeg not found on PATH.\n"
            "Install from https://ffmpeg.org/download.html and add to PATH.\n"
        )

def _load_webm(src: Path) -> np.ndarray:
    
    """Load a .webm file → float32 numpy array at TARGET_SR."""

    audio = AudioSegment.from_file(str(src), format="webm")
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    max_val = float(2 ** (audio.sample_width * 8 - 1))
    
    return samples / max_val

def _normalize(y: np.ndarray) -> np.ndarray:
    
    """Peak-normalize to [-1, 1]."""
    
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def _trim_silence(y: np.ndarray) -> np.ndarray:
    
    """Trim leading/trailing silence using librosa."""
    
    trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return trimmed

def _save_wav(y: np.ndarray, dest: Path):
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), y, TARGET_SR, subtype="PCM_16")
    
def run() -> int:

    """
    Preprocess .webm files directly from dataset/raw/<language>/<emotion>/.
    Skips files if they already exist in dataset/processed/.
    No CSV dependencies.
    """

    _check_ffmpeg()
    print("STEP 1 — Preprocessing & Renaming (.webm → 16kHz WAV)")

    ok = skip = fail = 0

    for lang in LANGUAGES:

        for emo in LABEL_MAP.keys():
            raw_folder = RAW_DIR / lang / emo

            if not raw_folder.exists():
                continue
                
            webm_files = sorted(raw_folder.glob("*.webm"))
            
            pbar = tqdm(webm_files, desc=f"{lang}/{emo}", leave=False)

            for i, src in enumerate(pbar, start=1):

                # Clean naming convention: just numbers (001.wav)

                clean_name = f"{i:03d}.wav"
                dest = PROCESSED_DIR / lang / emo / clean_name
                
                # Prevent duplicates: skip if file is already there
                
                if dest.exists() and dest.stat().st_size > 0:
                    skip += 1
                    continue
                    
                try:

                    y = _load_webm(src)
                    y = _normalize(y)
                    y = _trim_silence(y)

                    if len(y) < TARGET_SR * MIN_SECS:
                        print(f"\n  Too short after trim, skipping: {src.name}")
                        continue
                        
                    _save_wav(y, dest)
                    ok += 1

                except Exception as e:

                    fail += 1
                    print(f"\n  ✗ {src.name}: {e}")

            pbar.close()

    print(f"\n  ✓ Processed : {ok}")
    print(f"  ↷ Skipped   : {skip}  (already present, no duplicates)")
    print(f"  ✗ Failed    : {fail}")
    return ok + skip