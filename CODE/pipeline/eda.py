# pipeline/eda.py
# Step 4 — Exploratory Data Analysis: statistics tables + visual charts.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import random

from .config import BASE_DIR, TARGET_SR, EMOTIONS, LANGUAGES, LABEL_MAP, EDA_DIR, EDA_REPORT

COLORS = {"english": "#4A90D9", "urdu": "#E06C75"}
random.seed(42)

def _print_tabular_stats(df: pd.DataFrame):
    total = len(df)
    orig = df[df["is_augmented"] == 0]
    
    print("\n======================================================================")
    print("  CEP Dataset — Pandas Statistics Report")
    print("======================================================================")
    print(f"Total samples : {total}")
    print(f"Originals     : {len(orig)}")
    print(f"Augmented     : {len(df[df['is_augmented'] == 1])}")
    
    # 1. Samples by Language
    print("\n--- TABLE 1: Samples by Language ---")
    lang_counts = df['language'].value_counts()
    lang_df = pd.DataFrame({'Count': lang_counts, '%': (lang_counts / total * 100).round(1)})
    print(lang_df.to_string())
    
    # 2. Samples by Emotion (Orig vs Aug)
    print("\n--- TABLE 2: Samples by Emotion ---")
    emo_orig = orig['emotion'].value_counts()
    emo_aug = df[df['is_augmented'] == 1]['emotion'].value_counts()
    emo_df = pd.DataFrame({'Original': emo_orig, 'Augmented': emo_aug}).fillna(0).astype(int)
    emo_df['Total'] = emo_df['Original'] + emo_df['Augmented']
    emo_df['%'] = (emo_df['Total'] / total * 100).round(1)
    print(emo_df.to_string())
    
    # 3. Emotion x Language (Originals only)
    print("\n--- TABLE 3: Emotion × Language (Originals) ---")
    cross = pd.crosstab(orig['emotion'], orig['language'], margins=True, margins_name="Total")
    print(cross.to_string())
    
    # 4. Audio Durations
    if "duration" in df.columns:
        print("\n--- TABLE 4: Audio Durations (seconds) - Original Recordings ---")
        dur = orig.groupby('emotion')['duration'].agg(['mean', 'min', 'max']).round(2)
        print(dur.to_string())
        
    # 5. Breakdown by Aug Type
    if "aug_type" in df.columns:
        print("\n--- TABLE 5: Breakdown of Datapoints ---")
        aug_counts = df['aug_type'].value_counts()
        aug_df = pd.DataFrame({'Count': aug_counts, '%': (aug_counts / total * 100).round(1)})
        print(aug_df.to_string())
        
    print("======================================================================\n")

def _save_fig(name):
    plt.tight_layout()
    plt.savefig(EDA_DIR / name, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}")

def run(df: pd.DataFrame):
    print("\n" + "─" * 65)
    print("  STEP 4 — Exploratory Data Analysis (EDA)")
    print("─" * 65)
    
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Print tables to the terminal
    _print_tabular_stats(df)

    orig = df[df["is_augmented"] == 0]
    if orig.empty:
        return
        
    print("\n  Generating charts...")

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=orig, x="emotion", hue="language", palette=COLORS, ax=ax)
    ax.set_title("Class Distribution per Emotion & Language", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    _save_fig("class_distribution.png")
    
    # 2. Language Split Pie Chart
    lang_counts = orig['language'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(lang_counts, labels=lang_counts.index, autopct="%1.1f%%", colors=list(COLORS.values()),
           wedgeprops={"linewidth": 2, "edgecolor": "white"}, textprops={"fontsize": 13})
    ax.set_title("Language Distribution (Originals)", fontsize=14, fontweight="bold")
    _save_fig("language_split.png")
    
    # 3. Heatmap
    matrix = pd.crosstab(orig["emotion"], orig["language"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 13})
    ax.set_title("Sample Count Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Emotion", fontsize=12)
    _save_fig("heatmap.png")
    
    # 4. Original vs Augmented Stacked Bar
    grouped = df.groupby(["language", "emotion", "is_augmented"]).size().unstack(fill_value=0)
    ax = grouped.plot(kind="bar", stacked=True, figsize=(12, 5), color=["#56B6C2", "#E5C07B"])
    ax.set_title("Original vs Augmented Samples per Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Language & Emotion", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.legend(["Original", "Augmented"])
    _save_fig("orig_vs_aug.png")
    
    # 5. Durations Boxplot
    if "duration" in orig.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=orig, x="emotion", y="duration", hue="language", palette=COLORS, ax=ax)
        ax.set_title("Audio Durations per Emotion", fontsize=14, fontweight="bold")
        ax.set_xlabel("Emotion", fontsize=12)
        ax.set_ylabel("Duration (seconds)", fontsize=12)
        _save_fig("durations_boxplot.png")
    
    # 6. Waveforms and Spectrograms Grids
    fig_w, ax_w = plt.subplots(2, 3, figsize=(15, 6), sharey=True)
    fig_s, ax_s = plt.subplots(2, 3, figsize=(15, 6))
    
    for i, emo in enumerate(EMOTIONS):
        sub = orig[orig["emotion"] == emo]
        if sub.empty: continue
            
        sample_path = BASE_DIR / sub.sample(1, random_state=42).iloc[0]["file_path"]
        y, sr = librosa.load(str(sample_path), sr=TARGET_SR)
        
        # Plot Waveform randomly selected
        curr_ax_w = ax_w[i//3, i%3]
        librosa.display.waveshow(y, sr=sr, ax=curr_ax_w, color="#4A90D9", alpha=0.8)
        curr_ax_w.set_title(f"Waveform: {emo.capitalize()}", fontsize=12)
        curr_ax_w.set_xlabel("Time (s)")

        # Plot Spectrogram
        curr_ax_s = ax_s[i//3, i%3]
        S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=curr_ax_s, cmap='magma')
        curr_ax_s.set_title(f"Mel Spectrogram: {emo.capitalize()}", fontsize=12)

    # Save grids explicitly
    plt.figure(fig_w.number)
    _save_fig("waveforms.png")
    
    plt.figure(fig_s.number)
    _save_fig("spectrograms.png")
    
    print(f"\n  All outputs → {EDA_DIR}")
