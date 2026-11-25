"""Dataset loading, null/duplicate cleaning, and EDA plots."""

import re
from collections import Counter
from pathlib import Path

import emoji
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords

from src.utils import LABEL_COL, RAW_DATA_PATH, RESULTS_DIR, TEXT_COL, ensure_dirs, get_logger

logger = get_logger("data_loader")

URL_RE = re.compile(r"(?:https?://|www\.|httpwww\.)\S+", re.I)
ELONGATED_RE = re.compile(r"([a-zA-Z])\1{2,}")


def load_clean_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load CSV, drop nulls and duplicates, and log removed row counts."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset missing at {filepath}. Download from https://data.mendeley.com/datasets/rw57wvg6z2/2"
        )

    df = pd.read_csv(filepath)
    start_rows = len(df)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop nulls and blank rows
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[(df[TEXT_COL].str.strip() != "") & (df[LABEL_COL] != "")]

    # Drop exact duplicates and case/whitespace-normalized duplicates
    df = df.drop_duplicates(subset=[TEXT_COL], keep="first")
    norm_key = df[TEXT_COL].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.loc[~norm_key.duplicated(keep="first")].reset_index(drop=True)

    logger.info("Loaded %d rows -> removed %d invalid/duplicate rows -> %d final rows", start_rows, start_rows - len(df), len(df))
    return df


def run_eda(filepath: Path = RAW_DATA_PATH, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Clean raw data, log imbalance stats, and save 4 concise EDA plots."""
    ensure_dirs()
    df = load_clean_data(filepath)

    counts = df[LABEL_COL].value_counts()
    ratio = counts.max() / counts.min()
    logger.info("Classes (%d): %s | Imbalance ratio: %.2fx", len(counts), counts.to_dict(), ratio)

    # 1. Class distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("viridis", len(counts)))
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 120, f"{c:,}\n({c/len(df)*100:.1f}%)", ha="center", fontsize=9)
    ax.set_title(f"Class Distribution (N={len(df):,}, Imbalance={ratio:.2f}x)", fontweight="bold")
    ax.set_ylabel("Posts")
    ax.set_ylim(0, counts.max() * 1.16)
    fig.tight_layout()
    fig.savefig(results_dir / "eda_class_distribution.png", dpi=180)
    plt.close(fig)

    # 2. Word & character length histograms
    words = df[TEXT_COL].str.split().str.len()
    chars = df[TEXT_COL].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(words.clip(upper=words.quantile(0.99)), bins=35, color="#2b6cb0", edgecolor="white")
    axes[0].axvline(words.median(), color="red", linestyle="--", label=f"Median: {words.median():.0f}")
    axes[0].set_title("Word Length (clipped at p99)")
    axes[0].set_xlabel("Words")
    axes[0].legend()

    axes[1].hist(chars.clip(upper=chars.quantile(0.99)), bins=35, color="#2f855a", edgecolor="white")
    axes[1].axvline(chars.median(), color="red", linestyle="--", label=f"Median: {chars.median():.0f}")
    axes[1].set_title("Character Length (clipped at p99)")
    axes[1].set_xlabel("Characters")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "eda_length_distribution.png", dpi=180)
    plt.close(fig)

    # 3. Top 10 words per class
    stop = set(stopwords.words("english")) | {"im", "dont", "cant", "feel", "feeling", "like", "really"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, cls in zip(axes.flatten(), counts.index):
        toks = [w for t in df.loc[df[LABEL_COL] == cls, TEXT_COL] for w in re.findall(r"[a-z]{2,}", t.lower()) if w not in stop]
        top_w = Counter(toks).most_common(10)
        w, f = zip(*reversed(top_w))
        ax.barh(w, f, color="#4c72b0")
        ax.set_title(cls, fontweight="bold")
    fig.suptitle("Top 10 Words per Class", fontweight="bold")
    fig.tight_layout()
    fig.savefig(results_dir / "eda_top_words_per_class.png", dpi=180)
    plt.close(fig)

    # 4. Social media artifacts per class (%)
    art = pd.DataFrame({
        LABEL_COL: df[LABEL_COL],
        "Hashtags": df[TEXT_COL].str.contains(r"#\w+").astype(int) * 100,
        "Mentions": df[TEXT_COL].str.contains(r"@\w+").astype(int) * 100,
        "Emojis": df[TEXT_COL].apply(lambda s: 100 if emoji.emoji_count(s) > 0 else 0),
        "URLs": df[TEXT_COL].str.contains(URL_RE).astype(int) * 100,
        "Elongated": df[TEXT_COL].apply(lambda s: 100 if ELONGATED_RE.search(s) else 0),
    })
    melted = art.groupby(LABEL_COL).mean().reset_index().melt(id_vars=LABEL_COL, var_name="Artifact", value_name="% Posts")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=melted, x=LABEL_COL, y="% Posts", hue="Artifact", ax=ax)
    ax.set_title("Social Media Artifacts by Class (Pre-cleaned CSV has 0% hashtags/mentions/emojis)")
    fig.tight_layout()
    fig.savefig(results_dir / "eda_social_artifacts_per_class.png", dpi=180)
    plt.close(fig)

    logger.info("Saved 4 EDA plots to %s", results_dir)
    return df
