"""Dataset loading, quality auditing, and exploratory data analysis (EDA) plots."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords

from src.utils import (
    LABEL_COL,
    RAW_DATA_PATH,
    RESULTS_DIR,
    TEXT_COL,
    ensure_directories,
    get_logger,
    save_json,
)

logger = get_logger(__name__)

URL_PATTERN = re.compile(r"(?:https?://|www\.|httpwww\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
ELONGATED_PATTERN = re.compile(r"(.)\1{2,}")
REPEATED_PUNCT_PATTERN = re.compile(r"[!?.]{2,}")
APOSTROPHE_LESS_CONTRACTION_PATTERN = re.compile(
    r"\b(?:im|dont|wasnt|cant|didnt|isnt|wont|ive|youre|theyre|weve|shouldnt|couldnt|wouldnt|hasnt|havent)\b",
    re.IGNORECASE,
)


def load_and_clean_raw_dataset(filepath: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the raw CSV, drop unused index, remove nulls/duplicates, and report row counts."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {filepath}. "
            "Download Mental_Health_6Class_Final_Cleaned.csv from "
            "https://data.mendeley.com/datasets/rw57wvg6z2/2 and place it in data/raw/."
        )

    df = pd.read_csv(filepath)
    rows_initial = len(df)
    logger.info("Loaded raw dataset from %s: %d rows, columns=%s", filepath.name, rows_initial, list(df.columns))

    # Drop the exported CSV index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 1. Drop nulls in required columns
    before_nulls = len(df)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()
    nulls_removed = before_nulls - len(df)

    # 2. Drop empty/whitespace-only texts or labels
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    before_empty = len(df)
    mask_valid = (df[TEXT_COL].str.strip() != "") & (df[LABEL_COL] != "")
    df = df.loc[mask_valid].copy()
    empty_removed = before_empty - len(df)

    # 3. Drop exact text duplicates
    before_exact_dup = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL], keep="first").copy()
    exact_dups_removed = before_exact_dup - len(df)

    # 4. Drop case/whitespace-normalized duplicates to prevent train/val/test leakage
    before_norm_dup = len(df)
    norm_key = df[TEXT_COL].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.loc[~norm_key.duplicated(keep="first")].copy()
    normalized_dups_removed = before_norm_dup - len(df)

    df = df.reset_index(drop=True)
    total_removed = rows_initial - len(df)

    audit_counts = {
        "rows_initial": rows_initial,
        "nulls_removed": nulls_removed,
        "empty_removed": empty_removed,
        "exact_dups_removed": exact_dups_removed,
        "normalized_dups_removed": normalized_dups_removed,
        "total_removed": total_removed,
        "rows_final": len(df),
    }

    logger.info(
        "Deduplication & null check complete: initial=%d | nulls=%d | empty=%d | "
        "exact_dups=%d | norm_dups=%d | total_removed=%d | final=%d",
        rows_initial,
        nulls_removed,
        empty_removed,
        exact_dups_removed,
        normalized_dups_removed,
        total_removed,
        len(df),
    )
    return df, audit_counts


def compute_eda_summary(df: pd.DataFrame, audit_counts: dict[str, int]) -> dict[str, Any]:
    """Compute class distribution, text length stats, and social media artifact metrics."""
    texts = df[TEXT_COL]
    char_lens = texts.str.len()
    word_lens = texts.str.split().str.len()

    class_counts = df[LABEL_COL].value_counts().to_dict()
    total = len(df)
    class_pct = {k: round(v / total * 100.0, 2) for k, v in class_counts.items()}
    imbalance_ratio = round(max(class_counts.values()) / min(class_counts.values()), 2)

    per_class_length = (
        df.assign(word_len=word_lens, char_len=char_lens)
        .groupby(LABEL_COL)[["word_len", "char_len"]]
        .agg(["mean", "median", "max"])
        .round(2)
    )
    per_class_length_dict = {
        cls: {
            "word_mean": float(per_class_length.loc[cls, ("word_len", "mean")]),
            "word_median": float(per_class_length.loc[cls, ("word_len", "median")]),
            "word_max": int(per_class_length.loc[cls, ("word_len", "max")]),
            "char_mean": float(per_class_length.loc[cls, ("char_len", "mean")]),
            "char_median": float(per_class_length.loc[cls, ("char_len", "median")]),
        }
        for cls in class_counts
    }

    # Count social-media artifacts across the cleaned dataset
    artifact_flags = pd.DataFrame(
        {
            LABEL_COL: df[LABEL_COL],
            "has_url": texts.str.contains(URL_PATTERN),
            "has_mention": texts.str.contains(MENTION_PATTERN),
            "has_hashtag": texts.str.contains(HASHTAG_PATTERN),
            "has_emoji": texts.apply(lambda s: emoji.emoji_count(s) > 0),
            "has_elongated": texts.apply(lambda s: bool(ELONGATED_PATTERN.search(s))),
            "has_repeated_punct": texts.str.contains(REPEATED_PUNCT_PATTERN),
            "has_no_apostrophe_contraction": texts.str.contains(APOSTROPHE_LESS_CONTRACTION_PATTERN),
        }
    )

    artifact_totals = {
        col: int(artifact_flags[col].sum())
        for col in artifact_flags.columns
        if col != LABEL_COL
    }

    return {
        "audit_counts": audit_counts,
        "num_classes": len(class_counts),
        "class_counts": class_counts,
        "class_percentages": class_pct,
        "imbalance_ratio": imbalance_ratio,
        "overall_word_length": {
            "min": int(word_lens.min()),
            "median": float(word_lens.median()),
            "mean": round(float(word_lens.mean()), 2),
            "p95": round(float(word_lens.quantile(0.95)), 2),
            "max": int(word_lens.max()),
        },
        "overall_char_length": {
            "min": int(char_lens.min()),
            "median": float(char_lens.median()),
            "mean": round(float(char_lens.mean()), 2),
            "p95": round(float(char_lens.quantile(0.95)), 2),
            "max": int(char_lens.max()),
        },
        "per_class_length": per_class_length_dict,
        "artifact_totals": artifact_totals,
    }


def _plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    counts = df[LABEL_COL].value_counts()
    total = len(df)
    ratio = counts.max() / counts.min()

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("viridis", n_colors=len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="black", linewidth=0.6)

    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.004,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        f"Class Distribution (N = {total:,}, Imbalance Ratio = {ratio:.2f}x)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Emotion / Mental Health Category", fontsize=11)
    ax.set_ylabel("Number of Posts", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_length_distributions(df: pd.DataFrame, output_path: Path) -> None:
    word_lens = df[TEXT_COL].str.split().str.len()
    char_lens = df[TEXT_COL].str.len()
    plot_df = pd.DataFrame({LABEL_COL: df[LABEL_COL], "words": word_lens, "chars": char_lens})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Word length histogram clipped at 99th percentile for readability
    w_p99 = word_lens.quantile(0.99)
    axes[0].hist(word_lens.clip(upper=w_p99), bins=40, color="#2b6cb0", edgecolor="white", alpha=0.85)
    axes[0].axvline(word_lens.median(), color="crimson", linestyle="--", label=f"Median: {word_lens.median():.0f}")
    axes[0].set_title(f"Word Length (clipped at p99={w_p99:.0f})", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Words per Post")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Character length histogram clipped at 99th percentile
    c_p99 = char_lens.quantile(0.99)
    axes[1].hist(char_lens.clip(upper=c_p99), bins=40, color="#2f855a", edgecolor="white", alpha=0.85)
    axes[1].axvline(char_lens.median(), color="crimson", linestyle="--", label=f"Median: {char_lens.median():.0f}")
    axes[1].set_title(f"Character Length (clipped at p99={c_p99:.0f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Characters per Post")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    # Per-class word length boxplot (log scale shows Depression length contrast clearly)
    order = df[LABEL_COL].value_counts().index.tolist()
    sns.boxplot(data=plot_df, x=LABEL_COL, y="words", order=order, ax=axes[2], palette="viridis", hue=LABEL_COL, legend=False)
    axes[2].set_yscale("log")
    axes[2].set_title("Word Length by Class (Log Scale)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Category")
    axes[2].set_ylabel("Words (log scale)")
    axes[2].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        sns.despine(ax=ax)

    fig.suptitle("Post Length Distributions Across Dataset", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_top_words_per_class(df: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    stop_words = set(stopwords.words("english")) | {
        "im", "dont", "cant", "didnt", "wasnt", "wouldnt", "couldnt", "ive",
        "feel", "feeling", "like", "really", "get", "one", "know", "time", "want",
    }
    classes = df[LABEL_COL].value_counts().index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    palette = sns.color_palette("mako", n_colors=len(classes))

    for idx, cls in enumerate(classes):
        ax = axes_flat[idx]
        tokens: list[str] = []
        for text in df.loc[df[LABEL_COL] == cls, TEXT_COL]:
            for tok in re.findall(r"[a-z]{2,}", text.lower()):
                if tok not in stop_words:
                    tokens.append(tok)
        most_common = Counter(tokens).most_common(top_n)
        words, freqs = zip(*most_common) if most_common else ([], [])
        ax.barh(list(reversed(words)), list(reversed(freqs)), color=palette[idx], edgecolor="black", linewidth=0.4)
        ax.set_title(f"Top Words: {cls}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        sns.despine(ax=ax)

    fig.suptitle("Top Unigrams per Emotion Class (Common Filler Words Filtered)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_social_artifacts(df: pd.DataFrame, output_path: Path) -> None:
    texts = df[TEXT_COL]
    art_df = pd.DataFrame(
        {
            LABEL_COL: df[LABEL_COL],
            "Hashtags (#)": texts.str.contains(HASHTAG_PATTERN).astype(int) * 100,
            "Mentions (@)": texts.str.contains(MENTION_PATTERN).astype(int) * 100,
            "Emojis": texts.apply(lambda s: 100 if emoji.emoji_count(s) > 0 else 0),
            "URLs": texts.str.contains(URL_PATTERN).astype(int) * 100,
            "Elongated Words": texts.apply(lambda s: 100 if ELONGATED_PATTERN.search(s) else 0),
            "Repeated Punct": texts.str.contains(REPEATED_PUNCT_PATTERN).astype(int) * 100,
        }
    )

    melted = art_df.groupby(LABEL_COL).mean().reset_index().melt(
        id_vars=LABEL_COL,
        var_name="Artifact",
        value_name="Percentage of Posts (%)",
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=melted, x=LABEL_COL, y="Percentage of Posts (%)", hue="Artifact", ax=ax, palette="Set2")
    ax.set_title(
        "Social Media Artifacts per Class (% of Posts)\n"
        "(Note: Pre-cleaned CSV has 0% hashtags/mentions/emojis; elongated text & repeated punctuation remain)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("Posts Containing Artifact (%)", fontsize=11)
    ax.legend(title="Artifact Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_eda(filepath: Path = RAW_DATA_PATH, results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute data loading, deduplication, summary statistics, and EDA plot generation."""
    ensure_directories()
    df, audit_counts = load_and_clean_raw_dataset(filepath)
    summary = compute_eda_summary(df, audit_counts)
    save_json(summary, results_dir / "eda_summary.json")

    logger.info("Generating EDA plots in %s ...", results_dir)
    _plot_class_distribution(df, results_dir / "eda_class_distribution.png")
    _plot_length_distributions(df, results_dir / "eda_length_distribution.png")
    _plot_top_words_per_class(df, results_dir / "eda_top_words_per_class.png")
    _plot_social_artifacts(df, results_dir / "eda_social_artifacts_per_class.png")
    logger.info("EDA complete. Summary and 4 plots saved to %s.", results_dir)
    return df, summary
