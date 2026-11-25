"""Social-media-aware text preprocessing and stratified dataset splitting."""

from __future__ import annotations

import html
import re
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.utils import (
    LABEL_COL,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    TEXT_COL,
    ensure_directories,
    get_logger,
    save_json,
)

logger = get_logger(__name__)

# Preserve negation cues because removing them flips sentiment/mental-health polarity
NEGATION_WORDS: set[str] = {
    "no",
    "nor",
    "not",
    "never",
    "neither",
    "none",
    "nothing",
    "nowhere",
    "cannot",
    "cant",
    "hardly",
    "barely",
    "scarcely",
}

BASE_STOPWORDS: set[str] = set(stopwords.words("english")) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()

# Contraction map covering both standard apostrophe forms and apostrophe-less forms
# (Mental_Health_6Class_Final_Cleaned.csv stripped apostrophes in >7,700 rows).
CONTRACTION_MAP: dict[str, str] = {
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "you're": "you are",
    "youre": "you are",
    "you've": "you have",
    "youve": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "he's": "he is",
    "she's": "she is",
    "shes": "she is",
    "it's": "it is",
    "we're": "we are",
    "we've": "we have",
    "weve": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "theyre": "they are",
    "they've": "they have",
    "theyve": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "what's": "what is",
    "whats": "what is",
    "who's": "who is",
    "let's": "let us",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "can't": "can not",
    "cant": "can not",
    "couldn't": "could not",
    "couldnt": "could not",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "haven't": "have not",
    "havent": "have not",
    "hasn't": "has not",
    "hasnt": "has not",
    "hadn't": "had not",
    "hadnt": "had not",
    "mustn't": "must not",
    "needn't": "need not",
    "ain't": "am not",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
}

CONTRACTION_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTION_MAP, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Regex patterns (handles standard URLs plus broken 'httpwww.*' URLs present in raw CSV)
URL_REGEX = re.compile(r"(?:https?://|www\.|httpwww\.)\S+", flags=re.IGNORECASE)
MENTION_REGEX = re.compile(r"@\w+")
HASHTAG_REGEX = re.compile(r"#(\w+)")
CAMEL_SPLIT_REGEX = re.compile(r"([a-z0-9])([A-Z])")
ELONGATED_CHAR_REGEX = re.compile(r"([a-zA-Z])\1{2,}")
REPEATED_PUNCT_REGEX = re.compile(r"([!?.])\1+")
HTML_REMNANT_REGEX = re.compile(r"\b(?:amp|quot)\b", flags=re.IGNORECASE)
TOKEN_REGEX = re.compile(r"<URL>|<USER>|[a-z0-9]+")


def _replace_hashtag(match: re.Match[str]) -> str:
    """Drop '#' and split CamelCase hashtags (e.g., #FoodLover -> food lover)."""
    tag = match.group(1)
    spaced = CAMEL_SPLIT_REGEX.sub(r"\1 \2", tag)
    spaced = spaced.replace("_", " ")
    return f" {spaced.lower()} "


def _demojize_text(text: str) -> str:
    """Convert Unicode emojis to readable words while keeping ASCII text intact."""
    demojized = emoji.demojize(text, delimiters=(" ", " "))
    # Convert colon/underscore emoji aliases such as ':crying_face:' -> 'crying face'
    return demojized.replace("_", " ")


def _expand_contractions(text: str) -> str:
    return CONTRACTION_REGEX.sub(lambda m: CONTRACTION_MAP[m.group(0).lower()], text)


def clean_social_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Apply full social-media-specific cleaning, normalization, stopword filtering, and lemmatization."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # 1. Decode HTML entities and normalize URLs / mentions / hashtags
    text = html.unescape(text)
    text = URL_REGEX.sub(" URLPLACEHOLDER ", text)
    text = MENTION_REGEX.sub(" USERPLACEHOLDER ", text)
    text = HASHTAG_REGEX.sub(_replace_hashtag, text)

    # 2. Convert emojis to text descriptions so emotional signal is preserved
    text = _demojize_text(text)

    # 3. Normalize elongated characters ("sooooo" -> "soo") and repeated punctuation ("!!!" -> "!")
    text = ELONGATED_CHAR_REGEX.sub(r"\1\1", text)
    text = REPEATED_PUNCT_REGEX.sub(r"\1", text)

    # 4. Lowercase and expand contractions (with and without apostrophes)
    text = text.lower()
    text = _expand_contractions(text)

    # 5. Clean leftover stripped HTML artifacts and restore placeholder tokens
    text = HTML_REMNANT_REGEX.sub(" ", text)
    text = text.replace("urlplaceholder", "<URL>").replace("userplaceholder", "<USER>")
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenize, optionally filter stopwords (preserving negations), and lemmatize
    tokens = TOKEN_REGEX.findall(text)
    cleaned_tokens: list[str] = []
    for tok in tokens:
        if tok in ("<URL>", "<USER>"):
            cleaned_tokens.append(tok)
            continue
        if remove_stopwords and tok in BASE_STOPWORDS:
            continue
        if lemmatize:
            tok = LEMMATIZER.lemmatize(tok)
        cleaned_tokens.append(tok)

    result = " ".join(cleaned_tokens)
    # Fallback: if stopword filtering empties a short post, retain lemmatized non-placeholder tokens
    if not result:
        fallback_tokens = [
            tok if tok in ("<URL>", "<USER>") else (LEMMATIZER.lemmatize(tok) if lemmatize else tok)
            for tok in tokens
        ]
        result = " ".join(fallback_tokens)
    return result


def select_before_after_examples(df: pd.DataFrame, cleaned_col: str = "clean_text", n: int = 5) -> list[dict[str, str]]:
    """Select diverse illustrative raw vs. preprocessed examples across different classes."""
    examples: list[dict[str, str]] = []
    specs = [
        ("URL + Noise", lambda s: bool(URL_REGEX.search(s))),
        ("Elongated / Repeated Punct", lambda s: bool(ELONGATED_CHAR_REGEX.search(s) or REPEATED_PUNCT_REGEX.search(s))),
        ("Contraction + Negation", lambda s: bool(re.search(r"\b(?:dont|cant|wasnt|didnt|not)\b", s, re.I))),
        ("Long Mental Health Post", lambda s: len(s.split()) >= 60),
        ("Short Emotion Post", lambda s: 5 <= len(s.split()) <= 15),
    ]

    used_idx: set[int] = set()
    used_labels: set[str] = set()
    for tag, pred in specs:
        mask = df[TEXT_COL].apply(pred) & (~df.index.isin(used_idx))
        subset = df.loc[mask]
        # Prefer a class not yet shown in the examples
        diverse_subset = subset.loc[~subset[LABEL_COL].isin(used_labels)]
        pick_pool = diverse_subset if not diverse_subset.empty else subset
        if not pick_pool.empty:
            row = pick_pool.iloc[0]
            used_idx.add(int(row.name))
            used_labels.add(str(row[LABEL_COL]))
            examples.append(
                {
                    "category": tag,
                    "label": str(row[LABEL_COL]),
                    "raw": str(row[TEXT_COL])[:220],
                    "cleaned": str(row[cleaned_col])[:220],
                }
            )
        if len(examples) >= n:
            break
    return examples


def preprocess_and_split(
    df: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Clean texts, log 5 before/after examples, split 70/15/15 stratified, and compute class weights."""
    ensure_directories()
    work_df = df[[TEXT_COL, LABEL_COL]].copy()

    logger.info("Preprocessing %d posts (URLs, mentions, hashtags, emojis, contractions, lemmatization)...", len(work_df))
    work_df["clean_text"] = work_df[TEXT_COL].apply(clean_social_text)

    # Drop any edge-case rows that became completely empty after cleaning
    empty_after_clean = int((work_df["clean_text"].str.strip() == "").sum())
    if empty_after_clean > 0:
        work_df = work_df.loc[work_df["clean_text"].str.strip() != ""].reset_index(drop=True)
        logger.warning("Removed %d rows that became empty after preprocessing.", empty_after_clean)

    # Log 5 before/after examples from the dataset + 1 raw social-media demo
    examples = select_before_after_examples(work_df, n=5)
    logger.info("=== 5 Before / After Preprocessing Examples ===")
    for i, ex in enumerate(examples, 1):
        logger.info("[%d | %s | %s]\n  RAW  : %s\n  CLEAN: %s", i, ex["category"], ex["label"], ex["raw"], ex["cleaned"])

    synthetic_raw = "@user I'm sooooo stressed about tomorrow!! Check https://example.com #MentalHealth 😢"
    logger.info("[Social Demo] RAW: %s -> CLEAN: %s", synthetic_raw, clean_social_text(synthetic_raw))

    # Stratified split: 70% train / 15% val / 15% test (split before any class weighting)
    train_df, temp_df = train_test_split(
        work_df,
        test_size=0.30,
        stratify=work_df[LABEL_COL],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[LABEL_COL],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    # Compute balanced class weights strictly from the training split
    classes = np.array(sorted(train_df[LABEL_COL].unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df[LABEL_COL].to_numpy())
    class_weights = {str(cls): round(float(w), 4) for cls, w in zip(classes, weights)}

    split_summary = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "empty_after_clean_removed": empty_after_clean,
        "class_weights_train": class_weights,
        "examples": examples,
        "synthetic_social_demo": {
            "raw": synthetic_raw,
            "cleaned": clean_social_text(synthetic_raw),
        },
    }
    save_json(split_summary, results_dir / "preprocessing_summary.json")

    logger.info(
        "Stratified split saved to %s: train=%d (70%%), val=%d (15%%), test=%d (15%%)",
        processed_dir,
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info("Training-set balanced class weights: %s", class_weights)
    return train_df, val_df, test_df, class_weights
