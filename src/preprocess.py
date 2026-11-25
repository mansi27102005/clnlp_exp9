"""Social-media text cleaning and stratified 70/15/15 train-val-test split."""

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

from src.utils import LABEL_COL, PROCESSED_DIR, RANDOM_SEED, TEXT_COL, ensure_dirs, get_logger

logger = get_logger("preprocess")

# Keep negation words so sentiment polarity is not flipped
NEGATIONS = {"no", "nor", "not", "never", "neither", "none", "nothing", "cannot"}
STOPWORDS = set(stopwords.words("english")) - NEGATIONS
LEMMATIZER = WordNetLemmatizer()

# Covers both standard ("don't") and apostrophe-less ("dont") contractions in this dataset
CONTRACTIONS = {
    "i'm": "i am", "im": "i am", "i've": "i have", "ive": "i have",
    "you're": "you are", "youre": "you are", "they're": "they are", "theyre": "they are",
    "we're": "we are", "it's": "it is", "that's": "that is", "thats": "that is",
    "there's": "there is", "theres": "there is", "what's": "what is", "whats": "what is",
    "don't": "do not", "dont": "do not", "doesn't": "does not", "doesnt": "does not",
    "didn't": "did not", "didnt": "did not", "can't": "can not", "cant": "can not",
    "couldn't": "could not", "couldnt": "could not", "won't": "will not", "wont": "will not",
    "wouldn't": "would not", "wouldnt": "would not", "shouldn't": "should not", "shouldnt": "should not",
    "isn't": "is not", "isnt": "is not", "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not", "weren't": "were not", "werent": "were not",
    "haven't": "have not", "havent": "have not", "hasn't": "has not", "hasnt": "has not",
}
CONTRACTION_RE = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACTIONS)) + r")\b", re.I)
URL_RE = re.compile(r"(?:https?://|www\.|httpwww\.)\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
ELONGATED_RE = re.compile(r"([a-zA-Z])\1{2,}")
REPEAT_PUNCT_RE = re.compile(r"([!?.])\1+")
TOKEN_RE = re.compile(r"<URL>|<USER>|[a-z0-9]+")


def clean_social_text(text: str) -> str:
    """Clean a raw social post: URLs, mentions, hashtags, emojis, contractions, stopwords, lemmas."""
    text = html.unescape("" if pd.isna(text) else str(text))
    text = URL_RE.sub(" URLPLACEHOLDER ", text)
    text = MENTION_RE.sub(" USERPLACEHOLDER ", text)
    text = HASHTAG_RE.sub(lambda m: " " + CAMEL_RE.sub(r"\1 \2", m.group(1)).replace("_", " ").lower() + " ", text)

    # Convert emojis into words instead of deleting them
    text = emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")

    # Normalize elongated words ("sooooo" -> "soo") and repeated punctuation
    text = ELONGATED_RE.sub(r"\1\1", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text).lower()
    text = CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)
    text = re.sub(r"\b(?:amp|quot)\b", " ", text)
    text = text.replace("urlplaceholder", "<URL>").replace("userplaceholder", "<USER>")

    tokens = TOKEN_RE.findall(text)
    kept = [
        tok if tok in ("<URL>", "<USER>") else LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok in ("<URL>", "<USER>") or tok not in STOPWORDS
    ]
    return " ".join(kept) if kept else " ".join(tokens)


def preprocess_and_split(df: pd.DataFrame, out_dir: Path = PROCESSED_DIR, seed: int = RANDOM_SEED):
    """Clean all texts, print 5 before/after examples, split 70/15/15 stratified, and compute class weights."""
    ensure_dirs()
    work = df[[TEXT_COL, LABEL_COL]].copy()
    work["clean_text"] = work[TEXT_COL].apply(clean_social_text)
    work = work[work["clean_text"].str.strip() != ""].reset_index(drop=True)

    logger.info("=== 5 Before / After Preprocessing Examples ===")
    for i, (_, row) in enumerate(work.groupby(LABEL_COL).first().head(5).iterrows(), 1):
        logger.info("[%d] RAW: %s\n    CLEAN: %s", i, str(row[TEXT_COL])[:140], str(row["clean_text"])[:140])

    # Stratified split: 70% train / 15% val / 15% test
    train_df, temp_df = train_test_split(work, test_size=0.30, stratify=work[LABEL_COL], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[LABEL_COL], random_state=seed)

    for name, part in (("train.csv", train_df), ("val.csv", val_df), ("test.csv", test_df)):
        part.reset_index(drop=True).to_csv(out_dir / name, index=False)

    # Balanced class weights computed strictly from training split
    classes = np.array(sorted(train_df[LABEL_COL].unique()))
    weights = compute_class_weight("balanced", classes=classes, y=train_df[LABEL_COL].to_numpy())
    class_weights = {str(c): round(float(w), 4) for c, w in zip(classes, weights)}

    logger.info("Split saved: train=%d, val=%d, test=%d | weights=%s", len(train_df), len(val_df), len(test_df), class_weights)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), class_weights
