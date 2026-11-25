"""Feature extraction for classical ML (TF-IDF) and deep learning (BiLSTM sequences)."""

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder

from src.utils import MODELS_DIR, ensure_dirs, get_logger

logger = get_logger("features")


def build_word_only_tfidf() -> TfidfVectorizer:
    """Word (1, 2)-gram TF-IDF vectorizer."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=50000, sublinear_tf=True)


def build_word_char_tfidf() -> FeatureUnion:
    """Combined Word (1, 2) + Char_wb (2, 5) TF-IDF features."""
    return FeatureUnion([
        ("word", build_word_only_tfidf()),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_df=0.95, max_features=60000, sublinear_tf=True)),
    ])


def extract_tfidf_features(train_texts, val_texts, test_texts, models_dir: Path = MODELS_DIR) -> dict:
    """Fit Word+Char and Word-Only vectorizers on train split and transform all splits."""
    ensure_dirs()
    wc_vec = build_word_char_tfidf()
    wo_vec = build_word_only_tfidf()

    X_train_wc = wc_vec.fit_transform(train_texts)
    X_val_wc = wc_vec.transform(val_texts)
    X_test_wc = wc_vec.transform(test_texts)

    X_train_wo = wo_vec.fit_transform(train_texts)
    X_val_wo = wo_vec.transform(val_texts)
    X_test_wo = wo_vec.transform(test_texts)

    joblib.dump(wc_vec, models_dir / "tfidf_word_char.joblib")
    joblib.dump(wo_vec, models_dir / "tfidf_word_only.joblib")
    logger.info("TF-IDF shapes | word+char=%s | word-only=%s", X_train_wc.shape, X_train_wo.shape)

    return {
        "X_train_wc": X_train_wc, "X_val_wc": X_val_wc, "X_test_wc": X_test_wc,
        "X_train_wo": X_train_wo, "X_val_wo": X_val_wo, "X_test_wo": X_test_wo,
    }


def fit_label_encoder(train_labels, models_dir: Path = MODELS_DIR) -> LabelEncoder:
    """Fit and save LabelEncoder."""
    le = LabelEncoder().fit(sorted(set(train_labels)))
    joblib.dump(le, models_dir / "label_encoder.joblib")
    return le


def build_vocab(train_texts, max_vocab: int = 30000, min_freq: int = 2, models_dir: Path = MODELS_DIR) -> dict[str, int]:
    """Build and save token-to-id vocabulary from training texts."""
    counts = Counter(tok for txt in train_texts for tok in str(txt).split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in counts.most_common(max_vocab - 2):
        if freq < min_freq:
            break
        vocab[tok] = len(vocab)
    (models_dir / "bilstm_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    logger.info("Built BiLSTM vocab: %d tokens", len(vocab))
    return vocab


def encode_sequences(texts, vocab: dict[str, int], max_len: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Convert texts to zero-padded integer sequences and clamped lengths."""
    texts = list(texts)
    seqs = np.zeros((len(texts), max_len), dtype=np.int64)
    lens = np.ones(len(texts), dtype=np.int64)
    for i, txt in enumerate(texts):
        toks = str(txt).split()[:max_len]
        if toks:
            seqs[i, : len(toks)] = [vocab.get(t, 1) for t in toks]
            lens[i] = len(toks)
    return seqs, lens
