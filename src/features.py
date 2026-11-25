"""Feature extraction for classical ML (TF-IDF word + char n-grams) and deep learning (BiLSTM sequences)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder

from src.utils import MODELS_DIR, ensure_directories, get_logger, load_json, save_json

logger = get_logger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def build_word_only_vectorizer() -> TfidfVectorizer:
    """Create a word (1, 2)-gram TF-IDF vectorizer for ablation comparison."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        token_pattern=r"(?u)<URL>|<USER>|\b[a-z0-9]{1,}\b",
    )


def build_word_char_vectorizer() -> FeatureUnion:
    """Combine word (1, 2)-grams with character_wb (2, 5)-grams for slang/typo robustness."""
    word_vec = build_word_only_vectorizer()
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        max_df=0.95,
        max_features=60000,
        sublinear_tf=True,
    )
    return FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vec),
            ("char_tfidf", char_vec),
        ]
    )


def fit_transform_tfidf(
    train_texts: Iterable[str],
    val_texts: Iterable[str],
    test_texts: Iterable[str],
    models_dir: Path = MODELS_DIR,
) -> dict[str, object]:
    """Fit both word+char and word-only TF-IDF vectorizers on train split and transform all splits."""
    ensure_directories()
    train_list = list(train_texts)
    val_list = list(val_texts)
    test_list = list(test_texts)

    logger.info("Fitting combined Word (1-2) + Char_wb (2-5) TF-IDF vectorizer...")
    wc_vec = build_word_char_vectorizer()
    X_train_wc: csr_matrix = wc_vec.fit_transform(train_list)
    X_val_wc: csr_matrix = wc_vec.transform(val_list)
    X_test_wc: csr_matrix = wc_vec.transform(test_list)
    joblib.dump(wc_vec, models_dir / "tfidf_word_char.joblib")

    logger.info("Fitting Word-only (1-2) TF-IDF vectorizer for ablation...")
    wo_vec = build_word_only_vectorizer()
    X_train_wo: csr_matrix = wo_vec.fit_transform(train_list)
    X_val_wo: csr_matrix = wo_vec.transform(val_list)
    X_test_wo: csr_matrix = wo_vec.transform(test_list)
    joblib.dump(wo_vec, models_dir / "tfidf_word_only.joblib")

    logger.info(
        "TF-IDF matrices ready | word+char shape=%s | word-only shape=%s",
        X_train_wc.shape,
        X_train_wo.shape,
    )
    return {
        "word_char_vectorizer": wc_vec,
        "word_only_vectorizer": wo_vec,
        "X_train_wc": X_train_wc,
        "X_val_wc": X_val_wc,
        "X_test_wc": X_test_wc,
        "X_train_wo": X_train_wo,
        "X_val_wo": X_val_wo,
        "X_test_wo": X_test_wo,
    }


def fit_label_encoder(train_labels: Iterable[str], models_dir: Path = MODELS_DIR) -> LabelEncoder:
    """Fit and persist a deterministic LabelEncoder on sorted training labels."""
    ensure_directories()
    le = LabelEncoder()
    le.fit(sorted(set(train_labels)))
    joblib.dump(le, models_dir / "label_encoder.joblib")
    logger.info("LabelEncoder fitted with classes: %s", list(le.classes_))
    return le


def build_sequence_vocab(
    train_texts: Iterable[str],
    max_vocab_size: int = 30000,
    min_freq: int = 2,
    models_dir: Path = MODELS_DIR,
) -> dict[str, int]:
    """Build and save token-to-index vocabulary strictly from training texts."""
    ensure_directories()
    counter: Counter[str] = Counter()
    for text in train_texts:
        counter.update(str(text).split())

    vocab: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common(max_vocab_size - 2):
        if freq < min_freq:
            break
        vocab[token] = len(vocab)

    save_json(vocab, models_dir / "bilstm_vocab.json")
    logger.info("Built BiLSTM vocabulary: %d tokens (min_freq=%d, max=%d)", len(vocab), min_freq, max_vocab_size)
    return vocab


def load_sequence_vocab(models_dir: Path = MODELS_DIR) -> dict[str, int]:
    """Load persisted BiLSTM token-to-index vocabulary."""
    return {str(k): int(v) for k, v in load_json(models_dir / "bilstm_vocab.json").items()}


def texts_to_padded_sequences(
    texts: Iterable[str],
    vocab: dict[str, int],
    max_len: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert preprocessed texts into padded integer sequences and clamped sequence lengths."""
    text_list = list(texts)
    sequences = np.full((len(text_list), max_len), PAD_IDX, dtype=np.int64)
    lengths = np.ones(len(text_list), dtype=np.int64)

    for idx, text in enumerate(text_list):
        tokens = str(text).split()[:max_len]
        if tokens:
            encoded = [vocab.get(tok, UNK_IDX) for tok in tokens]
            sequences[idx, : len(encoded)] = encoded
            lengths[idx] = len(encoded)

    return sequences, lengths
