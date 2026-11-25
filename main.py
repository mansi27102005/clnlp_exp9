"""End-to-end execution entry point for the Social Media Content Classification pipeline."""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.data_loader import run_eda
from src.evaluate import evaluate_all_models, predict_post
from src.features import (
    build_sequence_vocab,
    fit_label_encoder,
    fit_transform_tfidf,
    texts_to_padded_sequences,
)
from src.preprocess import preprocess_and_split
from src.train_classical import train_classical_models
from src.train_deep import train_deep_model
from src.utils import LABEL_COL, RANDOM_SEED, ensure_directories, get_logger, set_seed

logger = get_logger("main")


def run_full_pipeline(max_len: int = 128, max_epochs: int = 10) -> None:
    """Run all 5 stages end-to-end with deterministic seeding."""
    ensure_directories()
    set_seed(RANDOM_SEED)

    logger.info("============================================================")
    logger.info("STAGE 1: Data Loading, Auditing & Exploratory Data Analysis")
    logger.info("============================================================")
    df, eda_summary = run_eda()

    logger.info("============================================================")
    logger.info("STAGE 2: Social-Media Preprocessing & Stratified 70/15/15 Split")
    logger.info("============================================================")
    train_df, val_df, test_df, class_weights = preprocess_and_split(df)

    logger.info("============================================================")
    logger.info("STAGE 3: Feature Extraction (TF-IDF Word+Char & BiLSTM Seq)")
    logger.info("============================================================")
    le = fit_label_encoder(train_df[LABEL_COL])
    y_train = le.transform(train_df[LABEL_COL])
    y_val = le.transform(val_df[LABEL_COL])
    y_test = le.transform(test_df[LABEL_COL])

    tfidf_artifacts = fit_transform_tfidf(
        train_df["clean_text"],
        val_df["clean_text"],
        test_df["clean_text"],
    )

    vocab = build_sequence_vocab(train_df["clean_text"], max_vocab_size=30000, min_freq=2)
    X_train_seq, len_train = texts_to_padded_sequences(train_df["clean_text"], vocab, max_len=max_len)
    X_val_seq, len_val = texts_to_padded_sequences(val_df["clean_text"], vocab, max_len=max_len)
    X_test_seq, len_test = texts_to_padded_sequences(test_df["clean_text"], vocab, max_len=max_len)

    logger.info("============================================================")
    logger.info("STAGE 4A: Classical Models, Ablation & GridSearchCV Tuning")
    logger.info("============================================================")
    classical_out = train_classical_models(tfidf_artifacts, y_train, y_val)

    logger.info("============================================================")
    logger.info("STAGE 4B: Deep Learning Model (GPU check -> PyTorch BiLSTM)")
    logger.info("============================================================")
    ordered_weights = np.array([class_weights[str(cls)] for cls in le.classes_], dtype=np.float32)
    deep_out = train_deep_model(
        X_train_seq=X_train_seq,
        len_train=len_train,
        y_train=y_train,
        X_val_seq=X_val_seq,
        len_val=len_val,
        y_val=y_val,
        vocab_size=len(vocab),
        num_classes=len(le.classes_),
        class_weights_ordered=ordered_weights,
        max_len=max_len,
        max_epochs=max_epochs,
    )

    logger.info("============================================================")
    logger.info("STAGE 5: Test-Set Evaluation, Error Analysis & Sample Inference")
    logger.info("============================================================")
    eval_out = evaluate_all_models(
        test_df=test_df,
        y_test=y_test,
        label_encoder=le,
        tfidf_artifacts=tfidf_artifacts,
        classical_models=classical_out["models"],
        deep_artifacts=deep_out,
        X_test_seq=X_test_seq,
        len_test=len_test,
    )

    sample_posts = [
        "I feel so overwhelmed and anxious about my exams tomorrow :( #stressed",
        "Nothing brings me joy anymore, I just lie in bed feeling empty every day.",
        "I am SO happy and grateful for my friends today!! 😊 #blessed",
    ]
    logger.info("=== Sample Live Post Predictions (Best Model: %s) ===", eval_out["best_model_name"])
    for post in sample_posts:
        pred = predict_post(post)
        logger.info("  Post: %s -> %s (conf=%.4f)", post, pred["predicted_label"], pred["confidence"])

    logger.info("Pipeline finished successfully! Dataset size=%d, imbalance=%.2fx", len(df), eda_summary["imbalance_ratio"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Social Media Content Classification Pipeline")
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Predict the emotion/mental-health category for a raw social media post.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best",
        help="Model to use with --predict ('best', 'bilstm', 'LogisticRegression', 'LinearSVM', 'MultinomialNB').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum epochs for BiLSTM training (early stopping applies).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict is not None:
        result = predict_post(args.predict, model_choice=args.model)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        run_full_pipeline(max_epochs=args.epochs)
