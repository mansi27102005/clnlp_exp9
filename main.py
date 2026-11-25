"""Run the full Social Media Content Classification pipeline or predict on a single post."""

import argparse
import json

import numpy as np

from src.data_loader import run_eda
from src.evaluate import evaluate_all, predict_post
from src.features import build_vocab, encode_sequences, extract_tfidf_features, fit_label_encoder
from src.preprocess import preprocess_and_split
from src.train_classical import train_classical_models
from src.train_deep import train_bilstm
from src.utils import LABEL_COL, RANDOM_SEED, ensure_dirs, get_logger, set_seed

logger = get_logger("main")


def run_pipeline(epochs: int = 10) -> None:
    """Execute EDA, preprocessing, feature extraction, training, and evaluation end-to-end."""
    ensure_dirs()
    set_seed(RANDOM_SEED)

    logger.info("--- Stage 1: Data Loading & EDA ---")
    df = run_eda()

    logger.info("--- Stage 2: Preprocessing & Stratified 70/15/15 Split ---")
    train_df, val_df, test_df, class_weights = preprocess_and_split(df)

    logger.info("--- Stage 3: Feature Extraction ---")
    le = fit_label_encoder(train_df[LABEL_COL])
    y_train, y_val, y_test = le.transform(train_df[LABEL_COL]), le.transform(val_df[LABEL_COL]), le.transform(test_df[LABEL_COL])
    tfidf = extract_tfidf_features(train_df["clean_text"], val_df["clean_text"], test_df["clean_text"])

    vocab = build_vocab(train_df["clean_text"])
    X_tr_seq, len_tr = encode_sequences(train_df["clean_text"], vocab)
    X_va_seq, len_va = encode_sequences(val_df["clean_text"], vocab)
    X_te_seq, len_te = encode_sequences(test_df["clean_text"], vocab)

    logger.info("--- Stage 4: Training Classical & Deep Models ---")
    classical_models = train_classical_models(tfidf, y_train, y_val)
    weights_arr = np.array([class_weights[str(c)] for c in le.classes_], dtype=np.float32)
    bilstm, device = train_bilstm(
        X_tr_seq, len_tr, y_train, X_va_seq, len_va, y_val,
        vocab_size=len(vocab), num_classes=len(le.classes_), class_weights=weights_arr, epochs=epochs,
    )

    logger.info("--- Stage 5: Test Evaluation & Sample Prediction ---")
    evaluate_all(test_df, y_test, le, tfidf, classical_models, bilstm, device, X_te_seq, len_te)
    sample = predict_post("Nothing brings me joy anymore, I feel empty every day.")
    logger.info("Sample Prediction: %s", sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Media Content Classification")
    parser.add_argument("--predict", type=str, default=None, help="Classify a raw social media post.")
    parser.add_argument("--model", type=str, default="best", help="Model name for --predict.")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs for BiLSTM.")
    args = parser.parse_args()

    if args.predict is not None:
        print(json.dumps(predict_post(args.predict, args.model), indent=2))
    else:
        run_pipeline(epochs=args.epochs)
