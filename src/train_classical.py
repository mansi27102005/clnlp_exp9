"""Classical ML baselines, Word-only ablation, and GridSearchCV tuning."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.utils import MODELS_DIR, RANDOM_SEED, ensure_dirs, get_logger

logger = get_logger("train_classical")


def train_classical_models(tfidf: dict, y_train: np.ndarray, y_val: np.ndarray, models_dir: Path = MODELS_DIR) -> dict:
    """Train NB, Logistic Regression, Linear SVM, Word-only ablation, and tune the best model."""
    ensure_dirs()
    X_tr, X_va = tfidf["X_train_wc"], tfidf["X_val_wc"]

    candidates = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED),
        "LinearSVM": LinearSVC(class_weight="balanced", max_iter=3000, random_state=RANDOM_SEED),
    }

    models, val_f1 = {}, {}
    for name, clf in candidates.items():
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_va)
        val_f1[name] = f1_score(y_val, preds, average="macro")
        models[name] = clf
        logger.info("%s -> val_acc=%.4f | val_macro_f1=%.4f", name, accuracy_score(y_val, preds), val_f1[name])

    best_name = max(val_f1, key=val_f1.get)

    # Ablation: Word-only TF-IDF vs Word+Char TF-IDF on the winning baseline
    if best_name == "LinearSVM":
        ablation = LinearSVC(class_weight="balanced", max_iter=3000, random_state=RANDOM_SEED)
        tune_est, grid = LinearSVC(class_weight="balanced", max_iter=4000, random_state=RANDOM_SEED), {"C": [0.25, 0.5, 1.0, 2.0]}
    elif best_name == "LogisticRegression":
        ablation = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
        tune_est, grid = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED), {"C": [0.5, 1.0, 2.0]}
    else:
        ablation = MultinomialNB(alpha=0.5)
        tune_est, grid = MultinomialNB(), {"alpha": [0.1, 0.5, 1.0]}

    ablation.fit(tfidf["X_train_wo"], y_train)
    models[f"{best_name}_WordOnly"] = ablation
    logger.info("%s_WordOnly -> val_macro_f1=%.4f", best_name, f1_score(y_val, ablation.predict(tfidf["X_val_wo"]), average="macro"))

    # Light hyperparameter tuning with 3-fold CV scored on macro-F1
    gs = GridSearchCV(tune_est, grid, cv=3, scoring="f1_macro", n_jobs=-1, refit=True)
    gs.fit(X_tr, y_train)
    tuned = gs.best_estimator_
    models[f"{best_name}_Tuned"] = tuned
    logger.info("GridSearchCV (%s) -> best_params=%s | cv_macro_f1=%.4f", best_name, gs.best_params_, gs.best_score_)

    # Save models
    joblib.dump(models["MultinomialNB"], models_dir / "multinomial_nb.joblib")
    joblib.dump(models["LogisticRegression"], models_dir / "logistic_regression.joblib")
    joblib.dump(models["LinearSVM"], models_dir / "linear_svm.joblib")
    joblib.dump(ablation, models_dir / "word_only_ablation.joblib")
    joblib.dump(tuned, models_dir / "best_classical_tuned.joblib")
    return models
