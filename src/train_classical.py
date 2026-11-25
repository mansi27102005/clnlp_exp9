"""Classical ML baseline training, TF-IDF ablation, and GridSearchCV tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.utils import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, ensure_directories, get_logger, save_json

logger = get_logger(__name__)


def _val_scores(model: Any, X_val: csr_matrix, y_val: np.ndarray) -> dict[str, float]:
    preds = model.predict(X_val)
    return {
        "val_accuracy": round(float(accuracy_score(y_val, preds)), 4),
        "val_macro_f1": round(float(f1_score(y_val, preds, average="macro")), 4),
        "val_weighted_f1": round(float(f1_score(y_val, preds, average="weighted")), 4),
    }


def train_classical_models(
    tfidf_artifacts: dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Train NB, Logistic Regression, Linear SVM, run word-only ablation, and tune best model."""
    ensure_directories()

    X_train_wc = tfidf_artifacts["X_train_wc"]
    X_val_wc = tfidf_artifacts["X_val_wc"]
    X_train_wo = tfidf_artifacts["X_train_wo"]
    X_val_wo = tfidf_artifacts["X_val_wo"]

    baselines = {
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=seed,
        ),
        "LinearSVM": LinearSVC(
            class_weight="balanced",
            max_iter=3000,
            C=1.0,
            random_state=seed,
        ),
    }

    trained_models: dict[str, Any] = {}
    val_metrics: dict[str, dict[str, float]] = {}

    for name, clf in baselines.items():
        logger.info("Training classical baseline: %s (Word+Char TF-IDF)...", name)
        clf.fit(X_train_wc, y_train)
        scores = _val_scores(clf, X_val_wc, y_val)
        trained_models[name] = clf
        val_metrics[name] = scores
        logger.info(
            "  %s -> val_acc=%.4f | val_macro_f1=%.4f | val_weighted_f1=%.4f",
            name,
            scores["val_accuracy"],
            scores["val_macro_f1"],
            scores["val_weighted_f1"],
        )

    joblib.dump(trained_models["MultinomialNB"], models_dir / "multinomial_nb.joblib")
    joblib.dump(trained_models["LogisticRegression"], models_dir / "logistic_regression.joblib")
    joblib.dump(trained_models["LinearSVM"], models_dir / "linear_svm.joblib")

    # Select best baseline on validation macro-F1
    best_name = max(val_metrics, key=lambda k: val_metrics[k]["val_macro_f1"])
    logger.info("Best baseline on validation macro-F1: %s (%.4f)", best_name, val_metrics[best_name]["val_macro_f1"])

    # Small ablation: Word-only TF-IDF vs Word+Char TF-IDF on the winning baseline architecture
    if best_name == "LinearSVM":
        ablation_clf = LinearSVC(class_weight="balanced", max_iter=3000, C=1.0, random_state=seed)
        tune_estimator = LinearSVC(class_weight="balanced", max_iter=4000, random_state=seed)
        param_grid = {"C": [0.25, 0.5, 1.0, 2.0]}
    elif best_name == "LogisticRegression":
        ablation_clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=seed)
        tune_estimator = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        param_grid = {"C": [0.5, 1.0, 2.0, 4.0]}
    else:
        ablation_clf = MultinomialNB(alpha=0.5)
        tune_estimator = MultinomialNB()
        param_grid = {"alpha": [0.1, 0.25, 0.5, 1.0]}

    ablation_key = f"{best_name}_WordOnly_Ablation"
    logger.info("Running TF-IDF ablation: %s (Word-only 1-2 n-grams)...", ablation_key)
    ablation_clf.fit(X_train_wo, y_train)
    ablation_scores = _val_scores(ablation_clf, X_val_wo, y_val)
    trained_models[ablation_key] = ablation_clf
    val_metrics[ablation_key] = ablation_scores
    joblib.dump(ablation_clf, models_dir / "word_only_ablation.joblib")
    logger.info(
        "  %s -> val_acc=%.4f | val_macro_f1=%.4f",
        ablation_key,
        ablation_scores["val_accuracy"],
        ablation_scores["val_macro_f1"],
    )

    # Light GridSearchCV (cv=3, scoring='f1_macro') on the best classical model
    logger.info("Running GridSearchCV (cv=3, scoring='f1_macro') on %s with %s...", best_name, param_grid)
    grid = GridSearchCV(
        estimator=tune_estimator,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train_wc, y_train)
    tuned_key = f"{best_name}_Tuned"
    tuned_model = grid.best_estimator_
    tuned_scores = _val_scores(tuned_model, X_val_wc, y_val)
    trained_models[tuned_key] = tuned_model
    val_metrics[tuned_key] = tuned_scores
    joblib.dump(tuned_model, models_dir / "best_classical_tuned.joblib")

    logger.info(
        "GridSearchCV complete | best_params=%s | cv_macro_f1=%.4f | val_macro_f1=%.4f",
        grid.best_params_,
        grid.best_score_,
        tuned_scores["val_macro_f1"],
    )

    summary = {
        "best_baseline_name": best_name,
        "tuned_model_name": tuned_key,
        "ablation_model_name": ablation_key,
        "gridsearch_best_params": grid.best_params_,
        "gridsearch_cv_macro_f1": round(float(grid.best_score_), 4),
        "validation_metrics": val_metrics,
    }
    save_json(summary, results_dir / "classical_training_summary.json")
    return {
        "models": trained_models,
        "summary": summary,
    }
