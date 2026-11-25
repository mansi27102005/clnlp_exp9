"""Test-set evaluation, confusion matrices, error analysis, and live post prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.features import load_sequence_vocab, texts_to_padded_sequences
from src.preprocess import clean_social_text
from src.train_deep import BiLSTMClassifier
from src.utils import (
    LABEL_COL,
    MODELS_DIR,
    RESULTS_DIR,
    TEXT_COL,
    ensure_directories,
    get_device,
    get_logger,
    load_json,
    save_json,
)

logger = get_logger(__name__)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _compute_metrics_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_mac), 4),
        "macro_recall": round(float(r_mac), 4),
        "macro_f1": round(float(f1_mac), 4),
        "weighted_precision": round(float(p_wt), 4),
        "weighted_recall": round(float(r_wt), 4),
        "weighted_f1": round(float(f1_wt), 4),
    }


def _classical_probs_and_preds(model: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        probs = softmax(scores, axis=1)
    else:
        n_classes = int(np.max(preds)) + 1
        probs = np.eye(n_classes)[preds]
    return preds, probs


def _bilstm_probs_and_preds(
    model: BiLSTMClassifier,
    X_seq: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_seq), batch_size):
            bx = torch.tensor(X_seq[start : start + batch_size], dtype=torch.long, device=device)
            bl = torch.tensor(lengths[start : start + batch_size], dtype=torch.long, device=device)
            logits = model(bx, bl)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(all_probs)
    preds = probs.argmax(axis=1)
    return preds, probs


def _plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Plot raw-count and row-normalized confusion matrices side by side."""
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title(f"{model_name} — Raw Counts", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis="x", rotation=25)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"{model_name} — Normalized (Recall per Class)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_model_comparison(comp_df: pd.DataFrame, output_path: Path) -> None:
    melted = comp_df.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="model", y="Score", hue="Metric", palette="viridis", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2, rotation=90)

    ax.set_title("Model Comparison on Held-Out Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, min(1.08, melted["Score"].max() + 0.12))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Metric", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _infer_error_reason(raw_text: str, true_label: str, pred_label: str) -> str:
    """Provide an interpretable error-analysis hint for why a post may be ambiguous."""
    words = str(raw_text).split()
    lower = str(raw_text).lower()
    reasons: list[str] = []

    if len(words) <= 6:
        reasons.append("very short post (limited context)")
    if {true_label, pred_label} <= {"Depression", "Sadness", "Anxiety"}:
        reasons.append("overlapping mental-health/negative affect vocabulary")
    if {true_label, pred_label} == {"Positive", "Neutral"} or {true_label, pred_label} == {"Anger", "Sadness"}:
        reasons.append("subtle tone or borderline sentiment intensity")
    if any(tok in lower for tok in ("not ", "no ", "never ", "dont", "cant", "but ", "though")):
        reasons.append("negation or mixed-sentiment clause")
    if any(tok in lower for tok in ("lol", "haha", "lmao", "wkwk", "yay", "yeah right")):
        reasons.append("colloquial slang / potential irony or sarcasm")

    if not reasons:
        reasons.append("topic/emotion overlap or implicit emotional expression")
    return "; ".join(reasons)


def _perform_error_analysis(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    best_model_name: str,
    results_dir: Path,
    n_samples: int = 15,
) -> pd.DataFrame:
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        return pd.DataFrame()

    # Sample across diverse (true, pred) pairs deterministically so multiple failure modes appear
    mis_rows = []
    for idx in mis_idx:
        t_lbl = class_names[int(y_true[idx])]
        p_lbl = class_names[int(y_pred[idx])]
        conf = float(y_probs[idx, int(y_pred[idx])])
        raw = str(test_df.iloc[idx][TEXT_COL])
        clean = str(test_df.iloc[idx]["clean_text"])
        mis_rows.append(
            {
                "test_index": int(idx),
                "true_label": t_lbl,
                "predicted_label": p_lbl,
                "confidence": round(conf, 4),
                "word_count": len(raw.split()),
                "raw_text": raw[:280],
                "clean_text": clean[:240],
                "likely_reason": _infer_error_reason(raw, t_lbl, p_lbl),
            }
        )

    mis_df = pd.DataFrame(mis_rows)
    # Pick top confident mistakes plus diverse confusion pairs
    selected_parts: list[pd.DataFrame] = []
    seen_pairs: set[tuple[str, str]] = set()
    for _, row in mis_df.sort_values("confidence", ascending=False).iterrows():
        pair = (row["true_label"], row["predicted_label"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected_parts.append(pd.DataFrame([row]))
        if len(selected_parts) >= n_samples:
            break

    sampled_df = pd.concat(selected_parts, ignore_index=True)
    if len(sampled_df) < min(n_samples, len(mis_df)):
        remaining = mis_df.loc[~mis_df["test_index"].isin(sampled_df["test_index"])].head(n_samples - len(sampled_df))
        sampled_df = pd.concat([sampled_df, remaining], ignore_index=True)

    sampled_df.to_csv(results_dir / "error_analysis.csv", index=False)
    save_json(
        {
            "best_model": best_model_name,
            "total_test_samples": len(y_true),
            "total_misclassified": int(len(mis_idx)),
            "error_rate": round(float(len(mis_idx) / len(y_true)), 4),
            "samples": sampled_df.to_dict(orient="records"),
        },
        results_dir / "error_analysis.json",
    )

    logger.info("=== Error Analysis (%s | %d/%d misclassified, %.2f%% error rate) ===", best_model_name, len(mis_idx), len(y_true), len(mis_idx) / len(y_true) * 100)
    for i, row in sampled_df.iterrows():
        logger.info(
            "  [%02d] True=%-10s | Pred=%-10s (conf=%.2f) | Reason=%s\n       Post: %s",
            i + 1,
            row["true_label"],
            row["predicted_label"],
            row["confidence"],
            row["likely_reason"],
            row["raw_text"][:140],
        )
    return sampled_df


def evaluate_all_models(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: Any,
    tfidf_artifacts: dict[str, Any],
    classical_models: dict[str, Any],
    deep_artifacts: dict[str, Any],
    X_test_seq: np.ndarray,
    len_test: np.ndarray,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Evaluate all classical & deep models on held-out test set, save metrics, CMs, and error analysis."""
    ensure_directories()
    class_names = [str(c) for c in label_encoder.classes_]

    X_test_wc = tfidf_artifacts["X_test_wc"]
    X_test_wo = tfidf_artifacts["X_test_wo"]

    predictions_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    comparison_rows: list[dict[str, Any]] = []
    reports_dict: dict[str, Any] = {}
    text_reports: list[str] = []

    # 1. Evaluate all classical models (Word+Char baselines, Word-Only ablation, Tuned best)
    for model_name, model_obj in classical_models.items():
        X_feat = X_test_wo if "WordOnly" in model_name else X_test_wc
        preds, probs = _classical_probs_and_preds(model_obj, X_feat)
        predictions_cache[model_name] = (preds, probs)

        row = _compute_metrics_row(model_name, y_test, preds)
        comparison_rows.append(row)
        reports_dict[model_name] = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
        text_rep = classification_report(y_test, preds, target_names=class_names, digits=4, zero_division=0)
        text_reports.append(f"=== {model_name} ===\n{text_rep}\n")

        cm_path = results_dir / f"cm_{_slugify(model_name)}.png"
        _plot_confusion_matrices(y_test, preds, class_names, model_name, cm_path)

    # 2. Evaluate BiLSTM
    bilstm_name = "BiLSTM"
    bilstm_preds, bilstm_probs = _bilstm_probs_and_preds(
        deep_artifacts["model"],
        X_test_seq,
        len_test,
        deep_artifacts["device"],
    )
    predictions_cache[bilstm_name] = (bilstm_preds, bilstm_probs)
    comparison_rows.append(_compute_metrics_row(bilstm_name, y_test, bilstm_preds))
    reports_dict[bilstm_name] = classification_report(y_test, bilstm_preds, target_names=class_names, output_dict=True, zero_division=0)
    text_reports.append(
        f"=== {bilstm_name} ===\n"
        + classification_report(y_test, bilstm_preds, target_names=class_names, digits=4, zero_division=0)
        + "\n"
    )
    _plot_confusion_matrices(y_test, bilstm_preds, class_names, bilstm_name, results_dir / "cm_bilstm.png")

    # 3. Build comparison table sorted by primary metric (macro_f1)
    comp_df = pd.DataFrame(comparison_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    comp_df.to_csv(results_dir / "model_comparison.csv", index=False)
    save_json({"models": comp_df.to_dict(orient="records")}, results_dir / "model_comparison.json")
    save_json(reports_dict, results_dir / "classification_reports.json")
    (results_dir / "classification_reports.txt").write_text("\n".join(text_reports), encoding="utf-8")

    _plot_model_comparison(comp_df, results_dir / "model_comparison.png")

    best_model_name = str(comp_df.iloc[0]["model"])
    best_macro_f1 = float(comp_df.iloc[0]["macro_f1"])
    best_acc = float(comp_df.iloc[0]["accuracy"])

    # Persist metadata of winning model for live predict_post()
    best_preds, best_probs = predictions_cache[best_model_name]
    _plot_confusion_matrices(y_test, best_preds, class_names, best_model_name, results_dir / "best_model_confusion_matrix.png")

    save_json(
        {
            "best_model_name": best_model_name,
            "best_macro_f1": best_macro_f1,
            "best_accuracy": best_acc,
            "class_names": class_names,
        },
        models_dir / "best_model_meta.json",
    )

    logger.info("=== Held-Out Test Set Comparison (Primary Metric: Macro-F1) ===")
    logger.info("\n%s", comp_df.to_string(index=False))
    logger.info("Winning Model: %s (macro_f1=%.4f, accuracy=%.4f)", best_model_name, best_macro_f1, best_acc)

    error_df = _perform_error_analysis(
        test_df=test_df,
        y_true=y_test,
        y_pred=best_preds,
        y_probs=best_probs,
        class_names=class_names,
        best_model_name=best_model_name,
        results_dir=results_dir,
        n_samples=15,
    )

    return {
        "comparison_df": comp_df,
        "best_model_name": best_model_name,
        "error_analysis_df": error_df,
    }


def predict_post(
    raw_text: str,
    model_choice: str = "best",
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Predict category and confidence score for a single raw social media post."""
    le = joblib.load(models_dir / "label_encoder.joblib")
    class_names = [str(c) for c in le.classes_]
    cleaned = clean_social_text(raw_text)

    if model_choice == "best":
        meta_path = models_dir / "best_model_meta.json"
        if meta_path.exists():
            model_choice = str(load_json(meta_path)["best_model_name"])
        else:
            model_choice = "best_classical_tuned"

    if model_choice.lower() == "bilstm":
        device = get_device()
        vocab = load_sequence_vocab(models_dir)
        ckpt = torch.load(models_dir / "bilstm_best.pt", map_location=device, weights_only=True)
        cfg = ckpt["config"]
        model = BiLSTMClassifier(
            vocab_size=cfg["vocab_size"],
            num_classes=cfg["num_classes"],
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        seq, lengths = texts_to_padded_sequences([cleaned], vocab, max_len=cfg["max_len"])
        _, probs = _bilstm_probs_and_preds(model, seq, lengths, device)
        prob_vec = probs[0]
    else:
        if "WordOnly" in model_choice:
            vec = joblib.load(models_dir / "tfidf_word_only.joblib")
            clf = joblib.load(models_dir / "word_only_ablation.joblib")
        else:
            vec = joblib.load(models_dir / "tfidf_word_char.joblib")
            model_file_map = {
                "MultinomialNB": "multinomial_nb.joblib",
                "LogisticRegression": "logistic_regression.joblib",
                "LinearSVM": "linear_svm.joblib",
            }
            fname = model_file_map.get(model_choice, "best_classical_tuned.joblib")
            clf = joblib.load(models_dir / fname)

        X_vec = vec.transform([cleaned])
        _, probs = _classical_probs_and_preds(clf, X_vec)
        prob_vec = probs[0]

    pred_idx = int(np.argmax(prob_vec))
    ranked_probs = {
        class_names[i]: round(float(prob_vec[i]), 4)
        for i in np.argsort(prob_vec)[::-1]
    }
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "model_used": model_choice,
        "predicted_label": class_names[pred_idx],
        "confidence": round(float(prob_vec[pred_idx]), 4),
        "class_probabilities": ranked_probs,
    }
