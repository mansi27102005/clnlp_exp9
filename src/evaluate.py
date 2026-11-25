"""Test-set evaluation, confusion matrices, error analysis, and single-post prediction."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from src.features import encode_sequences
from src.preprocess import clean_social_text
from src.train_deep import BiLSTMClassifier
from src.utils import MODELS_DIR, RESULTS_DIR, TEXT_COL, ensure_dirs, get_device, get_logger

logger = get_logger("evaluate")


def _predict_classical(model, X):
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:
        probs = softmax(model.decision_function(X), axis=1)
    return preds, probs


def _predict_bilstm(model: BiLSTMClassifier, X_seq: np.ndarray, lengths: np.ndarray, device: torch.device):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_seq), 256):
            bx = torch.tensor(X_seq[i:i + 256], dtype=torch.long, device=device)
            bl = torch.tensor(lengths[i:i + 256], dtype=torch.long, device=device)
            probs.append(torch.softmax(model(bx, bl), dim=1).cpu().numpy())
    prob_arr = np.vstack(probs)
    return prob_arr.argmax(axis=1), prob_arr


def _plot_cm_grid(preds_map: dict, y_test: np.ndarray, classes: list[str], normalize: bool, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, (preds, _)) in zip(axes.flatten(), preds_map.items()):
        cm = confusion_matrix(y_test, preds, normalize="true" if normalize else None)
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="YlGnBu" if normalize else "Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
        ax.set_title(name, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    title = "Confusion Matrices — Normalized" if normalize else "Confusion Matrices — Raw Counts"
    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate_all(
    test_df: pd.DataFrame, y_test: np.ndarray, le, tfidf: dict,
    classical_models: dict, bilstm_model: BiLSTMClassifier, device: torch.device,
    X_test_seq: np.ndarray, len_test: np.ndarray,
    models_dir: Path = MODELS_DIR, results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Evaluate all models on the test set, plot confusion matrices, and save error analysis."""
    ensure_dirs()
    classes = [str(c) for c in le.classes_]

    preds_map = {}
    for name, clf in classical_models.items():
        X_feat = tfidf["X_test_wo"] if "WordOnly" in name else tfidf["X_test_wc"]
        preds_map[name] = _predict_classical(clf, X_feat)
    preds_map["BiLSTM"] = _predict_bilstm(bilstm_model, X_test_seq, len_test, device)

    rows, reports = [], []
    for name, (preds, _) in preds_map.items():
        acc = accuracy_score(y_test, preds)
        p_m, r_m, f_m, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
        p_w, r_w, f_w, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        rows.append({
            "model": name, "accuracy": round(acc, 4),
            "macro_precision": round(p_m, 4), "macro_recall": round(r_m, 4), "macro_f1": round(f_m, 4),
            "weighted_precision": round(p_w, 4), "weighted_recall": round(r_w, 4), "weighted_f1": round(f_w, 4),
        })
        reports.append(f"=== {name} ===\n" + classification_report(y_test, preds, target_names=classes, digits=4, zero_division=0))

    comp_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    comp_df.to_csv(results_dir / "model_comparison.csv", index=False)
    (results_dir / "classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")
    logger.info("\n%s", comp_df.to_string(index=False))

    # Comparison bar chart
    melted = comp_df.melt(id_vars="model", value_vars=["accuracy", "macro_f1", "weighted_f1"], var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Comparison (Test Set)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Raw & normalized confusion matrices for all models + dedicated best-model figure
    _plot_cm_grid(preds_map, y_test, classes, False, results_dir / "confusion_matrices_raw.png")
    _plot_cm_grid(preds_map, y_test, classes, True, results_dir / "confusion_matrices_normalized.png")

    best_name = str(comp_df.iloc[0]["model"])
    best_preds, best_probs = preds_map[best_name]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=False)
    axes[0].set_title(f"{best_name} — Raw Counts")
    sns.heatmap(confusion_matrix(y_test, best_preds, normalize="true"), annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f"{best_name} — Normalized")
    fig.tight_layout()
    fig.savefig(results_dir / "best_model_confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Error analysis: 12 misclassified examples from best model
    mis_idx = np.where(y_test != best_preds)[0]
    err_rows = []
    for idx in mis_idx:
        err_rows.append({
            "true_label": classes[int(y_test[idx])],
            "predicted_label": classes[int(best_preds[idx])],
            "confidence": round(float(best_probs[idx, int(best_preds[idx])]), 4),
            "raw_text": str(test_df.iloc[idx][TEXT_COL])[:220],
        })
    err_df = pd.DataFrame(err_rows).sort_values("confidence", ascending=False).drop_duplicates(subset=["true_label", "predicted_label"]).head(12)
    err_df.to_csv(results_dir / "error_analysis.csv", index=False)
    logger.info("Best model: %s | Error rate: %d/%d (%.2f%%)", best_name, len(mis_idx), len(y_test), len(mis_idx) / len(y_test) * 100)
    return comp_df


def predict_post(raw_text: str, model_choice: str = "best", models_dir: Path = MODELS_DIR, results_dir: Path = RESULTS_DIR) -> dict:
    """Predict emotion category and confidence score for a single raw social media post."""
    le = joblib.load(models_dir / "label_encoder.joblib")
    classes = [str(c) for c in le.classes_]
    cleaned = clean_social_text(raw_text)

    if model_choice == "best":
        comp_path = results_dir / "model_comparison.csv"
        model_choice = str(pd.read_csv(comp_path).iloc[0]["model"]) if comp_path.exists() else "LinearSVM_Tuned"

    if model_choice.lower() == "bilstm":
        device = get_device()
        vocab = json.loads((models_dir / "bilstm_vocab.json").read_text(encoding="utf-8"))
        ckpt = torch.load(models_dir / "bilstm_best.pt", map_location=device, weights_only=True)
        cfg = ckpt["config"]
        model = BiLSTMClassifier(cfg["vocab_size"], cfg["num_classes"], cfg["embed_dim"], cfg["hidden_dim"], cfg["dropout"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        seq, lens = encode_sequences([cleaned], vocab, cfg["max_len"])
        _, probs = _predict_bilstm(model, seq, lens, device)
    else:
        vec = joblib.load(models_dir / ("tfidf_word_only.joblib" if "WordOnly" in model_choice else "tfidf_word_char.joblib"))
        file_map = {"MultinomialNB": "multinomial_nb.joblib", "LogisticRegression": "logistic_regression.joblib", "LinearSVM": "linear_svm.joblib"}
        clf = joblib.load(models_dir / file_map.get(model_choice, "best_classical_tuned.joblib"))
        _, probs = _predict_classical(clf, vec.transform([cleaned]))

    p = probs[0]
    top_idx = int(np.argmax(p))
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "model_used": model_choice,
        "predicted_label": classes[top_idx],
        "confidence": round(float(p[top_idx]), 4),
    }
