"""Deep learning training pipeline (GPU check -> BiLSTM with class weights & early stopping)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.utils import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, ensure_directories, get_device, get_logger, save_json

logger = get_logger(__name__)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with masked mean+max pooling for multiclass text classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_drop = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_drop = nn.Dropout(dropout)
        # Concatenating mean-pooled and max-pooled bidirectional outputs -> 4 * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embed_drop(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))

        # Build padding mask: True for real tokens, False for PAD
        batch_size, seq_len, _ = lstm_out.shape
        arange = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        mask = arange < lengths.unsqueeze(1)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        mean_pool = (lstm_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        # Masked max pooling
        masked_for_max = lstm_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool = masked_for_max.max(dim=1).values

        features = self.head_drop(torch.cat([mean_pool, max_pool], dim=1))
        return self.fc(features)


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_len, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_len = batch_len.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x, batch_len)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    avg_loss = total_loss / len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return avg_loss, acc, macro_f1, y_pred, y_prob


def _plot_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss", linewidth=2)
    axes[0].set_title("BiLSTM Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], marker="s", label="Val Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_macro_f1"], marker="^", linestyle="--", label="Val Macro-F1", linewidth=2)
    axes[1].set_title("BiLSTM Accuracy & Macro-F1 Curves", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def train_deep_model(
    X_train_seq: np.ndarray,
    len_train: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    len_val: np.ndarray,
    y_val: np.ndarray,
    vocab_size: int,
    num_classes: int,
    class_weights_ordered: np.ndarray,
    max_len: int = 128,
    batch_size: int = 256,
    max_epochs: int = 10,
    patience: int = 3,
    lr: float = 2e-3,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Check hardware, train BiLSTM on CPU (or GPU if present) with early stopping, and save artifacts."""
    ensure_directories()
    device = get_device()
    if device.type == "cuda":
        logger.info("GPU detected (%s). Training PyTorch BiLSTM on CUDA.", torch.cuda.get_device_name(0))
    else:
        logger.info("No GPU detected (CPU-only environment). Defaulting to PyTorch BiLSTM as configured.")

    generator = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.long),
        torch.tensor(len_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.long),
        torch.tensor(len_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model_cfg = {
        "vocab_size": vocab_size,
        "num_classes": num_classes,
        "embed_dim": 64,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.35,
        "max_len": max_len,
    }
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    weight_tensor = torch.tensor(class_weights_ordered, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    ckpt_path = models_dir / "bilstm_best.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_len, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_len = batch_len.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x, batch_len)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total += batch_x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc, val_macro_f1, _, _ = _evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(val_macro_f1)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_f1"].append(round(val_macro_f1, 4))

        logger.info(
            "BiLSTM Epoch %02d/%02d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            max_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_f1,
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model_cfg,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered at epoch %d (best epoch=%d, val_macro_f1=%.4f).", epoch, best_epoch, best_val_f1)
                break

    # Reload best weights before returning
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    _plot_training_curves(history, results_dir / "bilstm_training_curves.png")
    save_json(
        {
            "device": str(device),
            "config": model_cfg,
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_f1, 4),
            "history": history,
        },
        results_dir / "bilstm_history.json",
    )
    logger.info("BiLSTM training complete. Best checkpoint saved to %s", ckpt_path)
    return {
        "model": model,
        "device": device,
        "config": model_cfg,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "history": history,
    }
