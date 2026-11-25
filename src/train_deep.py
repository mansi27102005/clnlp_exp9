"""PyTorch BiLSTM training with hardware check, class weights, and early stopping."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.utils import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, ensure_dirs, get_device, get_logger

logger = get_logger("train_deep")


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with masked mean+max pooling."""

    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=x.size(1))

        mask = torch.arange(out.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mean_pool = (out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        max_pool = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return self.fc(self.drop(torch.cat([mean_pool, max_pool], dim=1)))


def train_bilstm(
    X_tr, len_tr, y_tr, X_va, len_va, y_va,
    vocab_size: int, num_classes: int, class_weights: np.ndarray,
    max_len: int = 128, batch_size: int = 256, epochs: int = 10, patience: int = 3,
    models_dir: Path = MODELS_DIR, results_dir: Path = RESULTS_DIR,
) -> tuple[BiLSTMClassifier, torch.device]:
    """Train BiLSTM on GPU if available else CPU, with early stopping and loss/accuracy curves."""
    ensure_dirs()
    device = get_device()
    logger.info("Training BiLSTM on %s (GPU available=%s)", device, torch.cuda.is_available())

    gen = torch.Generator().manual_seed(RANDOM_SEED)
    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(len_tr), torch.tensor(y_tr)), batch_size=batch_size, shuffle=True, generator=gen)
    va_loader = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(len_va), torch.tensor(y_va)), batch_size=batch_size)

    cfg = {"vocab_size": vocab_size, "num_classes": num_classes, "embed_dim": 64, "hidden_dim": 64, "dropout": 0.35, "max_len": max_len}
    model = BiLSTMClassifier(vocab_size, num_classes, cfg["embed_dim"], cfg["hidden_dim"], cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1, wait, ckpt = -1.0, 0, models_dir / "bilstm_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, total = 0.0, 0, 0
        for bx, bl, by in tr_loader:
            bx, bl, by = bx.to(device), bl.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx, bl)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
            tr_correct += int((logits.argmax(1) == by).sum())
            total += bx.size(0)

        model.eval()
        va_loss, preds, targets = 0.0, [], []
        with torch.no_grad():
            for bx, bl, by in va_loader:
                bx, bl, by = bx.to(device), bl.to(device), by.to(device)
                logits = model(bx, bl)
                va_loss += criterion(logits, by).item() * bx.size(0)
                preds.append(logits.argmax(1).cpu().numpy())
                targets.append(by.cpu().numpy())

        y_true, y_pred = np.concatenate(targets), np.concatenate(preds)
        va_acc = accuracy_score(y_true, y_pred)
        va_f1 = f1_score(y_true, y_pred, average="macro")
        hist["train_loss"].append(tr_loss / total)
        hist["val_loss"].append(va_loss / len(y_true))
        hist["train_acc"].append(tr_correct / total)
        hist["val_acc"].append(va_acc)

        logger.info("Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f",
                    ep, epochs, hist["train_loss"][-1], hist["val_loss"][-1], va_acc, va_f1)

        if va_f1 > best_f1:
            best_f1, wait = va_f1, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg}, ckpt)
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d (best val_macro_f1=%.4f)", ep, best_f1)
                break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["state_dict"])

    # Plot training/validation curves
    eps = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(eps, hist["train_loss"], "o-", label="Train")
    ax[0].plot(eps, hist["val_loss"], "s-", label="Val")
    ax[0].set_title("BiLSTM Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(eps, hist["train_acc"], "o-", label="Train")
    ax[1].plot(eps, hist["val_acc"], "s-", label="Val")
    ax[1].set_title("BiLSTM Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "bilstm_training_curves.png", dpi=180)
    plt.close(fig)
    return model, device
