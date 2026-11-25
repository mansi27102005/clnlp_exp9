"""Shared configuration, reproducibility helpers, and logging utilities."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Global reproducibility seed
RANDOM_SEED: int = 42

# Dataset schema constants (verified against Mental_Health_6Class_Final_Cleaned.csv)
TEXT_COL: str = "Text"
LABEL_COL: str = "Emotion"
RAW_FILENAME: str = "Mental_Health_6Class_Final_Cleaned.csv"

# Repository paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_DATA_PATH: Path = RAW_DATA_DIR / RAW_FILENAME
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


def ensure_directories() -> None:
    """Create all standard project directories if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        NOTEBOOKS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "social_media_nlp", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_device() -> torch.device:
    """Detect GPU availability and return the appropriate PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist a dictionary to a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
