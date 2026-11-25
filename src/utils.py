"""Shared paths, seed setup, device detection, and logging."""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

RANDOM_SEED = 42
TEXT_COL = "Text"
LABEL_COL = "Emotion"

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Mental_Health_6Class_Final_Cleaned.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"


def ensure_dirs() -> None:
    """Create output folders if missing."""
    for d in (RAW_DATA_PATH.parent, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducible results."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "nlp") -> logging.Logger:
    """Return a simple stdout logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
