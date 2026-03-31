from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    stability: float


def stability_score(y_seq: np.ndarray) -> float:
    """
    A simple stability metric: fraction of consecutive predictions that stay the same.
    """

    y_seq = np.asarray(y_seq)
    if y_seq.size <= 1:
        return 1.0
    return float(np.mean(y_seq[1:] == y_seq[:-1]))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

