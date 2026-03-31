"""
Evaluation metrics for the HC-WF pipeline.

Provides standard classification metrics plus domain-specific metrics
for website fingerprinting and behavioral intent classification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    task_name: str = "classification",
) -> Dict[str, Any]:
    """
    Comprehensive classification metrics.

    Parameters
    ----------
    y_true : array of shape (N,), ground-truth integer labels
    y_pred : array of shape (N,), predicted integer labels
    y_prob : optional array of shape (N, C), predicted probabilities
    class_names : optional list of class name strings
    task_name : label prefix for the returned dict keys

    Returns
    -------
    dict with accuracy, precision, recall, F1, and optionally ROC-AUC
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, Any] = {
        f"{task_name}/accuracy": float(accuracy_score(y_true, y_pred)),
        f"{task_name}/precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{task_name}/recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{task_name}/f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{task_name}/f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    # ROC-AUC (requires probability predictions and >1 class)
    if y_prob is not None:
        n_classes = y_prob.shape[1] if y_prob.ndim == 2 else len(np.unique(y_true))
        if n_classes > 1:
            try:
                metrics[f"{task_name}/roc_auc_ovr"] = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro"
                    )
                )
            except ValueError:
                # Can happen if a class is missing from y_true
                metrics[f"{task_name}/roc_auc_ovr"] = float("nan")

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Return a sklearn-style classification report string."""
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )


def stability_score(y_seq: np.ndarray) -> float:
    """
    Fraction of consecutive predictions that agree within a session.
    A higher value indicates temporally consistent predictions.
    """
    y_seq = np.asarray(y_seq)
    if y_seq.size <= 1:
        return 1.0
    return float(np.mean(y_seq[1:] == y_seq[:-1]))


def session_accuracy(
    y_true_sessions: List[np.ndarray],
    y_pred_sessions: List[np.ndarray],
) -> float:
    """
    Accuracy computed at the session level: a session is correct if
    all its constituent trace predictions are correct.
    """
    correct = 0
    for yt, yp in zip(y_true_sessions, y_pred_sessions):
        if np.array_equal(np.asarray(yt), np.asarray(yp)):
            correct += 1
    return correct / max(len(y_true_sessions), 1)
