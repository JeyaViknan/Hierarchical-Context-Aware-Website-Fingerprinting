from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class BaselineConfig:
    """
    Baseline WF model config.

    We use multinomial logistic regression as a strong, fast classical baseline
    for tabular features.
    """

    c: float = 2.0
    max_iter: int = 1000


def make_baseline_model(cfg: BaselineConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    C=cfg.c,
                    max_iter=cfg.max_iter,
                    multi_class="auto",
                    n_jobs=None,
                ),
            ),
        ]
    )


def predict_topk(
    model: Pipeline, X: np.ndarray, class_names: List[str], k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (topk_indices, topk_probs) for each row.
    """

    proba = model.predict_proba(X)
    topk_idx = np.argsort(-proba, axis=1)[:, :k]
    topk_prob = np.take_along_axis(proba, topk_idx, axis=1)
    return topk_idx, topk_prob

