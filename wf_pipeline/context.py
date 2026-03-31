from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ContextConfig:
    """
    Context model config for sequence smoothing.

    stay_prob: probability of remaining in the same class between consecutive traces.
    eps: small probability floor to avoid zeros.
    """

    stay_prob: float = 0.85
    eps: float = 1e-6


def make_transition_matrix(n_classes: int, cfg: ContextConfig) -> np.ndarray:
    """
    A simple, interpretable transition matrix:
    - strong bias to remain in same class (diagonal)
    - uniform mass spread across other classes

    This is a pragmatic default when real session transitions are unknown.
    """

    if n_classes <= 1:
        return np.ones((n_classes, n_classes), dtype=np.float32)
    off = (1.0 - cfg.stay_prob) / float(n_classes - 1)
    T = np.full((n_classes, n_classes), off, dtype=np.float32)
    np.fill_diagonal(T, cfg.stay_prob)
    return np.clip(T, cfg.eps, 1.0).astype(np.float32)


def viterbi_decode(log_emission: np.ndarray, log_trans: np.ndarray, log_prior: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Viterbi decoding for most likely hidden class sequence.

    log_emission: shape (T, K) where K = #classes, T = sequence length
    log_trans: shape (K, K) where log_trans[i,j] = log P(z_t=j | z_{t-1}=i)
    log_prior: shape (K,)
    """

    T, K = log_emission.shape
    if log_prior is None:
        log_prior = np.full((K,), -np.log(K), dtype=np.float32)

    dp = np.empty((T, K), dtype=np.float32)
    back = np.empty((T, K), dtype=np.int32)

    dp[0] = log_prior + log_emission[0]
    back[0] = -1
    for t in range(1, T):
        # For each current state j, choose best previous i.
        scores = dp[t - 1][:, None] + log_trans  # shape (K, K)
        back[t] = np.argmax(scores, axis=0)
        dp[t] = scores[back[t], np.arange(K)] + log_emission[t]

    z = np.empty((T,), dtype=np.int32)
    z[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        z[t] = int(back[t + 1, z[t + 1]])
    return z


def context_aware_predictions(
    proba_seq: np.ndarray, transition: np.ndarray, prior: Optional[np.ndarray] = None, eps: float = 1e-9
) -> np.ndarray:
    """
    Apply Viterbi smoothing to a sequence of per-trace predicted probabilities.

    proba_seq: (T, K) emission probabilities from the baseline model.
    transition: (K, K) transition probabilities between classes.
    """

    proba = np.clip(proba_seq.astype(np.float32), eps, 1.0)
    log_em = np.log(proba)
    log_tr = np.log(np.clip(transition.astype(np.float32), eps, 1.0))
    log_pr = None if prior is None else np.log(np.clip(prior.astype(np.float32), eps, 1.0))
    return viterbi_decode(log_em, log_tr, log_pr)

