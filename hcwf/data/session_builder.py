"""
Session builder for the HC-WF pipeline.

Groups individual traces into browsing sessions using time-gap logic.
When timestamps are unavailable, synthetic sessions are constructed
by sampling traces with configurable session lengths.

Sessions are the input unit for Stage 2 (Session-Level Transformer).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hcwf.utils.config import SessionConfig


def build_sessions_from_timestamps(
    traces: List[np.ndarray],
    timestamps: List[float],
    labels: List[int],
    cfg: SessionConfig,
) -> List[Dict]:
    """
    Group traces into sessions based on inter-trace time gaps.

    A new session starts when the time gap between consecutive traces
    exceeds ``cfg.time_gap_threshold``.

    Parameters
    ----------
    traces     : list of preprocessed trace tensors
    timestamps : list of trace-level timestamps (one per trace)
    labels     : list of integer site labels
    cfg        : SessionConfig

    Returns
    -------
    list of session dicts, each containing:
      - "traces": list of trace tensors
      - "labels": list of site labels
      - "intent": synthetic intent label (int)
    """
    if not traces:
        return []

    # Sort by timestamp
    order = np.argsort(timestamps)
    sorted_traces = [traces[i] for i in order]
    sorted_times = [timestamps[i] for i in order]
    sorted_labels = [labels[i] for i in order]

    sessions = []
    current_traces = [sorted_traces[0]]
    current_labels = [sorted_labels[0]]

    for i in range(1, len(sorted_traces)):
        gap = sorted_times[i] - sorted_times[i - 1]
        session_full = len(current_traces) >= cfg.max_session_len
        new_session = gap > cfg.time_gap_threshold or session_full

        if new_session:
            if len(current_traces) >= cfg.min_session_len:
                sessions.append(_make_session_dict(current_traces, current_labels))
            current_traces = [sorted_traces[i]]
            current_labels = [sorted_labels[i]]
        else:
            current_traces.append(sorted_traces[i])
            current_labels.append(sorted_labels[i])

    # Final session
    if len(current_traces) >= cfg.min_session_len:
        sessions.append(_make_session_dict(current_traces, current_labels))

    return sessions


def build_sessions_synthetic(
    traces: List[torch.Tensor],
    labels: List[int],
    cfg: SessionConfig,
    seed: int = 42,
) -> List[Dict]:
    """
    Create synthetic sessions by grouping consecutive traces
    with randomised session lengths.

    Used when real timestamps are unavailable.

    Returns
    -------
    list of session dicts
    """
    rng = np.random.RandomState(seed)
    sessions = []
    idx = 0

    while idx < len(traces):
        session_len = rng.randint(cfg.min_session_len, cfg.max_session_len + 1)
        end = min(idx + session_len, len(traces))
        chunk_traces = traces[idx:end]
        chunk_labels = labels[idx:end]

        if len(chunk_traces) >= cfg.min_session_len:
            sessions.append(_make_session_dict(chunk_traces, chunk_labels))
        idx = end

    return sessions


def _make_session_dict(
    traces: List[torch.Tensor],
    labels: List[int],
) -> Dict:
    """
    Create a session dictionary with an inferred intent label.

    Intent is derived from the diversity and pattern of visited sites
    within the session – this is a synthetic heuristic that will be
    replaced by real labels when a dataset is available.
    """
    unique_sites = set(labels)
    n_unique = len(unique_sites)

    # Synthetic intent categories:
    #   0 = single-site (focused)
    #   1 = two-site (comparison)
    #   2 = multi-site sequential
    #   3 = multi-site with returns (looping)
    #   4 = rapid switching
    #   5 = general browsing
    if n_unique == 1:
        intent = 0
    elif n_unique == 2:
        intent = 1
    elif len(labels) >= 3 and labels[-1] == labels[0] and n_unique > 1:
        intent = 3
    elif n_unique >= 3 and len(set(labels[:3])) == 3:
        intent = 4
    elif n_unique >= 3:
        intent = 2
    else:
        intent = 5

    return {
        "traces": traces,
        "labels": labels,
        "intent": intent,
    }


def collate_sessions(
    sessions: List[Dict],
    max_session_len: int,
    embedding_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of session dicts into padded batch tensors.

    This is used *after* Stage 1 has produced embeddings for each trace.

    Parameters
    ----------
    sessions        : list of session dicts with "embeddings", "labels", "intent"
    max_session_len : pad sessions to this length
    embedding_dim   : dimension of each trace embedding

    Returns
    -------
    embeddings : (B, max_session_len, embedding_dim)
    site_labels: (B, max_session_len) – per-trace site labels
    intent_labels: (B,) – per-session intent labels
    mask       : (B, max_session_len) – True where a real trace exists
    """
    B = len(sessions)
    embeddings = torch.zeros(B, max_session_len, embedding_dim)
    site_labels = torch.zeros(B, max_session_len, dtype=torch.long)
    intent_labels = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B, max_session_len, dtype=torch.bool)

    for i, sess in enumerate(sessions):
        embs = sess["embeddings"]  # list of tensors (embedding_dim,)
        labs = sess["labels"]
        L = min(len(embs), max_session_len)

        for j in range(L):
            embeddings[i, j] = embs[j]
            site_labels[i, j] = labs[j]
            mask[i, j] = True

        intent_labels[i] = sess["intent"]

    return embeddings, site_labels, intent_labels, mask


# ---------------------------------------------------------------------------
# Dummy session generator
# ---------------------------------------------------------------------------

def generate_dummy_sessions(
    n_sessions: int,
    n_sites: int,
    n_intents: int,
    embedding_dim: int,
    max_session_len: int = 5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic session data for Stage 2 testing.

    Returns
    -------
    embeddings    : (n_sessions, max_session_len, embedding_dim)
    site_labels   : (n_sessions, max_session_len)
    intent_labels : (n_sessions,)
    mask          : (n_sessions, max_session_len) boolean
    """
    rng = np.random.RandomState(seed)

    embeddings = torch.zeros(n_sessions, max_session_len, embedding_dim)
    site_labels = torch.zeros(n_sessions, max_session_len, dtype=torch.long)
    intent_labels = torch.zeros(n_sessions, dtype=torch.long)
    mask = torch.zeros(n_sessions, max_session_len, dtype=torch.bool)

    for i in range(n_sessions):
        L = rng.randint(2, max_session_len + 1)
        intent = i % n_intents
        intent_labels[i] = intent

        for j in range(L):
            site = rng.randint(0, n_sites)
            # Create site-dependent embedding patterns for learnability
            base = np.zeros(embedding_dim, dtype=np.float32)
            base[site % embedding_dim] = 1.0
            noise = rng.randn(embedding_dim).astype(np.float32) * 0.3
            embeddings[i, j] = torch.from_numpy(base + noise)
            site_labels[i, j] = site
            mask[i, j] = True

    return embeddings, site_labels, intent_labels, mask
