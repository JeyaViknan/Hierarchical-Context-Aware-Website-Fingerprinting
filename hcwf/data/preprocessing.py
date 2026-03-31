"""
Data preprocessing for the HC-WF pipeline.

Converts raw packet traces into fixed-length tensor representations
suitable for the Packet-Level Transformer (Stage 1).

Each trace is represented as:
  - Direction: +1 (outgoing) / -1 (incoming), derived from signed packet sizes
  - Inter-arrival time (IAT): time delta between consecutive packets

The output tensor has shape (max_trace_len, n_features).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from hcwf.utils.config import PreprocessConfig


def _pad_truncate(arr: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad with pad_value or truncate *arr* to exactly *length* elements."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] >= length:
        return arr[:length]
    out = np.full((length,), pad_value, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def preprocess_trace(
    raw_trace: np.ndarray,
    cfg: PreprocessConfig,
    timestamps: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Convert a single raw trace into a fixed-length feature tensor.

    Parameters
    ----------
    raw_trace : array-like of signed packet sizes (sign = direction).
                Positive = outgoing, Negative = incoming.
    cfg       : PreprocessConfig controlling lengths and clipping.
    timestamps: optional array of packet arrival times (seconds).

    Returns
    -------
    Tensor of shape (max_trace_len, n_features)
    """
    sizes = np.asarray(raw_trace, dtype=np.float32)

    # Direction encoding: +1 / -1
    direction = np.sign(sizes).astype(np.float32)
    direction[direction == 0] = 1.0  # treat zero-size as outgoing

    # Clip and normalize sizes
    sizes_clipped = np.clip(sizes, -cfg.clip_size, cfg.clip_size)
    sizes_norm = sizes_clipped / cfg.clip_size  # normalize to [-1, 1]

    # Use direction-weighted normalised magnitude as primary feature
    feature_dir = direction * np.abs(sizes_norm)

    # Pad / truncate direction feature
    feat_dir = _pad_truncate(feature_dir, cfg.max_trace_len, pad_value=0.0)

    features = [feat_dir]

    if cfg.include_timing:
        if timestamps is not None:
            times = np.asarray(timestamps, dtype=np.float64)
            iat = np.zeros(len(times), dtype=np.float32)
            if len(times) > 1:
                iat[1:] = np.maximum(0.0, np.diff(times)).astype(np.float32)
            # Log-transform IAT to compress heavy tails
            iat_log = np.log1p(iat).astype(np.float32)
        else:
            # Simulate IAT with small uniform noise when timestamps unavailable
            iat_log = np.random.uniform(0.0, 0.1, size=len(sizes)).astype(np.float32)

        feat_iat = _pad_truncate(iat_log, cfg.max_trace_len, pad_value=0.0)
        features.append(feat_iat)

    # Stack features: shape (max_trace_len, n_features)
    tensor = np.stack(features, axis=-1)
    return torch.from_numpy(tensor)


def preprocess_batch(
    raw_traces: List[np.ndarray],
    cfg: PreprocessConfig,
    timestamps_list: Optional[List[np.ndarray]] = None,
) -> torch.Tensor:
    """
    Preprocess a batch of raw traces.

    Returns
    -------
    Tensor of shape (batch_size, max_trace_len, n_features)
    """
    tensors = []
    for i, trace in enumerate(raw_traces):
        ts = timestamps_list[i] if timestamps_list is not None else None
        tensors.append(preprocess_trace(trace, cfg, timestamps=ts))
    return torch.stack(tensors, dim=0)


# ---------------------------------------------------------------------------
# Dummy data generators (for dataset-free operation)
# ---------------------------------------------------------------------------

def generate_dummy_traces(
    n_traces: int,
    n_sites: int,
    cfg: PreprocessConfig,
    min_packets: int = 100,
    max_packets: int = 3000,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic trace data for testing without a real dataset.

    Each trace is a random sequence of +1/-1 directions with random magnitudes,
    mimicking encrypted packet sizes.

    Returns
    -------
    X : Tensor of shape (n_traces, max_trace_len, n_features)
    y : Tensor of shape (n_traces,) with integer site labels
    """
    rng = np.random.RandomState(seed)
    traces = []
    labels = []

    for i in range(n_traces):
        n_packets = rng.randint(min_packets, max_packets + 1)
        site_id = i % n_sites

        # Create class-specific patterns to make learning possible
        base_out_ratio = 0.3 + 0.5 * (site_id / max(n_sites - 1, 1))
        directions = rng.choice(
            [1, -1], size=n_packets, p=[base_out_ratio, 1 - base_out_ratio]
        )
        magnitudes = rng.exponential(500 + site_id * 20, size=n_packets)
        sizes = (directions * magnitudes).astype(np.float32)

        # Synthetic timestamps
        iat = rng.exponential(0.05 + 0.001 * site_id, size=n_packets).astype(np.float64)
        timestamps = np.cumsum(iat)

        traces.append((sizes, timestamps))
        labels.append(site_id)

    X_list = []
    for sizes, ts in traces:
        tensor = preprocess_trace(sizes, cfg, timestamps=ts)
        X_list.append(tensor)

    X = torch.stack(X_list, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y
