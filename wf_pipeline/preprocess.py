from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Preprocessing configuration.

    max_len: fixed sequence length after padding/truncation
    clip_size: clip signed packet sizes to [-clip_size, clip_size] for robustness
    """

    max_len: int = 2000
    clip_size: int = 20000


def pad_truncate_1d(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad with pad_value or truncate to max_len."""

    x = np.asarray(x)
    if x.shape[0] >= max_len:
        return x[:max_len]
    out = np.full((max_len,), pad_value, dtype=x.dtype if np.issubdtype(x.dtype, np.number) else np.float32)
    out[: x.shape[0]] = x
    return out


def preprocess_trace(
    times: np.ndarray, sizes: np.ndarray, cfg: PreprocessConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw arrays into standardized sequences:
    - clip sizes
    - derive inter-arrival times (iat)
    - pad/truncate both sequences to cfg.max_len
    """

    times = np.asarray(times, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float32)
    sizes = np.clip(sizes, -cfg.clip_size, cfg.clip_size)

    # Inter-arrival times: first packet has iat=0 by convention.
    iat = np.zeros_like(times, dtype=np.float32)
    if times.shape[0] > 1:
        iat[1:] = np.maximum(0.0, np.diff(times)).astype(np.float32)

    sizes_fixed = pad_truncate_1d(sizes, cfg.max_len, pad_value=0.0).astype(np.float32)
    iat_fixed = pad_truncate_1d(iat, cfg.max_len, pad_value=0.0).astype(np.float32)
    return sizes_fixed, iat_fixed

