from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """
    Feature configuration.

    n_prefix: how many (signed) packet sizes to keep as a raw prefix feature vector.
    """

    n_prefix: int = 400


def _safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (mean, std, min, max) with safe defaults for empty arrays."""

    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def extract_features(sizes_fixed: np.ndarray, iat_fixed: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """
    Turn a fixed-length trace representation into a tabular feature vector.

    We combine:
    - global stats of signed sizes
    - directionality stats (incoming/outgoing counts and total bytes)
    - iat stats
    - a raw prefix of the signed size sequence (helps classical ML baselines)
    """

    sizes = np.asarray(sizes_fixed, dtype=np.float32)
    iat = np.asarray(iat_fixed, dtype=np.float32)

    # Ignore padded zeros for many summary stats (but keep prefix raw as-is).
    nonzero = sizes[sizes != 0]

    mean_s, std_s, min_s, max_s = _safe_stats(nonzero)
    abs_sizes = np.abs(nonzero)
    mean_abs, std_abs, min_abs, max_abs = _safe_stats(abs_sizes)

    out_mask = nonzero > 0
    in_mask = nonzero < 0
    out_cnt = int(out_mask.sum())
    in_cnt = int(in_mask.sum())
    out_bytes = float(nonzero[out_mask].sum()) if out_cnt else 0.0
    in_bytes = float(np.abs(nonzero[in_mask]).sum()) if in_cnt else 0.0

    iat_nz = iat[iat > 0]
    mean_i, std_i, min_i, max_i = _safe_stats(iat_nz)

    # Raw prefix: signed sizes, clipped by preprocessing, zero-padded already.
    n = min(cfg.n_prefix, sizes.shape[0])
    prefix = sizes[:n]
    if n < cfg.n_prefix:
        prefix = np.pad(prefix, (0, cfg.n_prefix - n), constant_values=0.0)

    feats = np.concatenate(
        [
            np.array(
                [
                    mean_s,
                    std_s,
                    min_s,
                    max_s,
                    mean_abs,
                    std_abs,
                    min_abs,
                    max_abs,
                    out_cnt,
                    in_cnt,
                    out_bytes,
                    in_bytes,
                    mean_i,
                    std_i,
                    min_i,
                    max_i,
                ],
                dtype=np.float32,
            ),
            prefix.astype(np.float32),
        ]
    )
    return feats


def feature_names(cfg: FeatureConfig) -> List[str]:
    base = [
        "size_mean",
        "size_std",
        "size_min",
        "size_max",
        "abs_size_mean",
        "abs_size_std",
        "abs_size_min",
        "abs_size_max",
        "out_cnt",
        "in_cnt",
        "out_bytes",
        "in_bytes",
        "iat_mean",
        "iat_std",
        "iat_min",
        "iat_max",
    ]
    base += [f"size_prefix_{i}" for i in range(cfg.n_prefix)]
    return base

