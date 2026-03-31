from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class TraceExample:
    """
    One website visit trace.

    - site: domain (or site id) used as the supervised label for baseline WF
    - transport: "quic" or "non-quic"
    - trace_id: integer id inside the dataset
    - times: packet timestamps (float seconds)
    - sizes: packet sizes (signed int; sign encodes direction)
    """

    site: str
    transport: str
    trace_id: int
    times: np.ndarray  # shape: (n_packets,)
    sizes: np.ndarray  # shape: (n_packets,)


def _as_trace_packets(raw_packets: List[list]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset packet format appears to be: [proto, timestamp, signed_size].
    We extract time and size as dense numpy arrays.
    """

    # Defensive parsing: tolerate tuples/lists and coerce numeric types.
    times = np.fromiter((float(p[1]) for p in raw_packets), dtype=np.float64)
    sizes = np.fromiter((int(p[2]) for p in raw_packets), dtype=np.int32)
    return times, sizes


def load_150sites_npy(path: str) -> List[TraceExample]:
    """
    Load the provided dataset file (e.g., '150sites.npy') into a flat list of traces.

    Expected structure:
      dict[site] -> dict['quic'|'non-quic'] -> dict[trace_id:int] -> list[packet]
      packet := [proto, timestamp, signed_size]
    """

    obj = np.load(path, allow_pickle=True)
    data: Any = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict-like dataset, got {type(data)}")

    out: List[TraceExample] = []
    for site, by_transport in data.items():
        if not isinstance(by_transport, dict):
            continue
        for transport, traces in by_transport.items():
            if not isinstance(traces, dict):
                continue
            for trace_id, raw_packets in traces.items():
                if not raw_packets:
                    continue
                times, sizes = _as_trace_packets(raw_packets)
                out.append(
                    TraceExample(
                        site=str(site),
                        transport=str(transport),
                        trace_id=int(trace_id),
                        times=times,
                        sizes=sizes,
                    )
                )
    if not out:
        raise ValueError("No traces parsed; dataset structure may not match expectations.")
    return out


def build_label_space(examples: Iterable[TraceExample]) -> Tuple[List[str], Dict[str, int]]:
    """Build a stable site->index mapping."""

    sites = sorted({ex.site for ex in examples})
    return sites, {s: i for i, s in enumerate(sites)}

