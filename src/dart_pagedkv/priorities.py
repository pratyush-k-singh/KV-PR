"""Frozen token-priority extraction from prefill attention.

SnapKV-style: the priority of prompt-token ``j`` is the summed attention from
the last ``obs_window`` query positions to ``j`` in the prefill attention
matrix. Computed once at prefill and frozen for the rest of the run; together
with a monotone top-k filler this makes the budget vector the only adaptive
decision (and the hot sets nest as budget grows).
"""

from __future__ import annotations

import numpy as np


def snapkv_priority(attention_matrix, obs_window: int = 16) -> np.ndarray:
    """Per-token priority: column sums over the last ``obs_window`` rows."""
    arr = np.asarray(attention_matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"expected square attention matrix, got shape {tuple(arr.shape)}"
        )
    if obs_window <= 0:
        raise ValueError(f"obs_window must be >= 1, got {obs_window}")
    seq_len = arr.shape[0]
    start = max(seq_len - int(obs_window), 0)
    window = arr[start:, :]
    return window.sum(axis=0).astype(np.float32, copy=False)
