"""The formal service cost: cold attention demand.

The Stage-0 formal cost ``c_t`` is the attention probability mass that queries
place on cold-tier (offloaded) tokens. It is computed directly from captured
post-RoPE Q/K -- a measurement against the real cold keys, not a prediction --
which is what makes the reformulated problem's per-step cost observable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def cold_attention_demand(
    query: torch.Tensor,
    key: torch.Tensor,
    hot_key_indices: Sequence[int],
    *,
    scaling: float,
    num_key_value_groups: int = 1,
    query_offset: int | None = None,
) -> float:
    """Mean attention probability mass on cold keys over all (head, query) pairs.

    ``query`` is ``[n_heads, q_len, head_dim]`` and ``key`` is
    ``[n_kv_heads, k_len, head_dim]`` (post-RoPE, as captured by the trace
    layer). Query head ``h`` reads kv-head ``h // num_key_value_groups``. Query
    ``i`` sits at absolute position ``query_offset + i`` and attends causally to
    keys ``[0, query_offset + i]``; ``query_offset`` defaults to ``k_len - q_len``.
    Keys not in ``hot_key_indices`` are cold. Returns a float in ``[0, 1]``.
    """
    query = torch.as_tensor(query, dtype=torch.float32)
    key = torch.as_tensor(key, dtype=torch.float32)
    n_heads, q_len, _ = query.shape
    n_kv_heads, k_len, _ = key.shape
    if q_len == 0 or k_len == 0:
        return 0.0
    if query_offset is None:
        query_offset = k_len - q_len

    # Expand kv-heads up to query-heads for grouped-query attention.
    kv_index = (torch.arange(n_heads) // max(int(num_key_value_groups), 1)).clamp(
        max=n_kv_heads - 1
    )
    expanded_key = key.index_select(0, kv_index)

    scores = torch.matmul(query, expanded_key.transpose(-2, -1)) * float(scaling)

    query_abs = torch.arange(q_len) + int(query_offset)
    key_pos = torch.arange(k_len)
    future = key_pos.unsqueeze(0) > query_abs.unsqueeze(1)  # [q_len, k_len]
    scores = scores.masked_fill(future.unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)

    cold_mask = torch.ones(k_len, dtype=torch.bool)
    hot = [int(h) for h in hot_key_indices if 0 <= int(h) < k_len]
    if hot:
        cold_mask[torch.tensor(hot, dtype=torch.long)] = False

    cold_mass = (probs * cold_mask.to(probs.dtype)).sum(dim=-1)  # [n_heads, q_len]
    return float(cold_mass.mean().item())
