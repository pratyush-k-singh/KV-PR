"""Attention-output deviation: a per-layer teacher-forced fidelity rung.

For one layer, given captured Q/K/V and a hot key set, compares the layer's
attention output under the *full* (hot+cold) cache to its output under the
*hot-only* cache: ``||full - hot_only||`` averaged over (head, query). The
softmax is computed from the captured Q/K, so this is teacher-forced -- it
measures the per-layer distortion the compression would introduce under the
full-KV hidden-state trajectory. It does *not* capture downstream propagation;
logit-KL via a per-layer-masked forward pass is the stronger rung required
before any Stage-1 claim.

Both ``full`` and ``hot_only`` enforce causal masking. Queries whose every
attendable past key is cold get a zero hot output (rather than NaN), so their
deviation equals ``||full_output||``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def attention_output_deviation(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hot_key_indices: Sequence[int],
    *,
    scaling: float,
    num_key_value_groups: int = 1,
    query_offset: int | None = None,
) -> float:
    """Mean L2 norm of (full_attn_output - hot_only_attn_output) per (head, query).

    Shapes: ``query`` is ``[n_heads, q_len, head_dim]``; ``key`` and ``value``
    are ``[n_kv_heads, k_len, head_dim]`` (post-RoPE for K, as captured). Query
    head ``h`` reads kv-head ``h // num_key_value_groups``. Query ``i`` sits at
    absolute position ``query_offset + i`` and attends causally; ``query_offset``
    defaults to ``k_len - q_len``.
    """
    query = torch.as_tensor(query, dtype=torch.float32)
    key = torch.as_tensor(key, dtype=torch.float32)
    value = torch.as_tensor(value, dtype=torch.float32)

    n_heads, q_len, _ = query.shape
    n_kv_heads, k_len, _ = key.shape
    if q_len == 0 or k_len == 0:
        return 0.0
    if query_offset is None:
        query_offset = k_len - q_len

    kv_index = (torch.arange(n_heads) // max(int(num_key_value_groups), 1)).clamp(
        max=n_kv_heads - 1
    )
    expanded_key = key.index_select(0, kv_index)
    expanded_value = value.index_select(0, kv_index)

    scores = torch.matmul(query, expanded_key.transpose(-2, -1)) * float(scaling)

    query_abs = torch.arange(q_len) + int(query_offset)
    key_pos = torch.arange(k_len)
    future = key_pos.unsqueeze(0) > query_abs.unsqueeze(1)  # [q_len, k_len]

    # Full attention output (causal only).
    scores_full = scores.masked_fill(future.unsqueeze(0), float("-inf"))
    full_probs = torch.softmax(scores_full, dim=-1)
    full_output = torch.matmul(full_probs, expanded_value)

    # Hot-only attention output (causal AND cold-key-masked).
    cold_mask = torch.ones(k_len, dtype=torch.bool)
    hot = [int(h) for h in hot_key_indices if 0 <= int(h) < k_len]
    if hot:
        cold_mask[torch.tensor(hot, dtype=torch.long)] = False
    combined = future | cold_mask.unsqueeze(0)  # [q_len, k_len]

    # Rows where every key is masked would NaN under softmax. Un-mask such
    # rows, compute softmax, then zero the row's output afterwards.
    all_masked = combined.all(dim=-1)
    safe_combined = combined & ~all_masked.unsqueeze(-1)
    scores_hot = scores.masked_fill(safe_combined.unsqueeze(0), float("-inf"))
    hot_probs = torch.softmax(scores_hot, dim=-1)
    keep = (~all_masked).to(hot_probs.dtype).view(1, q_len, 1)
    hot_probs = hot_probs * keep
    hot_output = torch.matmul(hot_probs, expanded_value)

    diff = full_output - hot_output  # [n_heads, q_len, head_dim]
    per_query_l2 = diff.norm(dim=-1)  # [n_heads, q_len]
    return float(per_query_l2.mean().item())
