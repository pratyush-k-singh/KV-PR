"""Windowed SnapKV priority — an O(T) priority for the long-context path.

`trace.collect.recompute_attention_matrix` materialises the full
``[T, T]`` softmaxed attention; at 131k that is the O(T²) OOM Step 6
removed from the forward (a single ``[T,T]`` fp32 tensor at 131k is
~69 GB). But `priorities.snapkv_priority` only sums the last
``obs_window`` query rows — so the priority depends on ``obs_window``
rows of the attention, not all T.

`windowed_recompute_attention` recomputes attention for **only** the
last ``obs_window`` query rows → ``[obs_window, T]``, O(obs_window·T).
It consumes the same post-RoPE, GQA-aligned ``LayerQKCapture`` tensors
as `recompute_attention_matrix` and applies the identical scaling,
``head // num_key_value_groups`` head mapping and per-row causal mask,
so the result equals the full matrix's last ``obs_window`` rows
exactly (softmax is row-wise) — not an approximation. Reusing the
existing capture is deliberate: RoPE and the grouped-query mapping are
inherited, never re-derived.

Spec: docs/superpowers/specs/2026-05-17-longcontext-decode-probe.md §3.
"""

from __future__ import annotations

import numpy as np
import torch


def windowed_recompute_attention(
    qk_layers,
    obs_window: int,
    *,
    device: str = "cpu",
    heads_per_batch: int = 8,
    temperature: float = 1.0,
    window_position: str = "last",
) -> np.ndarray:
    """Mean attention for the last ``obs_window`` query rows only.

    The O(obs_window·T) analogue of
    `trace.collect.recompute_attention_matrix`: identical scaling,
    grouped-query head mapping and per-row causal mask, but the query
    is sliced to its last ``obs_window`` rows before the matmul, so the
    ``[T,T]`` accumulator is never allocated. Returns a ``[w, T]``
    float32 array (``w = min(obs_window, T)``), mean over heads then
    layers — equal (modulo fp order) to
    ``recompute_attention_matrix(qk_layers)[-w:, :]``.

    The ``temperature`` parameter applies a softmax temperature flatten
    to scores BEFORE softmax: ``softmax(scores / temperature)``. Default
    1.0 reproduces the standard behaviour. Larger τ flattens the
    distribution (raises PR); smaller τ sharpens it. This supports the
    P7 mechanism-falsification test (BDPI predicts that flattening
    LLaMA's attention should produce Phi-like priority behaviour). The
    flatten is applied ONLY here, so the model's actual prefill and
    masked decode are unaffected — only the priority computed from this
    block changes.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if window_position not in ("last", "first", "middle"):
        raise ValueError(
            f"window_position must be one of last/first/middle, got "
            f"{window_position!r}"
        )
    if not qk_layers:
        raise ValueError("No Q/K captures available.")
    if obs_window <= 0:
        raise ValueError(f"obs_window must be >= 1, got {obs_window}")

    first = next(iter(qk_layers.values()))
    seq_len = int(first.key.shape[2])
    w = min(int(obs_window), seq_len)

    # P13: pick which w query rows to use for the column-sum.
    # 'last'   (default, SnapKV-standard): rows [T-w, T)
    # 'first'  : rows [0, w)
    # 'middle' : rows [T/2 - w/2, T/2 + w/2)
    if window_position == "last":
        q_start = max(seq_len - w, 0)
    elif window_position == "first":
        q_start = 0
    else:  # middle
        q_start = max((seq_len - w) // 2, 0)
    q_stop = q_start + w

    # Per-row causal mask for these w query rows.
    abs_rows = torch.arange(q_start, q_stop, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = torch.zeros(w, seq_len, dtype=torch.float32, device=device)
    causal_mask.masked_fill_(cols > abs_rows, float("-inf"))

    accumulator = torch.zeros(w, seq_len, dtype=torch.float32)
    for capture in qk_layers.values():
        query = capture.query   # [1, H, T_or_w, d]
        key = capture.key       # [1, H_kv, T, d]
        n_heads = int(query.shape[1])
        # Slice: the producer captures only the last obs_window query rows
        # by default when query_slice_start is set. window_position != 'last'
        # requires the full-length Q to be present in capture.query — the
        # caller is responsible for setting query_slice_start=None on the
        # capture path when using non-last window positions.
        if query.shape[2] == seq_len:
            query_w = query[:, :, q_start:q_stop, :]
        elif window_position == "last":
            query_w = query[:, :, -w:, :]
        else:
            raise ValueError(
                "window_position != 'last' requires full-length Q captured "
                "(set query_slice_start=None in capture_prompt_trace); got "
                f"query shape {tuple(query.shape)}, seq_len {seq_len}."
            )
        layer_acc = torch.zeros(w, seq_len, dtype=torch.float32)
        for start in range(0, n_heads, heads_per_batch):
            stop = min(start + heads_per_batch, n_heads)
            query_batch = query_w[:, start:stop].to(device)
            key_indices = [
                head // capture.num_key_value_groups
                for head in range(start, stop)
            ]
            key_batch = key[:, key_indices].to(device)
            scores = torch.matmul(
                query_batch, key_batch.transpose(-2, -1)
            ) * capture.scaling                          # [1, hb, w, T]
            scores = scores + causal_mask
            if temperature != 1.0:
                scores = scores / float(temperature)
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            layer_acc += weights[0].sum(dim=0).cpu()     # sum heads -> [w, T]
            del query_batch, key_batch, scores, weights
        accumulator += layer_acc / max(n_heads, 1)       # mean over heads
    return (accumulator / max(len(qk_layers), 1)).numpy().astype(
        np.float32, copy=False
    )                                                    # mean over layers


def windowed_attention_per_head(
    capture,
    obs_window: int,
    *,
    device: str = "cpu",
    heads_per_batch: int = 8,
) -> np.ndarray:
    """Per-head windowed attention for ONE layer's capture → ``[H, w, T]``.

    The per-head analogue of `windowed_recompute_attention`, for a single
    layer and **without** the head mean. Identical scaling, ``head //
    num_key_value_groups`` GQA mapping and per-row causal mask — but every
    head's ``[w, T]`` block is kept, because effective-context support is a
    non-linear per-distribution statistic and must be computed per head/row
    *before* any aggregation (`effective_support.block_support`). The
    head-mean of this output equals `windowed_recompute_attention` for the
    same single-layer capture. ``w = min(obs_window, T)``.
    """
    if obs_window <= 0:
        raise ValueError(f"obs_window must be >= 1, got {obs_window}")

    query = capture.query   # [1, H, T, d]
    key = capture.key       # [1, H_kv, T, d]
    seq_len = int(key.shape[2])
    w = min(int(obs_window), seq_len)
    n_heads = int(query.shape[1])

    abs_rows = torch.arange(seq_len - w, seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_mask = torch.zeros(w, seq_len, dtype=torch.float32, device=device)
    causal_mask.masked_fill_(cols > abs_rows, float("-inf"))

    query_w = query[:, :, -w:, :]
    out = np.empty((n_heads, w, seq_len), dtype=np.float32)
    for start in range(0, n_heads, heads_per_batch):
        stop = min(start + heads_per_batch, n_heads)
        query_batch = query_w[:, start:stop].to(device)
        key_indices = [
            head // capture.num_key_value_groups for head in range(start, stop)
        ]
        key_batch = key[:, key_indices].to(device)
        scores = torch.matmul(
            query_batch, key_batch.transpose(-2, -1)
        ) * capture.scaling                              # [1, hb, w, T]
        scores = scores + causal_mask
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out[start:stop] = weights[0].cpu().numpy()
        del query_batch, key_batch, scores, weights
    return out


def windowed_snapkv_priority(
    windowed_attention, obs_window: int | None = None
) -> np.ndarray:
    """Per-token priority from a windowed ``[w, T]`` attention block.

    Column sum over the window rows — identical arithmetic to
    `priorities.snapkv_priority`, fed the pre-windowed block instead of
    a full ``[T,T]`` matrix. If ``obs_window`` is given, only the last
    ``obs_window`` rows are summed (mirroring `snapkv_priority`); the
    default sums every row of the input.
    """
    arr = np.asarray(windowed_attention, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"expected a 2-D [w, T] block, got shape {tuple(arr.shape)}"
        )
    if obs_window is not None:
        if obs_window <= 0:
            raise ValueError(f"obs_window must be >= 1, got {obs_window}")
        arr = arr[max(arr.shape[0] - int(obs_window), 0):, :]
    return arr.sum(axis=0).astype(np.float32, copy=False)


# --- priority-sweep controls (spec §8.3) -----------------------------------
# Alternative within-group priorities used to test whether a finding is an
# artifact of the SnapKV windowed priority or a property of the allocation
# itself. These are attention-free, so they need no Q/K trace.


def recent_priority(seq_len: int) -> np.ndarray:
    """Pure positional priority: higher index → higher priority.

    Top-k selection then keeps the most-recent prompt keys (a
    StreamingLLM-style recency baseline). Group-independent.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    return np.arange(seq_len, dtype=np.float32)


def random_priority(seq_len: int, *, seed: int) -> np.ndarray:
    """Seeded random per-token priority — the priority-sweep floor.

    Deterministic for a given ``seed`` so the sweep is reproducible.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    return np.random.default_rng(int(seed)).random(seq_len).astype(np.float32)


# --- priority-sweep controls (continued) ----------------------------------
# Two more priorities that take Q/K captures, for paper-grade priority-axis
# coverage beyond snapkv / recent / random. ``k_norm_priority`` is K-only
# (no Q needed; the cheapest attention-free Q/K-derived priority).
# ``accumulated_attention_priority`` is the full-prefill H2O analogue —
# windowed_snapkv with window=T, chunked over Q rows to dodge the O(T²)
# memory of the naive [T,T] attention block.


def k_norm_priority(qk_layers) -> np.ndarray:
    """Per-token priority from K vector norms — pure CPU, no attention compute.

    For each prompt token ``i``, computes ``mean_layers mean_heads ||K[i]||_2``.
    No Q is consumed, so this is the cheapest Q/K-derived priority and the only
    one that survives offline (saved K vectors are enough). It is **not** an
    attention proxy: norm reflects the projection-output magnitude, not how
    much the token will be attended to. Useful as a structural baseline:
    "does priority-by-magnitude land in the same allocation regime as the
    attention-based priorities?".
    """
    if not qk_layers:
        raise ValueError("No Q/K captures available.")
    first = next(iter(qk_layers.values()))
    seq_len = int(first.key.shape[2])
    accumulator = np.zeros(seq_len, dtype=np.float32)
    for capture in qk_layers.values():
        key = capture.key  # [1, H_kv, T, d]
        # ||K[i, h, :]||_2 → [1, H_kv, T] → mean over heads → [T]
        norms = torch.linalg.vector_norm(key, dim=-1).squeeze(0)
        accumulator += norms.mean(dim=0).cpu().numpy().astype(np.float32)
    return (accumulator / max(len(qk_layers), 1)).astype(np.float32, copy=False)


def accumulated_attention_priority(
    qk_layers,
    *,
    chunk_size: int = 512,
    device: str = "cpu",
    heads_per_batch: int = 8,
    temperature: float = 1.0,
) -> np.ndarray:
    """H2O-style accumulated-attention priority: full-prefill column sums.

    The full-prefill analogue of `windowed_snapkv_priority`: where the
    windowed variant sums attention from the last ``obs_window`` Q rows,
    this sums across **all** ``T`` Q rows. Chunked over Q to avoid the
    O(T²) memory of the naive ``[T,T]`` attention matrix: per chunk we
    materialise a ``[chunk, T]`` attention block, accumulate column sums
    and discard. Per-row causal masking matches `windowed_recompute_attention`.
    The result is mean-over-heads-then-layers of column-summed attention,
    i.e. an H2O score — the canonical "global accumulated attention"
    KV-eviction signal, repurposed here as a priority for the priority
    sweep (§20). Chunk size ``512`` keeps the per-chunk attention block
    under ``64 MB`` at ``T=32768`` (float32, single head batch).
    """
    if not qk_layers:
        raise ValueError("No Q/K captures available.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if heads_per_batch <= 0:
        raise ValueError(f"heads_per_batch must be >= 1, got {heads_per_batch}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    first = next(iter(qk_layers.values()))
    seq_len = int(first.key.shape[2])
    accumulator = torch.zeros(seq_len, dtype=torch.float32)

    for capture in qk_layers.values():
        query = capture.query    # [1, H, T, d]
        key = capture.key        # [1, H_kv, T, d]
        if int(query.shape[2]) != seq_len:
            raise ValueError(
                "accumulated_attention_priority requires full-length Q; "
                f"got query_len={int(query.shape[2])}, key_len={seq_len}"
            )
        n_heads = int(query.shape[1])

        layer_acc = torch.zeros(seq_len, dtype=torch.float32)
        for q_start in range(0, seq_len, chunk_size):
            q_stop = min(q_start + chunk_size, seq_len)
            chunk_len = q_stop - q_start

            # Per-row causal mask: chunk row r (absolute q_start+r) attends
            # keys 0..q_start+r inclusive.
            abs_rows = torch.arange(
                q_start, q_stop, device=device
            ).unsqueeze(1)
            cols = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = torch.zeros(
                chunk_len, seq_len, dtype=torch.float32, device=device
            )
            mask.masked_fill_(cols > abs_rows, float("-inf"))

            chunk_q = query[:, :, q_start:q_stop, :]   # [1, H, c, d]
            for h_start in range(0, n_heads, heads_per_batch):
                h_stop = min(h_start + heads_per_batch, n_heads)
                q_batch = chunk_q[:, h_start:h_stop].to(device)
                k_indices = [
                    h // capture.num_key_value_groups
                    for h in range(h_start, h_stop)
                ]
                k_batch = key[:, k_indices].to(device)
                scores = torch.matmul(
                    q_batch, k_batch.transpose(-2, -1)
                ) * capture.scaling                     # [1, hb, c, T]
                scores = scores + mask
                if temperature != 1.0:
                    scores = scores / float(temperature)
                weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
                # sum over (heads_in_batch, chunk_rows) → [T]
                layer_acc += weights[0].sum(dim=(0, 1)).cpu()
                del q_batch, k_batch, scores, weights
        accumulator += layer_acc / max(n_heads, 1)

    return (
        accumulator / max(len(qk_layers), 1)
    ).numpy().astype(np.float32, copy=False)
