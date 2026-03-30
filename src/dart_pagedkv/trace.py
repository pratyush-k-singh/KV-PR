"""Per-layer (Q, K, V) trace wrapper.

Reuses ``src/trace/collect.py``'s hook-based prefill capture for the post-RoPE
Q (the only piece the model doesn't already store), and pulls K and V directly
from ``past_key_values`` -- the same tensors the model's attention used. No
modification to ``collect.py`` is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from src.inference.kv_cache_utils import to_legacy_cache_tuple
from src.trace.collect import LayerQKCapture, run_prefill_with_qk_capture


@dataclass(slots=True)
class LayerQKV:
    """Per-layer Q/K/V for a single batch element.

    Shapes: ``query`` is ``[n_heads, seq, head_dim]``; ``key`` and ``value``
    are ``[n_kv_heads, seq, head_dim]`` (post-RoPE for K).
    """

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    scaling: float
    num_key_value_groups: int


@dataclass(slots=True)
class PromptTrace:
    """A teacher-forced prompt trace ready for replay-based cost/fidelity."""

    layers: Dict[int, LayerQKV]
    input_ids: torch.Tensor
    seq_len: int
    prefill_logits: Optional[torch.Tensor]


def _assemble_qkv_layers(
    qk_layers: Dict[int, LayerQKCapture],
    legacy_cache,
) -> Dict[int, LayerQKV]:
    """Combine hook-captured Q with cache-stored K and V into per-layer records."""
    layers: Dict[int, LayerQKV] = {}
    n_layers = len(legacy_cache)
    for layer_idx, capture in qk_layers.items():
        if layer_idx >= n_layers:
            continue
        cached_k, cached_v = legacy_cache[layer_idx]
        layers[layer_idx] = LayerQKV(
            query=capture.query[0].detach().to("cpu"),
            key=cached_k[0].detach().to("cpu"),
            value=cached_v[0].detach().to("cpu"),
            scaling=float(capture.scaling),
            num_key_value_groups=int(capture.num_key_value_groups),
        )
    return layers


def _assemble_qk_layers(qk_layers: Dict[int, LayerQKCapture]) -> Dict[int, LayerQKV]:
    """Build trace records from captured Q/K only.

    The long-context decode probe uses Q/K for priority and cost, but never
    consumes V. Avoiding ``past_key_values`` during that capture keeps the full
    prompt KV cache off the GPU for the capture prefill.
    """
    layers: Dict[int, LayerQKV] = {}
    for layer_idx, capture in qk_layers.items():
        layers[layer_idx] = LayerQKV(
            query=capture.query[0].detach().to("cpu"),
            key=capture.key[0].detach().to("cpu"),
            value=torch.empty(0),
            scaling=float(capture.scaling),
            num_key_value_groups=int(capture.num_key_value_groups),
        )
    return layers


def capture_prompt_trace(
    model,
    inputs,
    *,
    n_layers_to_track: Optional[int] = None,
    return_logits: bool = True,
    include_values: bool = True,
    query_slice_start: Optional[int] = None,
) -> PromptTrace:
    """Run prefill, return per-layer (Q, K, V) plus input ids and prefill logits."""
    artifacts = run_prefill_with_qk_capture(
        model,
        inputs,
        n_layers_to_track=n_layers_to_track,
        return_past_key_values=include_values,
        return_logits=return_logits,
        query_slice_start=query_slice_start,
    )
    if include_values:
        legacy = to_legacy_cache_tuple(artifacts.past_key_values)
        layers = _assemble_qkv_layers(artifacts.qk_layers, legacy)
    else:
        layers = _assemble_qk_layers(artifacts.qk_layers)
    return PromptTrace(
        layers=layers,
        input_ids=artifacts.input_ids,
        seq_len=artifacts.seq_len,
        prefill_logits=artifacts.logits,
    )
