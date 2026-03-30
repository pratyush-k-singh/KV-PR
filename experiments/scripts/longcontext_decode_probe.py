#!/usr/bin/env python3
"""Long-context decode-probe — trajectory producer (spec build-step 3).

The GPU half of the Step-7 gate
(`docs/superpowers/specs/2026-05-17-longcontext-decode-probe.md`):
per RULER prompt, prefill once, build the windowed SnapKV priority,
greedy-decode the full-KV reference, then for every
``(ratio, allocation)`` of the budget grid run a masked cached decode
and record the per-step logit-KL trajectory.

Writes to the ``--out`` directory (an ``analyses/2026-05-DD_*`` folder,
matching the repo convention): ``eval_config.json`` (run-config
provenance snapshot) and ``trajectories.json`` (per-prompt per-``(ratio,
allocation)`` records with both the **per-step logit-KL** (fidelity) and
the **formal service cost** (cold-attention-demand against the captured
post-RoPE Q/K). Both signals come from one capture so cost and fidelity
are measured on the same Q/K — the analyzer (`longcontext-decode-probe-
analyze`) then identifies `uniform`, `mincost_bad = argmin cost`,
`oracle_bstar = argmin total_KL`, and `worst_kl = argmax total_KL` per
``(tier, ratio)``. ``ratio_summaries`` flags cells where the budget
floor swallows the ratio (small tiers — ``collapsed=true``) so the
analyzer can skip them honestly instead of silently averaging.

The DART workload (scenario derivation, λ, recovery) is the *analyze*
half — a separate ``--from-trajectories`` step that reuses the existing
`decode_probe` simulation; it is not in this script. Per spec §4 the
pre-compute / analyze split is preserved.

Backend: the model is loaded under **SDPA** (O(T) prefill — native
flex prefill is the O(T²) OOM of §15.1); `FlexCachedDecodeSession`
switches to `flex_attention` only for the hooked masked decode.

Q/K-capture seam (spec §4): the windowed priority needs post-RoPE Q/K.
This runner uses a **separate, explicitly-timed SDPA capture prefill**
(`capture_prompt_trace`) — recorded as ``capture_prefill_s`` — rather
than threading capture hooks into the session prefill. Two SDPA
prefills per prompt; the one-prefill seam is a documented later
refinement (matters mainly for 3B memory headroom). The capture is
SDPA because the model is loaded SDPA — never flex.

Server-side only (HF flex needs Triton + a modern GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from experiments.scripts.stage0_bridge import _layer_qkv_to_capture, _layer_to_group_map
from src.dart_pagedkv.budget_spread import ratio_allocation_grid
from src.dart_pagedkv.effective_support import block_support, group_support_from_layers
from src.dart_pagedkv.flex_decode import FlexCachedDecodeSession
from src.dart_pagedkv.filler import protected_positions
from src.dart_pagedkv.published_allocators import (
    adakv_priority_weights,
    inverse_pyramid_alloc,
    proportional_alloc,
    pyramid_alloc,
)
from src.dart_pagedkv.ruler_manifest import load_manifest_records
from src.dart_pagedkv.service_cost import cold_attention_demand
from src.dart_pagedkv.trace import capture_prompt_trace
from src.dart_pagedkv.two_tier import group_hot_sets
from src.dart_pagedkv.windowed_priority import (
    accumulated_attention_priority,
    k_norm_priority,
    random_priority,
    recent_priority,
    windowed_attention_per_head,
    windowed_recompute_attention,
    windowed_snapkv_priority,
)
from src.trace.collect import (
    load_model_and_tokenizer,
    resolve_model_input_device,
    tokenize_prompt,
)
from src.dart_pagedkv.gqa_emulation import install_gqa_emulation_hooks
from src.dart_pagedkv.layernorm_intervention import install_post_norm_hooks
from src.dart_pagedkv.decoder_jacobian import compute_per_step_sigmas

DEFAULT_RATIOS = "0.005,0.01,0.02,0.04,0.08"


def attention_backends_for_model(model_path: str) -> tuple[str, str]:
    """Return ``(prefill_backend, masked_decode_backend)`` for the model.

    Phi-3.x HF remote code does not implement SDPA. Eager prefill would
    materialise the full ``[T, T]`` attention tensor (the §15.1 O(T²) trap
    — ~134 GB / layer at T=32k). Flash-attention 2 with sliding window
    gives O(T·w) prefill memory. Phi masked decode stays eager because
    Q_LEN=1 makes the backend irrelevant; flash-attention does not
    compose cleanly with the per-step BlockMask path.
    """
    name = str(model_path).lower()
    if "phi-3" in name or "phi3" in name or "phi35" in name:
        return "flash_attention_2", "flash_attention_2"
    return "sdpa", "flex_attention"


def parse_ratios(raw: str) -> list[float]:
    """Parse the comma-separated retention-ratio list."""
    ratios = [float(x) for x in raw.split(",") if x.strip()]
    if not ratios:
        raise ValueError("--ratios must list at least one ratio")
    return ratios


def protected_budget_floors(
    seq_len: int, num_groups: int, n_sink: int, n_recent: int
) -> list[int]:
    """Per-group floors needed to keep sink/recent positions hot.

    ``group_hot_sets`` forces protected positions hot, and those positions
    count against each group's hot budget. Therefore every group needs a floor
    at least as large as the deduplicated protected set; otherwise tiny-ratio
    or single-dominant allocations can make a non-dominant group drop key 0.
    """
    protected_count = len(protected_positions(seq_len, n_sink, n_recent))
    return [protected_count] * num_groups


def windowed_allocation_cost(
    trace, layer_to_group: dict[int, int], hot_per_group: list[list[int]],
    obs_window: int, seq_len: int,
) -> float:
    """Mean over tracked layers of cold-attention-demand for the windowed queries.

    The formal Stage-0 cost (`service_cost.cold_attention_demand`) measured
    against the last ``obs_window`` query rows of each layer's *captured
    post-RoPE Q/K* — the same windowed Q/K the priority is built from, so
    cost and priority share a single capture. Mirrors the per-layer-mean
    aggregation `stage0_bridge` uses for the sub-8k cost.
    """
    query_offset = max(seq_len - obs_window, 0)
    per_layer: list[float] = []
    for layer_idx, layer in trace.layers.items():
        group_idx = layer_to_group.get(layer_idx)
        if group_idx is None:
            continue
        q_win = layer.query[:, -obs_window:, :].contiguous()
        per_layer.append(cold_attention_demand(
            q_win, layer.key, hot_per_group[group_idx],
            scaling=layer.scaling,
            num_key_value_groups=layer.num_key_value_groups,
            query_offset=query_offset,
        ))
    return sum(per_layer) / max(len(per_layer), 1)


def group_windowed_priorities(
    trace, layer_to_group: dict[int, int], num_groups: int,
    obs_window: int, *, device: str, heads_per_batch: int = 8,
    temperature: float = 1.0, window_position: str = "last",
) -> list[list[float]]:
    """Per-layer-group windowed SnapKV priority from a prefill Q/K trace.

    For each group, recompute attention for only the last ``obs_window``
    query rows over that group's layers (O(obs_window·T)), then column-
    sum to the per-token priority. Mirrors the sub-8k per-group priority
    but never materialises the ``[T,T]`` matrix.

    ``temperature`` (default 1.0) flattens the attention softmax before
    column-summing — the P7 mechanism-falsification knob. The cache and
    masked decode are unaffected; only the priority selection changes.
    """
    priorities: list[list[float]] = []
    for group_idx in range(num_groups):
        captures = {
            layer: _layer_qkv_to_capture(trace.layers[layer])
            for layer, grp in layer_to_group.items()
            if grp == group_idx and layer in trace.layers
        }
        if not captures:
            raise ValueError(f"group {group_idx} has no tracked layers")
        windowed = windowed_recompute_attention(
            captures,
            obs_window=obs_window,
            device=device,
            heads_per_batch=heads_per_batch,
            temperature=temperature,
            window_position=window_position,
        )
        priorities.append(windowed_snapkv_priority(windowed).tolist())
    return priorities


def group_per_head_snapkv_priorities(
    trace, layer_to_group: dict[int, int], num_groups: int,
    obs_window: int, *, device: str, heads_per_batch: int = 8,
    temperature: float = 1.0, window_position: str = "last",
) -> list[list[float]]:
    """P14 mechanism probe: per-head SnapKV priority.

    For each layer-group, compute the windowed attention PER HEAD (using
    ``windowed_attention_per_head``), produce a per-head column-sum
    priority, then aggregate across heads by MAX (so a token wins a slot
    if any single head considers it important). This eliminates the
    cross-head averaging step in standard SnapKV. If per-head heads have
    cleanly peaked attention but aggregation washes the peaks out, this
    priority should pick the right tokens on Phi where the aggregated
    priority fails.
    """
    from src.dart_pagedkv.windowed_priority import windowed_attention_per_head
    import numpy as np
    priorities: list[list[float]] = []
    for group_idx in range(num_groups):
        layers = [
            layer for layer, grp in layer_to_group.items()
            if grp == group_idx and layer in trace.layers
        ]
        if not layers:
            raise ValueError(f"group {group_idx} has no tracked layers")
        # Per-head, per-layer: [H, w, T]. Column-sum over w → [H, T].
        # Apply temperature inside windowed_attention_per_head if needed.
        # For temperature support we'd need to extend that function too;
        # for now this path uses raw softmax (temperature ignored — P14
        # is a separate ablation from P7's temperature knob).
        per_token_max = None
        for layer_idx in layers:
            capture = _layer_qkv_to_capture(trace.layers[layer_idx])
            block = windowed_attention_per_head(
                capture, obs_window,
                device=device, heads_per_batch=heads_per_batch,
            )  # [H, w, T]
            head_priorities = block.sum(axis=1)  # [H, T] per-head column-sum
            # Aggregate across heads by max — winning tokens are those any
            # head strongly attended to.
            head_max = head_priorities.max(axis=0)  # [T]
            if per_token_max is None:
                per_token_max = head_max
            else:
                # Mean across layers within group, then max across heads
                # (already done above); accumulate the per-token-max via
                # layer-mean.
                per_token_max = per_token_max + head_max
        per_token_max = per_token_max / max(len(layers), 1)
        priorities.append(per_token_max.tolist())
    return priorities


def group_layer_weighted_priorities(
    trace, layer_to_group: dict[int, int], num_groups: int,
    obs_window: int, *, device: str, heads_per_batch: int = 8,
    temperature: float = 1.0, window_position: str = "last",
) -> list[list[float]]:
    """P16 mechanism probe: PR-inverse-weighted per-layer priority.

    For each layer in the group, compute its windowed attention's
    participation ratio PR_layer (1 / Σ p_i² over the head-mean of the
    last window-position row). Lower-PR (more peaked) layers get higher
    weight in the column-sum aggregation; higher-PR (more diffuse)
    layers contribute less. If high-PR layers cause the bias direction
    on Phi (per LRDB Claim 1 architectural origin), this priority should
    have lower α than the unweighted version.
    """
    from src.dart_pagedkv.windowed_priority import windowed_attention_per_head
    import numpy as np
    priorities: list[list[float]] = []
    for group_idx in range(num_groups):
        layers = [
            layer for layer, grp in layer_to_group.items()
            if grp == group_idx and layer in trace.layers
        ]
        if not layers:
            raise ValueError(f"group {group_idx} has no tracked layers")
        # Per-layer windowed attention, then per-layer PR weight.
        accumulator = None
        total_weight = 0.0
        for layer_idx in layers:
            captures = {layer_idx: _layer_qkv_to_capture(trace.layers[layer_idx])}
            block = windowed_recompute_attention(
                captures, obs_window=obs_window,
                device=device, heads_per_batch=heads_per_batch,
                temperature=temperature, window_position=window_position,
            )  # [w, T]
            # PR of the last row's distribution (proxy for layer-level PR).
            last_row = block[-1, :]
            last_row = last_row / max(float(last_row.sum()), 1e-12)
            inv_simpson = 1.0 / max(float((last_row ** 2).sum()), 1e-12)
            weight = 1.0 / max(inv_simpson, 1.0)  # higher PR → lower weight
            col_sum = block.sum(axis=0)
            if accumulator is None:
                accumulator = weight * col_sum
            else:
                accumulator = accumulator + weight * col_sum
            total_weight += weight
        accumulator = accumulator / max(total_weight, 1e-12)
        priorities.append(accumulator.tolist())
    return priorities


PRIORITY_MODES = (
    "snapkv", "recent", "random", "shared_random", "k_norm", "accumulated",
    "tova",
)

EXTRA_ALLOCATORS = ("pyramid", "inverse_pyramid", "adakv")


def build_extra_allocations(
    num_groups: int, total: int, floors: list[int],
    group_priorities: list[list[float]],
    names: list[str],
) -> list[tuple[str, list[int]]]:
    """Compute named published-allocator vectors for one ratio.

    Returns ``(name, allocation)`` pairs in the same order as ``names``.
    ``adakv`` uses per-group priority-mass weights (status §17): the AdaKV
    group-level analog of head-adaptive budget. ``pyramid`` /
    ``inverse_pyramid`` are data-free and use the published default
    ``alpha=0.2`` shape (1.5:1 max:min discretionary share).
    """
    out: list[tuple[str, list[int]]] = []
    for name in names:
        if name == "pyramid":
            out.append((name, pyramid_alloc(num_groups, total, floors)))
        elif name == "inverse_pyramid":
            out.append(
                (name, inverse_pyramid_alloc(num_groups, total, floors))
            )
        elif name == "adakv":
            weights = adakv_priority_weights(
                group_priorities, total, floors
            )
            out.append(
                (name, proportional_alloc(num_groups, total, floors, weights))
            )
        else:
            raise ValueError(
                f"unknown extra allocator {name!r}; "
                f"expected one of {EXTRA_ALLOCATORS}"
            )
    return out


def _per_group_captures(
    trace, layer_to_group: dict[int, int], num_groups: int,
) -> list[dict[int, Any]]:
    """Per-group ``{layer_idx: LayerQKCapture}`` view of the prefill trace.

    Mirrors the per-group capture slice :func:`group_windowed_priorities`
    builds inline; exposed as a helper so other group-level priorities
    (k_norm, accumulated) can use the same slicing without duplicating it.
    """
    per_group: list[dict[int, Any]] = []
    for group_idx in range(num_groups):
        captures = {
            layer: _layer_qkv_to_capture(trace.layers[layer])
            for layer, grp in layer_to_group.items()
            if grp == group_idx and layer in trace.layers
        }
        if not captures:
            raise ValueError(f"group {group_idx} has no tracked layers")
        per_group.append(captures)
    return per_group


def select_group_priorities(
    trace, layer_to_group: dict[int, int], num_groups: int, obs_window: int,
    seq_len: int, *, priority: str, seed: int, device: str,
    accumulated_chunk_size: int = 512,
    priority_heads_per_batch: int = 8,
    priority_temperature: float = 1.0,
    window_position: str = "last",
    per_head_snapkv: bool = False,
    per_layer_weighted_priority: bool = False,
) -> list[list[float]]:
    """Per-layer-group within-group priority for the chosen ``priority`` mode.

    ``snapkv`` (default) is the windowed SnapKV priority. ``recent`` and
    ``random`` are attention-free controls (status §20). ``k_norm`` is the
    cheap K-only structural baseline (``||K[i]||_2`` mean over layer-group's
    layers/heads); ``accumulated`` is the H2O-style full-prefill column-sum
    priority, chunked over Q rows. All five modes return one priority vector
    per group; ``snapkv`` / ``k_norm`` / ``accumulated`` use this group's
    layers, ``recent`` / ``random`` are group-independent.
    """
    if priority == "snapkv":
        if per_head_snapkv:
            return group_per_head_snapkv_priorities(
                trace, layer_to_group, num_groups, obs_window,
                device=device, heads_per_batch=priority_heads_per_batch,
                temperature=priority_temperature,
                window_position=window_position,
            )
        if per_layer_weighted_priority:
            return group_layer_weighted_priorities(
                trace, layer_to_group, num_groups, obs_window,
                device=device, heads_per_batch=priority_heads_per_batch,
                temperature=priority_temperature,
                window_position=window_position,
            )
        return group_windowed_priorities(
            trace,
            layer_to_group,
            num_groups,
            obs_window,
            device=device,
            heads_per_batch=priority_heads_per_batch,
            temperature=priority_temperature,
            window_position=window_position,
        )
    if priority == "tova":
        return group_windowed_priorities(
            trace, layer_to_group, num_groups, 1,
            device=device, heads_per_batch=priority_heads_per_batch,
            temperature=priority_temperature, window_position=window_position,
        )
    if priority == "recent":
        recent = recent_priority(seq_len).tolist()
        return [list(recent) for _ in range(num_groups)]
    if priority == "random":
        return [
            random_priority(seq_len, seed=seed + g).tolist()
            for g in range(num_groups)
        ]
    if priority == "shared_random":
        shared = random_priority(seq_len, seed=seed).tolist()
        return [list(shared) for _ in range(num_groups)]
    if priority == "k_norm":
        priorities: list[list[float]] = []
        for captures in _per_group_captures(trace, layer_to_group, num_groups):
            priorities.append(k_norm_priority(captures).tolist())
        return priorities
    if priority == "accumulated":
        priorities = []
        for captures in _per_group_captures(trace, layer_to_group, num_groups):
            priorities.append(
                accumulated_attention_priority(
                    captures, chunk_size=accumulated_chunk_size,
                    device=device, temperature=priority_temperature,
                ).tolist()
            )
        return priorities
    raise ValueError(
        f"unknown priority {priority!r}; expected one of {PRIORITY_MODES}"
    )


def retained_coverage_metrics(
    hot_per_group: list[list[int] | set[int]],
    reference_priorities: list[list[float]],
    seq_len: int,
    *,
    record_positions: bool = False,
) -> dict[str, Any]:
    """Retained-set coverage under a fixed useful-mass reference.

    The reference priority is normally the windowed SnapKV column-sum for
    each group, even when the selected keys came from random/shared_random.
    That keeps the measurement tied to retained useful attention mass rather
    than to the selector's own score.

    When ``record_positions`` is True (P10 mechanism probe), the output
    block also includes a 10-bin positional histogram per group and the
    mean retained position per group, normalized to [0, 1] over the
    prompt's context length. Optionally the full per-group token-id sets
    can be reconstructed from these histograms — for the cheapest probe
    we keep just the histograms and statistics, not the full sets.
    """
    sets = [set(int(i) for i in hot) for hot in hot_per_group]
    if not sets:
        block = {
            "union_size": 0,
            "intersection_size": 0,
            "mean_pairwise_jaccard": None,
            "per_group_hot_counts": [],
            "per_group_retained_reference_mass": [],
            "mean_retained_reference_mass": None,
            "min_retained_reference_mass": None,
            "max_retained_reference_mass": None,
            "total_retained_reference_mass": None,
            "union_fraction_of_context": 0.0,
            "union_fraction_of_group_budget_sum": None,
        }
        if record_positions:
            block["per_group_position_hist_10bin"] = []
            block["per_group_mean_retained_position"] = []
            block["per_group_p10_retained_position"] = []
            block["per_group_p50_retained_position"] = []
            block["per_group_p90_retained_position"] = []
        return block

    union = set().union(*sets)
    intersection = set(sets[0])
    for s in sets[1:]:
        intersection &= s

    jaccards: list[float] = []
    for i, left in enumerate(sets):
        for right in sets[i + 1:]:
            denom = len(left | right)
            jaccards.append(1.0 if denom == 0 else len(left & right) / denom)

    retained_masses: list[float] = []
    for hot, priority_vec in zip(sets, reference_priorities):
        denom = float(sum(max(0.0, float(v)) for v in priority_vec))
        if denom <= 0:
            retained_masses.append(0.0)
            continue
        retained = sum(
            max(0.0, float(priority_vec[idx]))
            for idx in hot
            if 0 <= idx < len(priority_vec)
        )
        retained_masses.append(float(retained / denom))

    budget_sum = sum(len(s) for s in sets)
    block = {
        "union_size": len(union),
        "intersection_size": len(intersection),
        "mean_pairwise_jaccard": (
            float(sum(jaccards) / len(jaccards)) if jaccards else None
        ),
        "per_group_hot_counts": [len(s) for s in sets],
        "per_group_retained_reference_mass": retained_masses,
        "mean_retained_reference_mass": (
            float(sum(retained_masses) / len(retained_masses))
            if retained_masses else None
        ),
        "min_retained_reference_mass": (
            float(min(retained_masses)) if retained_masses else None
        ),
        "max_retained_reference_mass": (
            float(max(retained_masses)) if retained_masses else None
        ),
        "total_retained_reference_mass": (
            float(sum(retained_masses)) if retained_masses else None
        ),
        "union_fraction_of_context": (
            float(len(union) / seq_len) if seq_len > 0 else 0.0
        ),
        "union_fraction_of_group_budget_sum": (
            float(len(union) / budget_sum) if budget_sum > 0 else None
        ),
    }
    if record_positions and seq_len > 0:
        per_group_hist: list[list[int]] = []
        per_group_mean_pos: list[float] = []
        per_group_p10: list[float] = []
        per_group_p50: list[float] = []
        per_group_p90: list[float] = []
        for s in sets:
            hist = [0] * 10
            positions: list[float] = []
            for idx in s:
                if 0 <= idx < seq_len:
                    rel = idx / max(seq_len - 1, 1)
                    positions.append(float(rel))
                    bin_idx = min(int(rel * 10), 9)
                    hist[bin_idx] += 1
            per_group_hist.append(hist)
            if positions:
                positions.sort()
                n = len(positions)
                per_group_mean_pos.append(float(sum(positions) / n))
                per_group_p10.append(float(positions[max(int(0.10 * n), 0)]))
                per_group_p50.append(float(positions[n // 2]))
                per_group_p90.append(float(positions[min(int(0.90 * n), n - 1)]))
            else:
                per_group_mean_pos.append(float("nan"))
                per_group_p10.append(float("nan"))
                per_group_p50.append(float("nan"))
                per_group_p90.append(float("nan"))
        block["per_group_position_hist_10bin"] = per_group_hist
        block["per_group_mean_retained_position"] = per_group_mean_pos
        block["per_group_p10_retained_position"] = per_group_p10
        block["per_group_p50_retained_position"] = per_group_p50
        block["per_group_p90_retained_position"] = per_group_p90
    return block


# Fixed grid of (estimator, tau) computed alongside the primary in one
# windowed-attention pass per layer. Lets the §9 saturation predictor be
# calibrated post-hoc (pick the best (estimator, tau)) without re-running
# the GPU producer. The windowed attention is the heavy step; per-entry
# block_support reductions are cheap, so the grid is essentially free.
SUPPORT_GRID: list[tuple[str, float | None]] = [
    ("mass_coverage", 0.5),
    ("mass_coverage", 0.7),
    ("mass_coverage", 0.9),
    ("mass_coverage", 0.95),
    ("participation_ratio", None),
    ("entropy", None),
]


def _entry_from_group_support(gs, *, estimator: str, tau: float | None) -> dict[str, Any]:
    return {
        "estimator": estimator,
        "tau": tau,
        "per_group": gs.per_group_mean,
        "per_group_max": gs.per_group_max,
    }


def prompt_effective_support(
    trace, layer_to_group: dict[int, int], num_groups: int, obs_window: int,
    *, primary_estimator: str, primary_tau: float, device,
) -> dict[str, Any]:
    """Per-group effective-context support from the prefill Q/K trace.

    The saturation predictor (spec §6). **Non-circular by construction:**
    support is measured on full-KV (uncompressed) windowed attention — the
    same post-RoPE Q/K capture the priority is built from — never on the
    masked/compressed attention nor on the SnapKV priority. Streamed per
    layer: a 131k run never holds more than one layer's ``[H, w, T]`` block
    at a time.

    Returns ``{"primary": {...}, "grid": [{...}, ...]}``. ``primary``
    matches the CLI ``--support-estimator/--support-tau`` (kept flat in
    ``trajectories.json`` as ``effective_support`` for backwards-compat).
    ``grid`` is :data:`SUPPORT_GRID` computed off the same per-layer
    windowed attention so the §9 predictor can be re-picked at analysis
    time without another GPU run.
    """
    groups: list[list[int]] = [[] for _ in range(num_groups)]
    for layer_idx, group_idx in sorted(layer_to_group.items()):
        groups[group_idx].append(layer_idx)

    # Heavy step: one windowed-attention pass per layer.
    layer_blocks: dict[int, Any] = {}
    for layer_idx in layer_to_group:
        if layer_idx not in trace.layers:
            continue
        capture = _layer_qkv_to_capture(trace.layers[layer_idx])
        layer_blocks[layer_idx] = windowed_attention_per_head(
            capture, obs_window=obs_window, device=str(device)
        )

    # Cheap reduction: per (estimator, tau).
    def _reduce(estimator: str, tau: float):
        layer_mean: dict[int, float] = {}
        layer_max: dict[int, float] = {}
        for layer_idx, block in layer_blocks.items():
            layer_mean[layer_idx], layer_max[layer_idx] = block_support(
                block, estimator=estimator, tau=tau
            )
        return group_support_from_layers(
            layer_mean, layer_max, groups, estimator=estimator, tau=tau
        )

    primary_gs = _reduce(primary_estimator, primary_tau)
    primary = _entry_from_group_support(
        primary_gs, estimator=primary_estimator, tau=primary_tau
    )
    primary["source"] = "obs_window"

    grid: list[dict[str, Any]] = []
    for estimator, tau in SUPPORT_GRID:
        gs = _reduce(estimator, tau if tau is not None else 0.95)
        grid.append(_entry_from_group_support(gs, estimator=estimator, tau=tau))

    return {"primary": primary, "grid": grid}


def _run_prompt(
    record, *, model, tokenizer, device, num_layers: int, num_groups: int,
    obs_window: int, k_dec: int, n_sink: int, n_recent: int,
    ratios: list[float], support_estimator: str, support_tau: float,
    priority: str, priority_seed: int, skip_support: bool = False,
    accumulated_chunk_size: int = 512,
    priority_heads_per_batch: int = 8,
    extra_allocators: tuple[str, ...] = (),
    record_coverage_metrics: bool = False,
    masked_attn_implementation: str = "flex_attention",
    priority_temperature: float = 1.0,
    floor_only: bool = False,
    record_retained_positions: bool = False,
    obs_window_position: str = "last",
    decode_temperature: float = 1.0,
    per_head_snapkv: bool = False,
    per_layer_weighted_priority: bool = False,
    record_decoder_queries: bool = False,
    record_decoder_jacobian: bool = False,
    jacobian_measure_steps: tuple[int, ...] = (),
    jacobian_n_iter: int = 10,
    jacobian_eps: float = 1e-3,
    jacobian_priority_seed: int = 0,
) -> dict[str, Any]:
    """One prompt: capture → windowed priority → reference → 45 trajectories."""
    out: dict[str, Any] = {
        "prompt_id": record.prompt_id,
        "task": record.task,
        "tier_length": record.tier_length,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inputs = tokenize_prompt(
        tokenizer, record.prompt, device=str(device),
        max_length=None, chat_template="never",
    )
    prompt_ids = inputs["input_ids"]
    seq_len = int(prompt_ids.shape[1])
    out["tokens"] = seq_len

    layer_to_group, _ = _layer_to_group_map(num_layers, num_groups, num_layers)

    # SDPA capture prefill (#1) — post-RoPE Q/K for the windowed priority.
    # All long-context priorities/costs except `accumulated` only consume the
    # final observation-window query rows. Capturing only those Q rows avoids a
    # full-length Q projection at 131k (Phi's full Q is ~768 MiB per layer).
    query_slice_start = None
    if priority != "accumulated":
        query_slice_start = max(seq_len - obs_window, 0)
    # P13 mechanism probe: non-last obs_window positions need the FULL Q
    # capture (not just the last w rows). Same for P14 per-head SnapKV and
    # P16 per-layer weighted priority — they may slice mid-prefill rows.
    if obs_window_position != "last" or per_head_snapkv or per_layer_weighted_priority:
        query_slice_start = None
    t0 = time.perf_counter()
    trace = capture_prompt_trace(
        model,
        inputs,
        n_layers_to_track=num_layers,
        return_logits=False,
        include_values=False,
        query_slice_start=query_slice_start,
    )
    out["capture_prefill_s"] = time.perf_counter() - t0

    group_priorities = select_group_priorities(
        trace, layer_to_group, num_groups, obs_window, seq_len,
        priority=priority, seed=priority_seed, device=str(device),
        accumulated_chunk_size=accumulated_chunk_size,
        priority_heads_per_batch=priority_heads_per_batch,
        priority_temperature=priority_temperature,
        window_position=obs_window_position,
        per_head_snapkv=per_head_snapkv,
        per_layer_weighted_priority=per_layer_weighted_priority,
    )
    # P9 floor-only ablation. If floor_only is requested, the producer scales
    # n_recent dynamically per ratio so that floor = budget exactly, i.e. the
    # discretionary budget is zero on every allocation. This is implemented
    # inside the ratio loop further down (search for "FLOOR_ONLY"). At that
    # point the priority is irrelevant — all four groups retain exactly the
    # n_sink first positions plus the (budget - n_sink) last positions.

    # Budget grid + every plan's hot_per_group precomputed once (pure CPU
    # from the priorities) so cost and KL loops share the same hot sets.
    floors = protected_budget_floors(seq_len, num_groups, n_sink, n_recent)
    floor_sum = int(sum(floors))
    grid = ratio_allocation_grid(seq_len, ratios, num_groups, floors)
    extra_allocator_tags: dict[tuple[float, tuple[int, ...]], str] = {}
    plans: list[dict[str, Any]] = []
    for entry in grid:
        # FLOOR_ONLY: scale n_recent per-ratio so floor = budget per group
        # and the discretionary budget is zero. The "alloc-effective"
        # n_recent is the per-group budget minus n_sink; hot_per_group will
        # then return exactly the protected positions (sinks + recents) and
        # the priority is irrelevant for selection.
        if floor_only:
            per_group_budget = max(int(entry["total"]) // num_groups, n_sink)
            effective_n_recent = max(per_group_budget - n_sink, 0)
            effective_n_sink = n_sink
        else:
            effective_n_recent = n_recent
            effective_n_sink = n_sink
        seen = {tuple(a) for a in entry["allocations"]}
        for allocation in entry["allocations"]:
            plans.append({
                "ratio": entry["ratio"],
                "total": entry["total"],
                "allocation": list(allocation),
                "hot_per_group": group_hot_sets(
                    allocation, group_priorities, seq_len,
                    effective_n_sink, effective_n_recent,
                ),
            })
        # Published-allocator extras (status §17, opt-in via --extra-allocators).
        for name, allocation in build_extra_allocations(
            num_groups, int(entry["total"]), list(floors),
            group_priorities, list(extra_allocators),
        ):
            tup = tuple(int(x) for x in allocation)
            extra_allocator_tags[(float(entry["ratio"]), tup)] = name
            if tup in seen:
                continue   # the spread already covered this exact vector
            seen.add(tup)
            plans.append({
                "ratio": entry["ratio"],
                "total": entry["total"],
                "allocation": list(allocation),
                "hot_per_group": group_hot_sets(
                    allocation, group_priorities, seq_len,
                    effective_n_sink, effective_n_recent,
                ),
            })

    # Optional mechanism telemetry: judge every retained set against the same
    # attention-derived reference mass. For non-SnapKV selectors this extra
    # pass is the causal-intermediate measurement: did the selected union cover
    # more useful attention mass, or merely get lucky in KL?
    if record_coverage_metrics:
        t0 = time.perf_counter()
        if priority == "snapkv":
            reference_priorities = group_priorities
        else:
            reference_priorities = group_windowed_priorities(
                trace,
                layer_to_group,
                num_groups,
                obs_window,
                device=device,
                heads_per_batch=priority_heads_per_batch,
            )
        for plan in plans:
            plan["coverage"] = retained_coverage_metrics(
                plan["hot_per_group"], reference_priorities, seq_len,
                record_positions=record_retained_positions,
            )
        out["coverage_compute_s"] = time.perf_counter() - t0
    else:
        out["coverage_compute_s"] = 0.0

    # Per-allocation formal service cost — computed from the trace BEFORE
    # del-ing it. cold_attention_demand against the same windowed Q/K the
    # priority was built from. This is what the analyzer uses to identify
    # mincost_bad honestly (argmin cost), not by assuming dominate-g0.
    t0 = time.perf_counter()
    for plan in plans:
        plan["cost"] = windowed_allocation_cost(
            trace, layer_to_group, plan["hot_per_group"], obs_window, seq_len
        )
    out["cost_compute_s"] = time.perf_counter() - t0

    # Effective-context support — the saturation predictor (spec §6).
    # Measured on the same full-KV capture, BEFORE del-ing the trace.
    # We compute the primary (CLI-flag) estimate AND a fixed grid of
    # (estimator, tau) in one pass (the windowed attention is the heavy
    # step; per-entry reductions are cheap). The grid lets the analyzer
    # pick a different predictor post-hoc without re-running the GPU
    # producer.
    #
    # ``skip_support`` skips this entirely: per-head windowed attention
    # is by far the dominant cost (~150 s/prompt at 32k vs ~10 s for the
    # masked decode). For experiments where the per-prompt support is
    # not needed (e.g. within-task replication of the tail, where
    # oracle_gap / worst_gap / mincost−uniform answer the question on
    # their own and the per-prompt support→saturation predictor has
    # already failed), skipping reclaims that time.
    if not skip_support:
        t0 = time.perf_counter()
        support = prompt_effective_support(
            trace, layer_to_group, num_groups, obs_window,
            primary_estimator=support_estimator, primary_tau=support_tau,
            device=device,
        )
        out["effective_support"] = support["primary"]
        out["effective_support_grid"] = support["grid"]
        out["support_compute_s"] = time.perf_counter() - t0
    else:
        out["support_compute_s"] = 0.0

    del trace
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SDPA session prefill (#2) — the cache the masked decode runs against.
    t0 = time.perf_counter()
    session = FlexCachedDecodeSession(
        model, prompt_ids, layer_to_group, device=device,
        masked_attn_implementation=masked_attn_implementation,
    )
    out["prefill_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gold, ref_logits = session.full_kv_reference(k_dec)
    out["ref_decode_s"] = time.perf_counter() - t0
    out["gold_continuation"] = gold

    trajectories: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for plan in plans:
        if record_decoder_queries:
            kls, query_norms = session.masked_decode(
                plan["hot_per_group"], gold, ref_logits,
                decode_temperature=decode_temperature,
                return_query_norms=True,
            )
        else:
            kls = session.masked_decode(
                plan["hot_per_group"], gold, ref_logits,
                decode_temperature=decode_temperature,
            )
            query_norms = None
        entry = {
            "ratio": plan["ratio"],
            "total": plan["total"],
            "allocation": plan["allocation"],
            "cost": plan["cost"],
            "per_step_kl": kls,
        }
        if query_norms is not None:
            entry["per_step_logit_diff_l2"] = query_norms
        if "coverage" in plan:
            entry["coverage"] = plan["coverage"]
        # P20: direct rho_R top-sigma measurement at uniform allocations only.
        # Uniform = max-min<=1 across groups (the canonical case the proxy is
        # reported against). Non-uniform plans skip this to keep wall-time
        # manageable.
        if record_decoder_jacobian and jacobian_measure_steps:
            allocation = plan["allocation"]
            is_uniform = (max(allocation) - min(allocation)) <= 1
            if is_uniform:
                rng = torch.Generator(device=device)
                rng.manual_seed(int(jacobian_priority_seed) + int(1000 * plan["ratio"]))
                t_jac0 = time.perf_counter()
                jac_results = compute_per_step_sigmas(
                    session, plan["hot_per_group"], gold,
                    measure_steps=jacobian_measure_steps,
                    n_iter=jacobian_n_iter, eps=jacobian_eps, rng=rng,
                )
                entry["per_step_top_sigma"] = {
                    int(s): {
                        "top_sigma": r["top_sigma"],
                        "baseline_logit_norm": r["baseline_logit_norm"],
                        "jv_norm_history": r["jv_norm_history"],
                    }
                    for s, r in jac_results.items()
                }
                entry["jacobian_compute_s"] = time.perf_counter() - t_jac0
        tag = extra_allocator_tags.get(
            (float(plan["ratio"]), tuple(int(x) for x in plan["allocation"]))
        )
        if tag is not None:
            entry["allocator_tag"] = tag
        trajectories.append(entry)
    out["masked_decode_s"] = time.perf_counter() - t0
    out["trajectories"] = trajectories
    out["n_trajectories"] = len(trajectories)

    # Per-ratio honesty: small tiers can have ratio*T <= floor_sum -> the
    # 9-grid collapses to the uniform floor. Flag it so the analyzer can
    # skip collapsed cells instead of silently averaging zero-variance rows.
    out["floor_sum"] = floor_sum
    out["ratio_summaries"] = [
        {
            "ratio": entry["ratio"],
            "total": entry["total"],
            "discretionary": entry["total"] - floor_sum,
            "n_distinct_allocations": len({tuple(a) for a in entry["allocations"]}),
            "collapsed": entry["total"] <= floor_sum,
        }
        for entry in grid
    ]
    if torch.cuda.is_available():
        out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    out["status"] = "ok"
    return out


def _trajectories_payload(
    eval_config: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """The ``trajectories.json`` document: run config + per-prompt records.

    ``prompts_ok`` counts records whose ``status`` is ``"ok"`` — a failed
    prompt is still recorded, with its exception type as ``status``.
    """
    ok = sum(1 for r in results if r.get("status") == "ok")
    return {
        **eval_config,
        "prompts_attempted": len(results),
        "prompts_ok": ok,
        "prompts": results,
    }


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Serialise ``obj`` to ``path`` as indented JSON, atomically.

    Writes a sibling ``.tmp`` file then ``os.replace``s it into place, so
    a process killed mid-write leaves the previous complete file intact
    rather than a truncated one. This lets the producer rewrite
    ``trajectories.json`` after every prompt — a cancelled or
    wall-time-killed run keeps every prompt finished so far.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="longcontext-decode-probe")
    p.add_argument("--manifest", type=Path, required=True,
                   help="Tiered RULER manifest from ruler-length-sweep-manifest.")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k-dec", type=int, default=16)
    p.add_argument("--obs-window", type=int, default=32,
                   help="SnapKV observation window (windowed-priority rows).")
    p.add_argument("--num-groups", type=int, default=4)
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--n-recent", type=int, default=64)
    p.add_argument("--ratios", default=DEFAULT_RATIOS,
                   help="Comma-separated retention ratios of the prompt length.")
    p.add_argument("--support-estimator", default="mass_coverage",
                   choices=["mass_coverage", "participation_ratio", "entropy"],
                   help="Effective-context support estimator (spec §6).")
    p.add_argument("--support-tau", type=float, default=0.95,
                   help="Mass-coverage threshold for the support estimator.")
    p.add_argument("--priority", default="snapkv",
                   choices=list(PRIORITY_MODES),
                   help="Within-group key-selection priority (status §20 "
                        "control axis): snapkv = windowed SnapKV (default); "
                        "tova = SnapKV with last query row only (obs_window=1) "
                        "matching the TOVA priority of Oren et al. 2024; "
                        "recent/random/shared_random are attention-free "
                        "controls; k_norm uses ||K[i]||_2 (cheap, K-only); "
                        "accumulated is "
                        "H2O-style chunked full-prefill column-sum.")
    p.add_argument("--priority-seed", type=int, default=0,
                   help="Seed for the `random` priority (reproducible).")
    p.add_argument("--accumulated-chunk-size", type=int, default=512,
                   help="Q-row chunk size for the `accumulated` priority "
                        "(memory/time trade-off; 512 keeps [chunk, T] under "
                        "~64MB at T=32k in float32).")
    p.add_argument("--priority-heads-per-batch", type=int, default=8,
                   help="Head batch size for SnapKV windowed-priority "
                        "recompute. Lower values reduce temporary GPU "
                        "memory at very long lengths without changing the "
                        "measured priority.")
    p.add_argument("--extra-allocators", default="",
                   help="Comma-separated published-allocator names to "
                        "evaluate alongside the predeclared spread "
                        f"(any of {','.join(EXTRA_ALLOCATORS)}). Vectors are "
                        "appended per-ratio and recorded with their "
                        "`allocator_tag`; the analyzer's Named-allocator "
                        "section then lands them next to uniform/oracle.")
    p.add_argument("--skip-support", action="store_true",
                   help="Skip the per-prompt effective-context support "
                        "computation (per-head windowed attention — by far the "
                        "dominant cost, ~150s/prompt at 32k). Use when only "
                        "oracle_gap / worst_gap / mincost−uniform are needed; "
                        "the saturation predictor section of the report will "
                        "be empty for these prompts.")
    p.add_argument("--record-coverage-metrics", action="store_true",
                   help="Record retained-set coverage metrics for each "
                        "trajectory: union size, cross-group overlap, and "
                        "retained reference attention mass under a SnapKV "
                        "windowed-priority reference. This is an opt-in "
                        "mechanism probe and can add one windowed-priority "
                        "pass for non-SnapKV priorities.")
    p.add_argument("--priority-temperature", type=float, default=1.0,
                   help="Softmax temperature applied to attention BEFORE the "
                        "priority's column-sum (snapkv / accumulated only). "
                        "Default 1.0 reproduces standard behaviour. >1 "
                        "flattens the attention (raises effective PR); <1 "
                        "sharpens. P7 mechanism-falsification knob: BDPI "
                        "predicts that flattening LLaMA's attention via large "
                        "temperature produces Phi-like priority behaviour. "
                        "The cache and masked decode are unaffected; only the "
                        "priority selection changes.")
    p.add_argument("--floor-only", action="store_true",
                   help="P9 mechanism-falsification ablation. When set, each "
                        "ratio's per-group n_recent is scaled so that floor "
                        "= budget per group (i.e. discretionary budget = 0). "
                        "The retained set then consists of exactly the "
                        "n_sink first positions plus the (budget - n_sink) "
                        "last positions, regardless of priority. BDPI "
                        "predicts: under this regime, priority KL is "
                        "invariant to priority choice and follows the "
                        "model's intrinsic floor-tolerance — Phi should "
                        "stay at small KL, LLaMA/Qwen should jump.")
    p.add_argument("--record-retained-positions", action="store_true",
                   help="P10 mechanism probe: record positional histograms "
                        "and percentile statistics of retained tokens per "
                        "group. Adds per_group_position_hist_10bin and "
                        "per_group_mean/p10/p50/p90_retained_position fields "
                        "to each trajectory's coverage block. CPU-cheap "
                        "(no extra GPU work beyond what --record-coverage-"
                        "metrics already does).")
    p.add_argument("--record-decoder-queries", action="store_true",
                   help="P11 Jacobian probe: log the decoder's per-step "
                        "query vector for both reference and masked runs "
                        "(L2 norm only, not the full vector — keeps file "
                        "size manageable). Lets the analyzer compute "
                        "Δq^(t+1) / Δq^(t) as a proxy for the masked-decode "
                        "Jacobian norm. Adds 'reference_query_norms' and "
                        "'masked_query_norms' (lists of floats over decode "
                        "steps) to each prompt's record.")
    p.add_argument("--obs-window-position", default="last",
                   choices=["last", "first", "middle"],
                   help="P13 mechanism probe: which obs_window query rows "
                        "to use for snapkv's column-sum. Default 'last' "
                        "reproduces SnapKV. 'first' uses the prefill's "
                        "first w queries; 'middle' uses queries centered "
                        "on T/2. Tests whether Phi's recency-bias is due "
                        "to the last-w query window specifically.")
    p.add_argument("--decode-temperature", type=float, default=1.0,
                   help="P15 mechanism probe: divides decoder logits by τ "
                        "before softmax for both reference and masked. "
                        "Larger τ flattens the decoder's output, contracts "
                        "the masked-decode Jacobian. Tests whether LLaMA "
                        "can be pushed into the bias regime by sharpening "
                        "its output distribution.")
    p.add_argument("--per-head-snapkv", action="store_true",
                   help="P14 mechanism probe: compute snapkv priority "
                        "per-head (each head picks top-b per group), then "
                        "union retained sets. Tests whether aggregation-"
                        "across-heads in standard SnapKV is the proximate "
                        "cause of Phi's bias direction.")
    p.add_argument("--per-layer-weighted-priority", action="store_true",
                   help="P16 mechanism probe: weight each layer's snapkv "
                        "contribution by 1/PR_layer in the column-sum "
                        "aggregation. Lower-PR (peaked) layers contribute "
                        "more; higher-PR layers contribute less. Tests "
                        "whether a few high-PR layers cause the bias on "
                        "Phi.")
    p.add_argument("--gqa-emulation-factor", type=int, default=1,
                   help="P24 architectural-causality intervention. When "
                        ">1, key/value projections are hooked to average "
                        "kv-heads in groups of this size, emulating "
                        "grouped-query attention at the given factor "
                        "without retraining. On Phi-3.5-mini (multi-head "
                        "attention) factor 2/4/8 emulates GQA-2/4/8; on a "
                        "model that already uses GQA it pushes the kv "
                        "sharing further. Must divide num_key_value_heads. "
                        "Tests whether the bias regime is caused by the "
                        "absence of GQA: if the inversion shrinks as the "
                        "emulated factor rises, GQA is the causal axis.")
    p.add_argument("--layernorm-placement", default="pre",
                   choices=["pre", "post"],
                   help="P27 architectural-causality intervention. Default "
                        "'pre' is the unmodified pre-norm forward used by "
                        "Phi-3, LLaMA-3.2, and Qwen3 (norm BEFORE each "
                        "sublayer, residual sum AFTER). 'post' monkey-patches "
                        "every decoder layer to apply norm AFTER each "
                        "residual sum, swapping pre-norm -> post-norm at "
                        "inference. The intervention is destructive (the "
                        "trained weights expect normed sublayer inputs); the "
                        "reference forward shifts substantially. We use it "
                        "to test whether layer-norm placement is the "
                        "architectural feature placing Phi in the bias "
                        "regime (paper Sec 5.5, Limitation L4 candidate).")
    p.add_argument("--record-decoder-jacobian", action="store_true",
                   help="P20: at uniform-allocation trajectories, measure the "
                        "direct top-sigma of the masked-decode operator by "
                        "perturbing the K/V the masked decode added at step t "
                        "and power-iterating on J^T J via finite-difference "
                        "JVPs + autograd VJPs. Pairs the resulting rho_R "
                        "against the existing rho-hat_R proxy. Wall-time "
                        "overhead ~2x for the canonical 4-cell P11 sweep.")
    p.add_argument("--jacobian-measure-steps", type=int, nargs="+",
                   default=[3, 7, 11, 14],
                   help="Masked-decode step indices at which to measure "
                        "top-sigma (P20). Each step s must satisfy "
                        "1 <= s <= k_dec - 2 so step s+1 exists. Default "
                        "spans the proxy's averaging range; reduce to "
                        "{7, 14} for a cheaper smoke test.")
    p.add_argument("--jacobian-n-iter", type=int, default=10,
                   help="Power-iteration steps per top-sigma measurement (P20). "
                        "10 is typically enough for the dominant singular "
                        "direction to converge; reduce to 5 for a smoke test.")
    p.add_argument("--jacobian-eps", type=float, default=0.05,
                   help="Per-entry probe magnitude in the P20 power "
                        "iteration's central-difference JVP. Internally the "
                        "probe is `eps * sqrt(total_dim) * v` for unit-norm "
                        "v, so per-entry perturbation magnitude is `eps`. "
                        "Default 0.05 is bf16-safe: per-entry probes need "
                        "to exceed ~1 bf16 ulp (~7.8e-3 at unit scale) for "
                        "the cache update to register. The initial smoke "
                        "test used the old eps=1e-3 (probe norm, not per-"
                        "entry) which rounded to zero in bf16; the new "
                        "parametrisation fixes that.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory (an analyses/2026-05-DD_* folder); "
                        "eval_config.json + trajectories.json written inside.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    ratios = parse_ratios(args.ratios)
    extra_allocators = tuple(
        name.strip() for name in args.extra_allocators.split(",")
        if name.strip()
    )
    for name in extra_allocators:
        if name not in EXTRA_ALLOCATORS:
            raise ValueError(
                f"unknown extra allocator {name!r}; "
                f"expected one of {EXTRA_ALLOCATORS}"
            )
    records = load_manifest_records(args.manifest)
    print(f"[lc-probe] manifest {args.manifest}: {len(records)} prompts; "
          f"ratios={ratios}", flush=True)

    prefill_attn, masked_attn = attention_backends_for_model(args.model)
    print(
        f"[lc-probe] loading {args.model} "
        f"({prefill_attn} prefill / {masked_attn} masked decode)",
        flush=True,
    )
    model, tokenizer = load_model_and_tokenizer(
        args.model, device=args.device, attn_implementation=prefill_attn
    )
    device = resolve_model_input_device(model, args.device)
    num_layers = int(model.config.num_hidden_layers)

    # P24 architectural-causality intervention. With --gqa-emulation-factor > 1
    # the model's key/value projections are hooked to average kv-heads in
    # groups, emulating grouped-query attention. The hooks act on every
    # attention layer and persist for both the prefill capture and the masked
    # decode, so the whole pipeline runs the GQA-emulated model.
    if args.gqa_emulation_factor and args.gqa_emulation_factor > 1:
        install_gqa_emulation_hooks(model, int(args.gqa_emulation_factor))

    # P27 layer-norm-placement intervention. When `--layernorm-placement post`
    # the model's decoder layers are monkey-patched to apply RMSNorm AFTER
    # each residual sum (post-norm) rather than BEFORE the sublayer
    # (pre-norm). This is a destructive intervention on a pre-norm-trained
    # model; the reference forward shifts substantially and the masked
    # decode is measured against the modified reference. The hooks persist
    # for the whole producer run.
    if getattr(args, "layernorm_placement", "pre") == "post":
        install_post_norm_hooks(model)

    eval_config = {
        "manifest": str(args.manifest),
        "model": args.model,
        "device": str(device),
        "chat_template": "never",
        "attn_backend": (
            f"{prefill_attn} prefill + capture / {masked_attn} masked decode"
        ),
        "k_dec": args.k_dec,
        "obs_window": args.obs_window,
        "num_groups": args.num_groups,
        "n_sink": args.n_sink,
        "n_recent": args.n_recent,
        "ratios": ratios,
        "support_estimator": args.support_estimator,
        "support_tau": args.support_tau,
        "priority": args.priority,
        "priority_seed": args.priority_seed,
        "accumulated_chunk_size": args.accumulated_chunk_size,
        "priority_heads_per_batch": args.priority_heads_per_batch,
        "extra_allocators": list(extra_allocators),
        "skip_support": args.skip_support,
        "record_coverage_metrics": args.record_coverage_metrics,
        "priority_temperature": args.priority_temperature,
        "floor_only": args.floor_only,
        "record_retained_positions": args.record_retained_positions,
        "record_decoder_queries": args.record_decoder_queries,
        "obs_window_position": args.obs_window_position,
        "decode_temperature": args.decode_temperature,
        "per_head_snapkv": args.per_head_snapkv,
        "per_layer_weighted_priority": args.per_layer_weighted_priority,
        "gqa_emulation_factor": args.gqa_emulation_factor,
        "layernorm_placement": args.layernorm_placement,
        "record_decoder_jacobian": args.record_decoder_jacobian,
        "jacobian_measure_steps": list(args.jacobian_measure_steps),
        "jacobian_n_iter": args.jacobian_n_iter,
        "jacobian_eps": args.jacobian_eps,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.out / "eval_config.json", eval_config)

    # Checkpoint trajectories.json after every prompt (atomic rewrite):
    # a cancelled or wall-time-killed run then keeps every prompt finished
    # so far instead of discarding the whole producer pass.
    results: list[dict[str, Any]] = []
    _write_json_atomic(
        args.out / "trajectories.json", _trajectories_payload(eval_config, results)
    )
    for i, record in enumerate(records):
        print(f"[lc-probe] {i + 1}/{len(records)}  L≈{record.tier_length}  "
              f"{record.task}  {record.prompt_id}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            result = _run_prompt(
                record, model=model, tokenizer=tokenizer, device=device,
                num_layers=num_layers, num_groups=args.num_groups,
                obs_window=args.obs_window, k_dec=args.k_dec,
                n_sink=args.n_sink, n_recent=args.n_recent, ratios=ratios,
                support_estimator=args.support_estimator,
                support_tau=args.support_tau,
                priority=args.priority, priority_seed=args.priority_seed,
                skip_support=args.skip_support,
                accumulated_chunk_size=args.accumulated_chunk_size,
                priority_heads_per_batch=args.priority_heads_per_batch,
                extra_allocators=extra_allocators,
                record_coverage_metrics=args.record_coverage_metrics,
                masked_attn_implementation=masked_attn,
                priority_temperature=args.priority_temperature,
                floor_only=args.floor_only,
                record_retained_positions=args.record_retained_positions,
                obs_window_position=args.obs_window_position,
                decode_temperature=args.decode_temperature,
                per_head_snapkv=args.per_head_snapkv,
                per_layer_weighted_priority=args.per_layer_weighted_priority,
                record_decoder_queries=args.record_decoder_queries,
                record_decoder_jacobian=args.record_decoder_jacobian,
                jacobian_measure_steps=tuple(args.jacobian_measure_steps),
                jacobian_n_iter=args.jacobian_n_iter,
                jacobian_eps=args.jacobian_eps,
                jacobian_priority_seed=args.priority_seed,
            )
            print(f"[lc-probe]   ok  tokens={result['tokens']}  "
                  f"capture={result['capture_prefill_s']:.2f}s  "
                  f"support={result['support_compute_s']:.2f}s  "
                  f"coverage={result['coverage_compute_s']:.2f}s  "
                  f"prefill={result['prefill_s']:.2f}s  "
                  f"masked={result['masked_decode_s']:.2f}s  "
                  f"trajectories={result['n_trajectories']}  "
                  f"S_g={'skipped' if not result.get('effective_support') else [round(s, 1) for s in result['effective_support']['per_group']]}  "
                  f"peak={result.get('peak_mem_gb', float('nan')):.2f}GB",
                  flush=True)
        except Exception as exc:  # OOM or any per-prompt failure
            status = type(exc).__name__
            print(f"[lc-probe]   FAIL ({status}): {exc}", flush=True)
            traceback.print_exc()
            result = {
                "prompt_id": record.prompt_id,
                "task": record.task,
                "tier_length": record.tier_length,
                "status": status,
                "error": str(exc),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(result)
        _write_json_atomic(
            args.out / "trajectories.json",
            _trajectories_payload(eval_config, results),
        )

    completed = [r for r in results if r.get("status") == "ok"]
    print(f"[lc-probe] wrote {args.out}/ (eval_config.json, trajectories.json) "
          f"— {len(completed)}/{len(results)} ok", flush=True)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
