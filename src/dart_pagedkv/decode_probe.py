"""Decode-level DART transport simulator.

Per-decode-step λ-update against pre-computed pure-ADV and pure-ROB
logit-KL trajectories. The simulator is path-dependent for the played
cache (a callback ``step_played(b_play, state) → (logit_kl, new_state)``
advances it one step at a time); ADV and ROB are deterministic policies
so their per-step trajectories can be pre-computed once per prompt and
passed in as arrays.

Spec: docs/superpowers/specs/2026-05-16-dart-pagedkv-decode-probe.md (§3).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .budget import interpolate_budget
from .logit_kl import (
    build_group_attention_masks,
    install_per_layer_mask_hooks,
    token_kl_divergence,
)
from .transport import snap_to_recorded


@dataclass(frozen=True)
class DecodeProbeConfig:
    rob_budget: tuple[int, ...]
    total: int
    floors: tuple[int, ...]
    K_dec: int
    K_probe: int
    eta: float = 1.0
    lam_init: float = 0.0
    eta_decay: float = 0.0


@dataclass
class DartDecodeTrace:
    lam_history: list[float] = field(default_factory=list)
    budgets_played: list[tuple[int, ...]] = field(default_factory=list)
    logit_kl_played: list[float] = field(default_factory=list)
    probe_regret_history: list[tuple[int, float]] = field(default_factory=list)
    final_lam: float = 0.0  # λ AFTER the last step's update (for cross-prompt persistence)


def _build_decode_inputs(
    prompt_ids: torch.Tensor, decoded_so_far: Sequence[int]
) -> torch.Tensor:
    """Concatenate the prefill ids with the tokens decoded so far."""
    if not decoded_so_far:
        return prompt_ids
    tail = torch.tensor(
        [list(decoded_so_far)], dtype=prompt_ids.dtype, device=prompt_ids.device,
    )
    return torch.cat([prompt_ids, tail], dim=1)


def _masked_step_kl(
    model,
    prompt_ids: torch.Tensor,
    decoded_so_far: Sequence[int],
    hot_per_group: Sequence[Sequence[int]],
    layer_to_group: dict[int, int],
    ref_logit: torch.Tensor,
    key_boundary: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> float:
    """One masked decode-step forward; KL of last-position logits vs ``ref_logit``.

    ``build_group_attention_masks`` auto-protects positions at or past
    ``key_boundary`` (decoded tokens are always hot, spec §3.1). Hooks are
    installed for this forward only and removed afterward.
    """
    cur_inputs = _build_decode_inputs(prompt_ids, decoded_so_far)
    masks = build_group_attention_masks(
        hot_per_group, cur_inputs.shape[1], key_boundary,
        dtype=dtype, device=device,
    )
    handles = install_per_layer_mask_hooks(model, masks, layer_to_group)
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=cur_inputs, use_cache=False, return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()
    step_logits = outputs.logits[0, -1, :]
    ref = ref_logit.to(step_logits.device)
    return token_kl_divergence(step_logits.unsqueeze(0), ref.unsqueeze(0))


def decode_with_mask(
    model,
    prompt_ids: torch.Tensor,
    gold_continuation: Sequence[int],
    hot_per_group_fn: Callable[[int], Sequence[Sequence[int]]],
    layer_to_group: dict[int, int],
    full_kv_ref_logits: torch.Tensor,
    key_boundary: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Run ``K_dec = len(gold_continuation)`` masked decode steps.

    At each step ``t``: build inputs from the prefill plus the gold tokens
    decoded so far, mask per ``hot_per_group_fn(t)``, run one masked forward,
    KL the last-position logits vs ``full_kv_ref_logits[t]``, then teacher-force
    ``gold_continuation[t]``. Used to pre-compute the pure-ADV and pure-ROB
    trajectories (constant ``hot_per_group_fn``).
    """
    decoded_so_far: list[int] = []
    kls: list[float] = []
    for t in range(len(gold_continuation)):
        kls.append(_masked_step_kl(
            model, prompt_ids, decoded_so_far, hot_per_group_fn(t),
            layer_to_group, full_kv_ref_logits[t], key_boundary,
            dtype=dtype, device=device,
        ))
        decoded_so_far.append(int(gold_continuation[t]))
    return kls


def compute_full_kv_reference(
    model,
    prompt_ids: torch.Tensor,
    K_dec: int,
) -> tuple[list[int], torch.Tensor]:
    """Greedy-decode ``K_dec`` tokens under full (unmasked) attention.

    Returns ``(gold_continuation, ref_logits)`` where ``ref_logits`` is
    ``[K_dec, vocab]`` on CPU. The greedy argmax sequence becomes the gold
    continuation every masked policy is teacher-forced against, so the
    experiment measures each budget policy's deviation from full-KV behavior.
    """
    decoded: list[int] = []
    ref_rows: list[torch.Tensor] = []
    for _ in range(K_dec):
        cur_inputs = _build_decode_inputs(prompt_ids, decoded)
        with torch.inference_mode():
            outputs = model(
                input_ids=cur_inputs, use_cache=False, return_dict=True,
            )
        step_logits = outputs.logits[0, -1, :].float()
        ref_rows.append(step_logits.detach().cpu())
        decoded.append(int(step_logits.argmax().item()))
    return decoded, torch.stack(ref_rows)


def make_dart_step_played(
    model,
    prompt_ids: torch.Tensor,
    gold_continuation: Sequence[int],
    budget_to_hot_fn: Callable[[tuple[int, ...]], Sequence[Sequence[int]]],
    layer_to_group: dict[int, int],
    full_kv_ref_logits: torch.Tensor,
    key_boundary: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    kl_cache: dict[tuple[tuple[int, ...], int], float] | None = None,
) -> Callable[[tuple[int, ...], Any], tuple[float, Any]]:
    """Build a ``step_played`` callback for :func:`simulate_dart_decode`.

    The callback maintains ``state = {"t": int, "decoded": list[int]}``,
    converts the played budget ``b_play`` to per-group hot sets via
    ``budget_to_hot_fn``, runs one masked forward, and teacher-forces the
    next gold token.

    ``kl_cache`` (optional) memoizes ``(b_play, t) → kl``. Because the gold
    continuation is fixed per prompt, the decoded prefix at step ``t`` is
    always ``gold[:t]``, so ``(b_play, t)`` fully determines the KL. Sharing
    one cache across all (scenario, K_probe, shuffle) runs for a prompt keeps
    the forward count independent of ``n_shuffles``.
    """
    def step_played(b_play: tuple[int, ...], state: Any) -> tuple[float, Any]:
        t = state.get("t", 0)
        decoded = state.get("decoded", [])
        b_key = tuple(int(x) for x in b_play)
        cache_key = (b_key, t)
        if kl_cache is not None and cache_key in kl_cache:
            kl = kl_cache[cache_key]
        else:
            hot_per_group = budget_to_hot_fn(b_key)
            kl = _masked_step_kl(
                model, prompt_ids, decoded, hot_per_group, layer_to_group,
                full_kv_ref_logits[t], key_boundary, dtype=dtype, device=device,
            )
            if kl_cache is not None:
                kl_cache[cache_key] = kl
        new_state = {"t": t + 1, "decoded": decoded + [int(gold_continuation[t])]}
        return kl, new_state
    return step_played


def make_lookup_step_played(
    played_traj_by_budget: dict[tuple[int, ...], Sequence[float]],
    gold_continuation: Sequence[int],
) -> Callable[[tuple[int, ...], Any], tuple[float, Any]]:
    """Build a pure-lookup ``step_played`` callback for :func:`simulate_dart_decode`.

    Instead of a model forward, the played budget is snapped to the nearest
    pre-computed budget on the recorded grid and its step-``t`` logit-KL is
    looked up from ``played_traj_by_budget``. With all 9 v3-grid budget
    trajectories pre-computed once per prompt, the DART workload loop runs
    with no model forwards — so K_probe values and MC shuffles cost nothing.
    This is the spec §3.6 snap-to-9 path, adopted after the smoke showed the
    continuous-played workload is many GPU-hours.
    """
    budgets = list(played_traj_by_budget.keys())

    def step_played(b_play: tuple[int, ...], state: Any) -> tuple[float, Any]:
        t = state.get("t", 0)
        decoded = state.get("decoded", [])
        _, snapped = snap_to_recorded(tuple(int(x) for x in b_play), budgets)
        kl = float(played_traj_by_budget[snapped][t])
        new_state = {"t": t + 1, "decoded": decoded + [int(gold_continuation[t])]}
        return kl, new_state
    return step_played


def simulate_dart_decode(
    adv_kl_per_step: Sequence[float],
    rob_kl_per_step: Sequence[float],
    step_played: Callable[[tuple[int, ...], Any], tuple[float, Any]],
    initial_played_state: Any,
    adv_budget: Sequence[int],
    config: DecodeProbeConfig,
) -> DartDecodeTrace:
    """Run one decode-level DART simulation.

    Stub form: the simplest implementation that passes the zero-regret
    test. Behavior will expand under further tests.
    """
    trace = DartDecodeTrace()
    state = initial_played_state
    lam = config.lam_init
    cached_regret: float | None = None
    b_adv = tuple(int(x) for x in adv_budget)
    for t in range(config.K_dec):
        trace.lam_history.append(lam)
        b_play = tuple(interpolate_budget(
            b_adv, config.rob_budget, lam, config.total, config.floors,
        ))
        kl, state = step_played(b_play, state)
        trace.budgets_played.append(b_play)
        trace.logit_kl_played.append(float(kl))

        if t % config.K_probe == 0:
            cached_regret = float(adv_kl_per_step[t]) - float(rob_kl_per_step[t])
            trace.probe_regret_history.append((t, cached_regret))

        if cached_regret is not None:
            eta_t = (
                config.eta / (1.0 + config.eta_decay * t)
                if config.eta_decay > 0
                else config.eta
            )
            lam = max(0.0, min(1.0, lam - eta_t * cached_regret))
    trace.final_lam = lam
    return trace
