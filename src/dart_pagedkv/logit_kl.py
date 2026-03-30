"""Logit-KL fidelity rung for the Stage-0 bridge.

Attention-output deviation (``fidelity.py``) measures per-layer distortion under
the full-KV hidden-state trajectory. It does not capture how that distortion
propagates through later layers to the model's actual prediction. Logit-KL is
the stronger, design-required rung: run the model with each layer-group's cold
keys masked out, and measure the KL divergence between the hot-only next-token
distribution and the full-KV one at the decode-shaped positions.

This module has two pure, unit-tested pieces (``token_kl_divergence`` and
``build_group_attention_masks``) and the model-integrated plumbing
(``install_per_layer_mask_hooks``, ``compute_hot_only_logits``,
``logit_kl_for_budget``) which must be validated on a GPU run.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.trace.collect import _find_attention_modules


def token_kl_divergence(
    full_logits: torch.Tensor, hot_logits: torch.Tensor
) -> float:
    """Mean per-position KL(full || hot) between two ``[n_pos, vocab]`` logit sets.

    ``full_logits`` is the full-KV next-token logits; ``hot_logits`` is the
    hot-only-cache logits. Computed in float32 for numerical stability.
    """
    full = torch.as_tensor(full_logits, dtype=torch.float32)
    hot = torch.as_tensor(hot_logits, dtype=torch.float32)
    if full.shape != hot.shape:
        raise ValueError(f"shape mismatch: {tuple(full.shape)} vs {tuple(hot.shape)}")
    if full.numel() == 0 or full.shape[0] == 0:
        return 0.0
    full_logp = torch.log_softmax(full, dim=-1)
    hot_logp = torch.log_softmax(hot, dim=-1)
    full_p = torch.exp(full_logp)
    kl = (full_p * (full_logp - hot_logp)).sum(dim=-1)  # [n_pos]
    return float(kl.mean().item())


def build_group_attention_masks(
    hot_per_group: Sequence[Sequence[int]],
    total_seq_len: int,
    key_boundary: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[int, torch.Tensor]:
    """Per-group additive attention mask of shape ``[1, 1, T, T]``.

    For group ``g``, a key position is blocked (``-inf``) when it is a future
    position (causal) or a *cold* prompt key -- a position in
    ``[0, key_boundary)`` not in ``hot_per_group[g]``. Positions at or past
    ``key_boundary`` (the answer / decode tail) are never cold; they stay hot
    subject only to causal masking. Allowed positions carry ``0.0``.
    """
    device = torch.device(device)
    positions = torch.arange(total_seq_len, device=device)
    future = positions.unsqueeze(0) > positions.unsqueeze(1)  # [q, k]: True where k > q
    masks: dict[int, torch.Tensor] = {}
    for group_idx, hot in enumerate(hot_per_group):
        hot_set = {int(h) for h in hot}
        prompt_len = min(int(key_boundary), total_seq_len)
        if prompt_len > 0 and 0 not in hot_set:
            raise ValueError(
                f"group {group_idx}: key 0 (the attention sink) is cold. The "
                f"position-0 query can only attend to key 0, so it would get an "
                f"all-masked attention row (softmax -> NaN) in the logit-KL "
                f"forward. Keep each group's protected sink positions hot -- "
                f"raise the budget floor so no group is starved below it."
            )
        cold = [j for j in range(prompt_len) if j not in hot_set]
        mask = torch.zeros(1, 1, total_seq_len, total_seq_len, dtype=dtype, device=device)
        mask.masked_fill_(future.unsqueeze(0).unsqueeze(0), float("-inf"))
        if cold:
            cold_idx = torch.tensor(cold, dtype=torch.long, device=device)
            mask[:, :, :, cold_idx] = float("-inf")
        masks[group_idx] = mask
    return masks


# --------------------------------------------------------------------------
# Model-integrated plumbing. The pure functions above are unit-tested; the
# hook-installation closure (the late-binding-prone part) is tested with a
# fake model. Whether HF's attention layer actually consumes the swapped-in
# ``attention_mask`` is validated only on a GPU run -- see the note on
# install_per_layer_mask_hooks.
# --------------------------------------------------------------------------


def install_per_layer_mask_hooks(
    model,
    group_masks: dict[int, torch.Tensor],
    layer_to_group: dict[int, int],
) -> list:
    """Register forward pre-hooks that swap each attention layer's mask.

    Each tracked layer's ``attention_mask`` kwarg is replaced with its group's
    precomputed additive mask. Layers absent from ``layer_to_group`` are left
    untouched (they run with the model's own full-causal mask). Returns the
    hook handles; the caller must remove them.

    GPU-validation note: this assumes the attention layer's ``forward`` takes
    ``attention_mask`` as a keyword and applies it additively. Use
    ``attn_implementation="eager"`` for unambiguous semantics -- flash-attention
    does not support arbitrary 4D masks, and the SDPA path can route through an
    ``is_causal`` shortcut. If a future HF version passes ``attention_mask``
    positionally, this hook must be adapted (or the attention function
    monkey-patched instead).
    """
    handles = []
    attn_modules = _find_attention_modules(model)
    for layer_idx, (_, module) in enumerate(attn_modules):
        group_idx = layer_to_group.get(layer_idx)
        if group_idx is None or group_idx not in group_masks:
            continue
        layer_mask = group_masks[group_idx]

        def _pre_hook(_module, args, kwargs, _mask=layer_mask):
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = _mask
            return args, kwargs

        handles.append(module.register_forward_pre_hook(_pre_hook, with_kwargs=True))
    return handles


def compute_hot_only_logits(
    model,
    inputs: dict,
    group_masks: dict[int, torch.Tensor],
    layer_to_group: dict[int, int],
) -> torch.Tensor:
    """Run one full forward with per-layer cold-key masking; return logits.

    GPU-validation note: see install_per_layer_mask_hooks.
    """
    handles = install_per_layer_mask_hooks(model, group_masks, layer_to_group)
    try:
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, return_dict=True)
        return outputs.logits
    finally:
        for handle in handles:
            handle.remove()


def logit_kl_for_budget(
    model,
    inputs: dict,
    full_logits: torch.Tensor,
    *,
    hot_per_group: Sequence[Sequence[int]],
    layer_to_group: dict[int, int],
    total_seq_len: int,
    key_boundary: int,
    query_start: int,
    query_end: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> float:
    """Logit-KL for one budget vector: build masks, forward, KL at query positions.

    ``full_logits`` is the full-KV logits ``[batch, seq, vocab]`` captured once
    per prompt. GPU-validation note: see install_per_layer_mask_hooks.
    """
    group_masks = build_group_attention_masks(
        hot_per_group, total_seq_len, key_boundary, dtype=dtype, device=device
    )
    try:
        hot_logits = compute_hot_only_logits(
            model, inputs, group_masks, layer_to_group
        )
        full_slice = full_logits[0, query_start:query_end]
        # Slice the small query window first, then reconcile devices: the
        # captured full logits live on CPU, the fresh hot logits on the model
        # device.
        hot_slice = hot_logits[0, query_start:query_end].to(full_slice.device)
        del hot_logits
        return token_kl_divergence(full_slice, hot_slice)
    finally:
        del group_masks
