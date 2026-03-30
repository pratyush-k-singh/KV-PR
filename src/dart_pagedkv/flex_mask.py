"""FlexAttention hot/cold masking for long-context decode.

The eager path (`logit_kl.build_group_attention_masks`) builds a
``[1,1,T,T]`` additive mask per layer-group — O(T²) memory, capping the
H100 at ~16k context. This module expresses the same hot/cold structure
as a FlexAttention ``mask_mod`` over an O(T) per-key boolean ``is_hot``
vector, so the masked forward runs at flash-class memory and 32k–131k
context becomes feasible.

Spec: docs/superpowers/specs/2026-05-16-flexattention-longcontext.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.nn.attention.flex_attention import create_block_mask

from src.trace.collect import _find_attention_modules


def build_is_hot_vector(
    hot_indices: Sequence[int],
    total_seq_len: int,
    key_boundary: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-key visibility vector for one layer-group.

    Returns a boolean ``[total_seq_len]`` tensor: ``True`` where a key is
    visible to this group. A key is visible iff it is a hot prompt key
    (in ``hot_indices``, all ``< key_boundary``) OR a decoded position
    (``>= key_boundary`` — decoded tokens are always hot, spec §3).
    """
    # Built on CPU with a single fancy-index assignment, then moved once.
    # A per-element loop on a CUDA tensor fires one kernel per hot index
    # — fine for a sink+recent set (~hundreds) but ~T kernels for an
    # all-hot set (the faithfulness control / a high budget at 131k).
    is_hot = torch.zeros(total_seq_len, dtype=torch.bool)
    valid = [j for j in (int(h) for h in hot_indices) if 0 <= j < total_seq_len]
    if valid:
        is_hot[torch.tensor(valid, dtype=torch.long)] = True
    if key_boundary < total_seq_len:
        is_hot[key_boundary:] = True
    return is_hot.to(device)


def make_group_mask_mod(is_hot: torch.Tensor, q_start: int = 0) -> Callable:
    """Build a FlexAttention ``mask_mod`` from a per-key ``is_hot`` vector.

    The returned ``mask_mod(b, h, q_idx, kv_idx)`` is ``True`` for an
    allowed (query, key) pair: causal (``kv_idx <= q_idx + q_start``)
    AND the key visible to this group (``is_hot[kv_idx]``).

    ``q_start`` is the **absolute position** of the query block's first
    token. For a full prefill it is 0 (``q_idx`` already is the absolute
    position). For cached incremental decode the query is length 1 and
    FlexAttention passes ``q_idx = 0`` while ``kv_idx`` spans the whole
    cache — ``q_start`` must then be the decoded token's absolute
    position, or the causal check would allow only key 0.

    ``create_block_mask`` pads the sequence to a block multiple and
    evaluates ``mask_mod`` past ``len(is_hot)``; the ``in_range`` guard
    blocks those padding positions and keeps the gather in bounds.
    ``mask_mod`` is called with broadcast tensor indices.
    """
    n = int(is_hot.shape[0])

    def mask_mod(b, h, q_idx, kv_idx):
        in_range = kv_idx < n
        safe_kv = torch.where(in_range, kv_idx, torch.zeros_like(kv_idx))
        return (kv_idx <= q_idx + q_start) & in_range & is_hot[safe_kv]

    return mask_mod


def build_group_block_masks(
    hot_per_group: Sequence[Sequence[int]],
    kv_len: int,
    key_boundary: int,
    *,
    q_len: int | None = None,
    q_start: int = 0,
    device: torch.device | str = "cpu",
) -> dict[int, object]:
    """Per-layer-group FlexAttention ``BlockMask`` — the O(T) analogue of
    ``logit_kl.build_group_attention_masks``.

    ``kv_len`` is the key/cache length; ``q_len`` defaults to ``kv_len``
    (square prefill). For a cached decode step set ``q_len=1`` and
    ``q_start`` to the decoded token's absolute position (see
    :func:`make_group_mask_mod`).

    Raises ``ValueError`` if a group's hot set excludes key 0 (the
    attention sink): the position-0 query would then have no visible key,
    a degenerate attention row — same guard as the eager path.
    """
    if q_len is None:
        q_len = kv_len
    masks: dict[int, object] = {}
    prompt_len = min(int(key_boundary), kv_len)
    for group_idx, hot in enumerate(hot_per_group):
        hot_set = {int(h) for h in hot}
        if prompt_len > 0 and 0 not in hot_set:
            raise ValueError(
                f"group {group_idx}: key 0 (the attention sink) is cold. "
                f"The position-0 query would have an all-masked attention "
                f"row. Keep each group's protected sink positions hot."
            )
        is_hot = build_is_hot_vector(
            sorted(hot_set), kv_len, key_boundary, device=device
        )
        masks[group_idx] = create_block_mask(
            make_group_mask_mod(is_hot, q_start=q_start),
            B=None, H=None,
            Q_LEN=q_len, KV_LEN=kv_len,
            device=str(device),
        )
    return masks


def install_per_layer_block_mask_hooks(
    model,
    group_block_masks: dict[int, object],
    layer_to_group: dict[int, int],
) -> list:
    """Register forward pre-hooks that swap each attention layer's mask.

    The flex analogue of ``logit_kl.install_per_layer_mask_hooks``: each
    tracked attention layer's ``attention_mask`` kwarg is replaced with
    its layer-group's ``BlockMask``. Layers absent from
    ``layer_to_group`` keep their original (HF-built causal) mask.
    Returns the hook handles; the caller must remove them.

    Whether HF's ``attn_implementation="flex_attention"`` path actually
    consumes a pre-hooked ``BlockMask`` is the Step-3 integration gate —
    verified by `experiments/scripts/flex_integration_check.py`, not
    assumed.
    """
    handles = []
    attn_modules = _find_attention_modules(model)
    for layer_idx, (_, module) in enumerate(attn_modules):
        group_idx = layer_to_group.get(layer_idx)
        if group_idx is None or group_idx not in group_block_masks:
            continue
        block_mask = group_block_masks[group_idx]

        def _pre_hook(_module, args, kwargs, _mask=block_mask):
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = _mask
            return args, kwargs

        handles.append(module.register_forward_pre_hook(_pre_hook, with_kwargs=True))
    return handles
