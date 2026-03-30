"""GQA-emulation intervention for the P24 architectural-causality test.

The audit's mechanism analysis leaves one feature confounded: Phi-3.5-mini
differs from LLaMA / Qwen on three architectural axes at once (no GQA,
much higher participation ratio, layer-norm placement). To isolate the
GQA axis we need an intervention that changes GQA while holding the rest
of the model fixed.

`install_gqa_emulation_hooks` registers forward hooks on every attention
layer's key/value projection that average key-value heads within groups
of size ``factor`` and broadcast the group mean back, so that ``factor``
query heads come to share one effective KV head. On a multi-head model
(Phi-3.5-mini, where ``num_key_value_heads == num_attention_heads``) this
emulates grouped-query attention at GQA factor ``factor`` without
retraining; on a model that already uses GQA it pushes the KV sharing
further.

The hooks act on the projection output, the earliest point in the
attention block, so RoPE, the attention matmul, the Q/K capture used to
build the priority, and the masked decode all see the emulated keys and
values. The masked-decode KL is therefore measured against a reference
that is itself GQA-emulated; the comparison isolates the priority effect
under the modified architecture.

Two projection layouts are handled: separate ``k_proj`` / ``v_proj``
(LLaMA, Qwen) and fused ``qkv_proj`` (Phi-3). The intervention is a pure
inference-time tensor operation and adds negligible compute.
"""

from __future__ import annotations

import torch


def _group_average_kv(
    tensor: torch.Tensor, n_kv_heads: int, head_dim: int, factor: int
) -> torch.Tensor:
    """Average KV heads within contiguous groups of ``factor`` and
    broadcast the group mean back.

    ``tensor`` has trailing dimension ``n_kv_heads * head_dim``; the
    return has the same shape. With ``factor`` query heads sharing a KV
    head, this is exactly the KV tensor a GQA-``factor`` model would
    produce from the same per-head projections.
    """
    *lead, last = tensor.shape
    expected = n_kv_heads * head_dim
    if last != expected:
        raise ValueError(
            f"GQA-emulation hook expected trailing dim {expected} "
            f"(= {n_kv_heads} kv-heads x {head_dim}), got {last}"
        )
    if factor <= 1:
        return tensor
    if n_kv_heads % factor != 0:
        raise ValueError(
            f"gqa_emulation_factor {factor} must divide num_key_value_heads "
            f"{n_kv_heads}"
        )
    n_groups = n_kv_heads // factor
    x = tensor.reshape(*lead, n_groups, factor, head_dim)
    x = x.mean(dim=-2, keepdim=True)                 # [..., n_groups, 1, head_dim]
    x = x.expand(*lead, n_groups, factor, head_dim)
    return x.reshape(*lead, n_kv_heads * head_dim).contiguous()


def install_gqa_emulation_hooks(model, factor: int) -> list:
    """Register GQA-emulation forward hooks on every attention layer.

    Returns the list of hook handles (call ``.remove()`` on each to
    revert). ``factor <= 1`` installs nothing and returns an empty list.
    Raises if ``factor`` does not divide the model's
    ``num_key_value_heads`` or if no recognised projection layout is
    found.
    """
    if factor is None or factor <= 1:
        return []

    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // n_heads))
    if n_kv % factor != 0:
        raise ValueError(
            f"gqa_emulation_factor {factor} must divide num_key_value_heads "
            f"{n_kv} for {model.__class__.__name__}"
        )

    handles = []
    n_separate = 0
    n_fused = 0
    for name, module in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        if hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            def kv_hook(m, inp, out, n_kv=n_kv, hd=head_dim, f=factor):
                return _group_average_kv(out, n_kv, hd, f)
            handles.append(module.k_proj.register_forward_hook(kv_hook))
            handles.append(module.v_proj.register_forward_hook(kv_hook))
            n_separate += 1
        elif hasattr(module, "qkv_proj"):
            q_size = n_heads * head_dim
            kv_size = n_kv * head_dim

            def qkv_hook(m, inp, out, qs=q_size, kvs=kv_size,
                         n_kv=n_kv, hd=head_dim, f=factor):
                q = out[..., :qs]
                k = out[..., qs:qs + kvs]
                v = out[..., qs + kvs:qs + 2 * kvs]
                k = _group_average_kv(k, n_kv, hd, f)
                v = _group_average_kv(v, n_kv, hd, f)
                return torch.cat([q, k, v], dim=-1)

            handles.append(module.qkv_proj.register_forward_hook(qkv_hook))
            n_fused += 1

    if not handles:
        raise ValueError(
            f"GQA-emulation found no recognised attention projection on "
            f"{model.__class__.__name__}; expected per-layer self_attn "
            f"modules with k_proj/v_proj or qkv_proj"
        )
    print(
        f"[gqa-emulation] factor={factor}: hooked {n_separate} separate-"
        f"projection and {n_fused} fused-projection attention layers "
        f"(n_kv_heads={n_kv}, head_dim={head_dim}; emulated GQA factor "
        f"{factor}, effective shared kv-heads {n_kv // factor})",
        flush=True,
    )
    return handles
