"""Post-norm intervention for the P27 architectural-causality test.

Phi-3.5-mini, LLaMA-3.2, and Qwen3 all use sequential pre-norm RMSNorm in
their decoder blocks: ``y = x + sublayer(norm(x))``. P27 swaps that to
post-norm at inference: ``y = norm(x + sublayer(x))``. The intervention
is a destructive change to the forward graph -- the model's weights were
trained to receive normalised inputs at each sublayer -- but it is the
cleanest way to test whether the "layer-norm placement" axis (as named
in Final_Paper/main.tex Sec 5.5 paragraph 2) contributes to the regime.

We expect three possible outcomes:

  1. The model degrades catastrophically under post-norm (huge full-KV
     reference KL even before masking); both priorities become equally
     bad, and the snapkv-vs-random inversion vanishes -- a degraded-
     model artifact, *not* evidence about the architectural cause. The
     Qwen-P24a high-factor results have the same artifact shape.

  2. The model degrades modestly but the snapkv-vs-random inversion
     persists; layer-norm placement is *not* what places Phi in the
     bias regime. Combined with the two GQA-emulation interventions
     ruling out KV-head sharing, this would leave per-layer
     participation ratio as the sole remaining candidate.

  3. The model survives post-norm and the inversion shrinks or vanishes;
     layer-norm placement is implicated as the regime-defining feature.

Outcomes (1) and (2) are the likely cases; outcome (3) would be a
strong positive result.

The intervention is implemented by monkey-patching ``Phi3DecoderLayer``
instances (and the equivalents for LLaMA / Qwen / generic
``*DecoderLayer`` classes that expose the standard
``input_layernorm`` / ``post_attention_layernorm`` / ``self_attn`` /
``mlp`` interface). The same module's original forward is restored on
context-manager exit.
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import Iterable

import torch


def _is_decoder_layer(module) -> bool:
    """Conservative check: the module exposes the four attributes a
    standard pre-norm decoder block uses (and so admits a post-norm
    rewrite). We also gate on a class-name suffix to avoid patching
    e.g. an embedding container that happens to have similarly-named
    children."""
    name = type(module).__name__
    if not name.endswith("DecoderLayer"):
        return False
    needed = (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "mlp",
    )
    return all(hasattr(module, attr) for attr in needed)


def _post_norm_forward_phi3(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    past_key_values=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Phi-3 post-norm rewrite.

    Layout matches transformers ``Phi3DecoderLayer.forward`` exactly,
    with the two normalisations relocated to the *output* side of each
    residual sum. The attention call signature, the resid-dropout
    placement, and the return-type (single tensor) are preserved.
    """
    residual = hidden_states
    attn_out, self_attn_weights, present_key_value = _call_self_attn(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        extra_kwargs=kwargs,
    )
    hidden_states = self.input_layernorm(
        residual + self.resid_attn_dropout(attn_out)
    )

    residual = hidden_states
    mlp_out = self.mlp(hidden_states)
    hidden_states = self.post_attention_layernorm(
        residual + self.resid_mlp_dropout(mlp_out)
    )
    return _format_layer_output(
        self, hidden_states, self_attn_weights, present_key_value,
        output_attentions, use_cache,
    )


def _post_norm_forward_llama_style(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    past_key_values=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """LLaMA / Qwen post-norm rewrite.

    Matches the transformers ``LlamaDecoderLayer.forward`` signature and
    return type (single tensor). The LLaMA-style block lacks the resid
    dropouts that Phi-3 has; the cache is mutated through
    ``past_key_values`` in-place, never returned explicitly. Attention
    weights are discarded.
    """
    residual = hidden_states
    attn_out, self_attn_weights, present_key_value = _call_self_attn(
        self,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        extra_kwargs=kwargs,
    )
    hidden_states = self.input_layernorm(residual + attn_out)

    residual = hidden_states
    mlp_out = self.mlp(hidden_states)
    hidden_states = self.post_attention_layernorm(residual + mlp_out)
    return _format_layer_output(
        self, hidden_states, self_attn_weights, present_key_value,
        output_attentions, use_cache,
    )


def _call_self_attn(
    layer,
    *,
    hidden_states,
    attention_mask,
    position_ids,
    past_key_value,
    past_key_values,
    output_attentions,
    use_cache,
    cache_position,
    position_embeddings,
    extra_kwargs: dict,
):
    """Call a HF attention module while matching its exact kwarg names.

    Transformers has used both ``past_key_value`` and ``past_key_values``
    across model families / versions. The post-norm intervention should not
    make that API choice; it reflects the attention module's signature.
    """
    sig = inspect.signature(layer.self_attn.forward)
    params = sig.parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    cache = past_key_value if past_key_value is not None else past_key_values
    candidates = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_value": cache,
        "past_key_values": cache,
        "output_attentions": output_attentions,
        "use_cache": use_cache,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
    }
    call_kwargs = {
        k: v for k, v in candidates.items()
        if k in params and not (v is None and k not in {"attention_mask"})
    }
    for k, v in extra_kwargs.items():
        if k not in call_kwargs and (k in params or accepts_kwargs):
            call_kwargs[k] = v

    out = layer.self_attn(**call_kwargs)
    if isinstance(out, tuple):
        attn_out = out[0]
        self_attn_weights = out[1] if len(out) > 1 else None
        present_key_value = out[2] if len(out) > 2 else None
    else:
        attn_out = out
        self_attn_weights = None
        present_key_value = None
    return attn_out, self_attn_weights, present_key_value


def _format_layer_output(
    layer, hidden_states, self_attn_weights, present_key_value,
    output_attentions, use_cache,
):
    """Return in the same broad style as the original decoder layer."""
    if getattr(layer, "_dart_pagedkv_return_tuple", False):
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
    return hidden_states


def _select_post_norm_forward(module):
    """Pick the right post-norm rewrite for a given decoder-layer class.

    Phi-3 has the resid-dropout members; LLaMA / Qwen do not. We branch
    on those to choose the matching forward."""
    if hasattr(module, "resid_attn_dropout") and hasattr(module, "resid_mlp_dropout"):
        return _post_norm_forward_phi3
    return _post_norm_forward_llama_style


def _original_forward_returns_tuple(module) -> bool:
    """Best-effort detector for decoder-layer return convention."""
    try:
        source = inspect.getsource(module.forward)
    except (OSError, TypeError):
        source = ""
    if "return outputs" in source or "(hidden_states,)" in source:
        return True
    module_name = type(module).__module__.lower()
    class_name = type(module).__name__
    return (
        "transformers" in module_name
        and class_name in {
            "Phi3DecoderLayer",
            "LlamaDecoderLayer",
            "Qwen2DecoderLayer",
            "Qwen3DecoderLayer",
        }
    )


def install_post_norm_hooks(model) -> list:
    """Monkey-patch every standard pre-norm decoder block in ``model`` to
    use post-norm forward. Returns a list of ``(module, original_forward)``
    tuples; pass that to :func:`uninstall_post_norm_hooks` to revert.

    Raises if no recognised decoder layer is found.
    """
    patches: list = []
    n_phi = 0
    n_llama_style = 0
    for _, module in model.named_modules():
        if not _is_decoder_layer(module):
            continue
        new_forward_fn = _select_post_norm_forward(module)
        patches.append((module, module.forward))
        module._dart_pagedkv_return_tuple = _original_forward_returns_tuple(module)
        module.forward = new_forward_fn.__get__(module, type(module))
        if new_forward_fn is _post_norm_forward_phi3:
            n_phi += 1
        else:
            n_llama_style += 1
    if not patches:
        raise ValueError(
            f"layernorm-intervention: no standard pre-norm decoder layer "
            f"found on {model.__class__.__name__}; expected modules with "
            "class name ending in 'DecoderLayer' and the standard "
            "input_layernorm / post_attention_layernorm / self_attn / mlp "
            "attributes"
        )
    print(
        f"[layernorm-intervention] patched {n_phi} Phi3-style + "
        f"{n_llama_style} LLaMA-style decoder layers to post-norm forward",
        flush=True,
    )
    return patches


def uninstall_post_norm_hooks(patches: Iterable) -> None:
    """Restore each module's original forward."""
    for module, original_forward in patches:
        module.forward = original_forward


@contextmanager
def temporary_post_norm(model, enabled: bool = True):
    """Context manager: temporarily monkey-patch the model to post-norm."""
    if not enabled:
        yield
        return
    patches = install_post_norm_hooks(model)
    try:
        yield
    finally:
        uninstall_post_norm_hooks(patches)
