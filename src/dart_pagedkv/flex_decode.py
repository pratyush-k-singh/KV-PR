"""Production-faithful cached decode for the FlexAttention long-context path.

The sub-8k decode-probe (`decode_probe.py`) runs ``use_cache=False`` full
re-forwards each step — O(T²) compute, and it masks the prompt's *own*
recomputation, which a real paged cache does not do.

This module implements the corrected, long-context-scalable semantics
(spec 2026-05-16-flexattention-longcontext.md §5):

  * **Prefill once, full attention** → a KV cache holding the prompt K/V
    computed normally. The cold KV physically exists (recoverable, just
    not in the fast path) — prefill is NOT masked.
  * **Decode incrementally**: each step forwards one token; only the
    decode-time *access* into the cached prompt K/V is masked (cold
    prompt keys blocked, decoded tokens always hot), via a per-step
    FlexAttention ``BlockMask``.

One cache per prompt, cropped back to the prefill length between budget
runs. Step 0 (predicting the first decoded token) is the shared full
prefill logit, so its masked logit-KL is 0 by construction.

KL convention: ``token_kl_divergence(full, masked)`` = KL(full ‖ masked),
the documented direction (reference distribution first). Note
`decode_probe._masked_step_kl` passes the arguments reversed — a
pre-existing inconsistency; moot for cross-comparison because the
long-context results are not number-comparable to the sub-8k gates
(spec §5).
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Sequence

import torch
import torch._dynamo

from .flex_mask import build_group_block_masks, install_per_layer_block_mask_hooks
from .logit_kl import install_per_layer_mask_hooks, token_kl_divergence
from src.trace.collect import _find_attention_modules


# FlexAttention's compiled kernel is guarded on the key tensor's sequence
# length. Cached incremental decode grows the cache by one key per step
# (kv_len = prefill_len + t), and each prompt carries its own prefill
# length, so even a single k_dec-step decode produces k_dec-1 distinct
# shapes — far past Dynamo's default cache_size_limit of 8. Once that
# limit trips Dynamo abandons the frame and runs flex_attention *eager*
# for the rest of the process (often 10-50x slower). Raising the limits
# keeps every shape compiled and, crucially, lets the compiled kernels be
# reused across the many budget trajectories that share a prompt's
# prefill length. Guarded so an explicit higher setting is never lowered.
_FLEX_CACHE_SIZE_LIMIT = 1024
if torch._dynamo.config.cache_size_limit < _FLEX_CACHE_SIZE_LIMIT:
    torch._dynamo.config.cache_size_limit = _FLEX_CACHE_SIZE_LIMIT
if torch._dynamo.config.accumulated_cache_size_limit < 4 * _FLEX_CACHE_SIZE_LIMIT:
    torch._dynamo.config.accumulated_cache_size_limit = 4 * _FLEX_CACHE_SIZE_LIMIT

# Compile flex_attention for *static* shapes, never symbolic. Dynamo's
# automatic dynamic-shape promotion turns the KV-length dimension
# symbolic after the first couple of recompiles; the resulting
# dynamic-shape Inductor pointwise kernels then hit a Triton autotuning
# "misaligned address" crash on H100 + this torch/triton stack —
# alignment cannot be proven for a symbolic size, so a vectorised
# autotune config issues a misaligned load. Static compilation gives
# Inductor concrete sizes (provably aligned), and the raised
# cache_size_limit above is exactly what makes one-compile-per-shape
# affordable.
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.assume_static_by_default = True


def decode_step_params(prefill_len: int, step: int) -> tuple[int, int]:
    """Mask parameters for decode ``step`` (1-indexed; step 0 is the prefill).

    At step ``t`` the forward processes the token at absolute position
    ``prefill_len + t - 1``; after it the cache holds ``prefill_len + t``
    keys. Returns ``(kv_len, q_start)``.
    """
    return prefill_len + step, prefill_len + step - 1


def current_attn_implementation(model) -> str | None:
    """Return the model's active HF attention backend, when exposed."""
    config = getattr(model, "config", None)
    return getattr(config, "_attn_implementation", None)


def set_attn_implementation(model, implementation: str | None) -> None:
    """Set the HF attention backend for models that support runtime swaps."""
    setter = getattr(model, "set_attn_implementation", None)
    if setter is not None:
        setter(implementation)
        return
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError(
            "Cannot switch attention implementation: model has no config "
            "or set_attn_implementation method."
        )
    config._attn_implementation = implementation


def _cache_with_crop(past_key_values):
    """Return a cache object that supports ``crop`` when HF gives a legacy tuple."""
    if hasattr(past_key_values, "crop"):
        return past_key_values
    if isinstance(past_key_values, tuple):
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return past_key_values
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(past_key_values)
        cache = DynamicCache()
        for layer_idx, layer_past in enumerate(past_key_values):
            if layer_past is None:
                continue
            key_states, value_states = layer_past[:2]
            cache.update(key_states, value_states, layer_idx=layer_idx)
        return cache
    return past_key_values


def build_group_decode_additive_masks(
    hot_per_group: Sequence[Sequence[int]],
    kv_len: int,
    key_boundary: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[int, torch.Tensor]:
    """Per-group additive masks for one cached decode query.

    Unlike the old eager full-forward probe, cached decode has ``q_len=1``.
    The additive mask is therefore ``[1, 1, 1, kv_len]`` rather than
    ``[1, 1, T, T]``: O(T) memory and usable for models such as Phi-3 whose
    HF implementation supports eager attention but not SDPA/FlexAttention.
    """
    masks: dict[int, torch.Tensor] = {}
    prompt_len = min(int(key_boundary), int(kv_len))
    for group_idx, hot in enumerate(hot_per_group):
        hot_set = {int(h) for h in hot}
        if prompt_len > 0 and 0 not in hot_set:
            raise ValueError(
                f"group {group_idx}: key 0 (the attention sink) is cold. "
                f"The decode query could see an all-masked prefix if decoded "
                f"tokens are absent. Keep each group's protected sink positions hot."
            )
        mask = torch.zeros(1, 1, 1, int(kv_len), dtype=dtype, device=device)
        cold = [j for j in range(prompt_len) if j not in hot_set]
        if cold:
            cold_idx = torch.tensor(cold, dtype=torch.long, device=device)
            mask[:, :, :, cold_idx] = float("-inf")
        masks[group_idx] = mask
    return masks


def build_group_decode_binary_masks(
    hot_per_group: Sequence[Sequence[int]],
    kv_len: int,
    key_boundary: int,
    *,
    dtype: torch.dtype = torch.long,
    device: torch.device | str = "cpu",
) -> dict[int, torch.Tensor]:
    """Per-group 2D key masks for flash-attention cached decode.

    Phi-3's flash-attention implementation accepts a ``[batch, kv_len]``
    padding-style mask where 1 means keep and 0 means drop. With cached
    ``q_len=1`` decode, this is enough to express our hot/cold prompt-key
    policy: cold prompt positions become 0, decoded/current positions remain 1.
    """
    masks: dict[int, torch.Tensor] = {}
    prompt_len = min(int(key_boundary), int(kv_len))
    for group_idx, hot in enumerate(hot_per_group):
        hot_set = {int(h) for h in hot}
        if prompt_len > 0 and 0 not in hot_set:
            raise ValueError(
                f"group {group_idx}: key 0 (the attention sink) is cold. "
                f"The decode query could see an all-masked prefix if decoded "
                f"tokens are absent. Keep each group's protected sink positions hot."
            )
        mask = torch.ones(1, int(kv_len), dtype=dtype, device=device)
        cold = [j for j in range(prompt_len) if j not in hot_set]
        if cold:
            cold_idx = torch.tensor(cold, dtype=torch.long, device=device)
            mask[:, cold_idx] = 0
        masks[group_idx] = mask
    return masks


@contextmanager
def temporary_attn_implementation(model, implementation: str | None):
    """Temporarily switch a HF model's attention backend."""
    if not implementation:
        yield
        return
    previous = current_attn_implementation(model)
    if previous == implementation:
        yield
        return
    set_attn_implementation(model, implementation)
    try:
        yield
    finally:
        set_attn_implementation(model, previous)


@contextmanager
def temporary_phi3_eager_attention(model, enabled: bool = True):
    """Temporarily route Phi-3 flash-attention modules through eager forward.

    Phi-3.5 remote code constructs attention modules from
    ``config._attn_implementation`` at model initialisation. Loading with
    ``flash_attention_2`` is required for long-context prefill, but changing
    the config later does not replace those module instances. For masked
    cached decode we need eager semantics because the hot/cold mask is an
    additive ``[1, 1, 1, kv_len]`` tensor. Phi's flash module subclasses the
    eager ``Phi3Attention`` class, so temporarily binding the base-class
    ``forward`` gives the desired decode path without touching weights.
    """
    if not enabled:
        yield
        return
    patches = []
    for _, module in _find_attention_modules(model):
        eager_forward = None
        for base in type(module).__mro__[1:]:
            if base.__name__ == "Phi3Attention" and hasattr(base, "forward"):
                eager_forward = base.forward
                break
        if eager_forward is None:
            continue
        patches.append((module, module.forward))
        module.forward = eager_forward.__get__(module, type(module))
    try:
        yield
    finally:
        for module, forward in patches:
            module.forward = forward


class FlexCachedDecodeSession:
    """Holds one prompt's prefill cache; serves reference + masked decodes.

    The prefill (full attention) is computed once in ``__init__``. Each
    ``full_kv_reference`` / ``masked_decode`` call decodes from the shared
    cache and crops it back to the prefill length afterward, so all budget
    runs share the single expensive prefill.
    """

    def __init__(
        self,
        model,
        prompt_ids: torch.Tensor,
        layer_to_group: dict[int, int],
        *,
        device: torch.device | str = "cuda",
        masked_attn_implementation: str | None = None,
    ):
        self.model = model
        self.prompt_ids = prompt_ids
        self.layer_to_group = layer_to_group
        self.device = device
        self.masked_attn_implementation = masked_attn_implementation
        self.prefill_len = int(prompt_ids.shape[1])
        with torch.inference_mode():
            base_model = getattr(model, "model", None)
            lm_head = getattr(model, "lm_head", None)
            if callable(base_model) and lm_head is not None:
                out = base_model(
                    input_ids=prompt_ids, use_cache=True, return_dict=True
                )
                last_hidden = out.last_hidden_state[:, -1:, :]
                prefill_last_logits = lm_head(last_hidden)[0, -1, :]
            else:
                out = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
                prefill_last_logits = out.logits[0, -1, :]
        self.cache = _cache_with_crop(out.past_key_values)
        self._prefill_last_logits = prefill_last_logits.float().detach().cpu()

    def _reset_cache(self) -> None:
        self.cache.crop(self.prefill_len)

    def _decode_token(
        self,
        token_id: int,
        *,
        block_masks=None,
        additive_masks=None,
    ) -> torch.Tensor:
        """Forward one token against the cache; return last-position logits (CPU)."""
        tok = torch.tensor(
            [[int(token_id)]],
            dtype=self.prompt_ids.dtype,
            device=self.prompt_ids.device,
        )
        handles = []
        if block_masks is not None and additive_masks is not None:
            raise ValueError("Pass either block_masks or additive_masks, not both.")
        if block_masks is not None:
            handles = install_per_layer_block_mask_hooks(
                self.model, block_masks, self.layer_to_group
            )
        elif additive_masks is not None:
            handles = install_per_layer_mask_hooks(
                self.model, additive_masks, self.layer_to_group
            )
        try:
            with torch.inference_mode():
                out = self.model(
                    input_ids=tok, past_key_values=self.cache,
                    use_cache=True, return_dict=True,
                )
        finally:
            for h in handles:
                h.remove()
        self.cache = out.past_key_values
        return out.logits[0, -1, :].float().detach().cpu()

    def full_kv_reference(self, k_dec: int) -> tuple[list[int], torch.Tensor]:
        """Greedy-decode ``k_dec`` tokens under full attention.

        Returns ``(gold_continuation, ref_logits)`` with ``ref_logits``
        shape ``[k_dec, vocab]`` on CPU. ``ref_logits[0]`` is the shared
        full prefill logit.
        """
        gold = [int(self._prefill_last_logits.argmax())]
        ref = [self._prefill_last_logits]
        use_phi3_eager = self.masked_attn_implementation == "eager"
        with temporary_attn_implementation(self.model, self.masked_attn_implementation):
            with temporary_phi3_eager_attention(self.model, enabled=use_phi3_eager):
                for _ in range(1, k_dec):
                    logits = self._decode_token(gold[-1], block_masks=None)
                    ref.append(logits)
                    gold.append(int(logits.argmax()))
        self._reset_cache()
        return gold, torch.stack(ref)

    def masked_decode(
        self,
        hot_per_group: Sequence[Sequence[int]],
        gold_continuation: Sequence[int],
        ref_logits: torch.Tensor,
        *,
        decode_temperature: float = 1.0,
        return_query_norms: bool = False,
    ) -> list[float] | tuple[list[float], list[float]]:
        """Teacher-forced masked decode; per-step logit-KL vs ``ref_logits``.

        Step 0 is the shared full prefill prediction → KL 0. Steps
        ``1..k_dec-1`` are masked incremental forwards: the decode query
        attends the cached prompt K/V through this budget's hot/cold
        ``BlockMask`` (decoded tokens always hot).

        Single-token (``q_len = 1``) decode is deliberate: the
        single-forward variant (``q_len = k_dec-1``) crashes Inductor's
        ``flex_decoding`` template with a Triton ``misaligned address``
        on head_dim-128 models on this torch/triton stack.

        ``decode_temperature`` (P15 mechanism probe, default 1.0):
        divides BOTH ref and masked logits by τ before softmax. τ > 1
        flattens the decoder's output distribution (contracts the
        masked-decode Jacobian); τ < 1 sharpens it. The KL is computed
        on the temperature-modified distributions.

        ``return_query_norms`` (P11 Jacobian probe, default False):
        when True, additionally returns a list of per-step L2 norms
        ``||logits_masked[t] - logits_ref[t]||_2``. This is the proxy
        for the masked-decode Jacobian's effect on the output: the
        ratio of consecutive norms estimates the operator norm of the
        decoder's masked-to-reference deviation map. Returns
        ``(kls, query_norms)`` tuple in that case; otherwise ``kls``.
        """
        if decode_temperature <= 0:
            raise ValueError(
                f"decode_temperature must be > 0, got {decode_temperature}"
            )
        k_dec = len(gold_continuation)
        kls = [0.0]
        query_norms: list[float] = [0.0]
        use_phi3_eager = self.masked_attn_implementation == "eager"
        with temporary_attn_implementation(self.model, self.masked_attn_implementation):
            with temporary_phi3_eager_attention(self.model, enabled=use_phi3_eager):
                for t in range(1, k_dec):
                    kv_len, q_start = decode_step_params(self.prefill_len, t)
                    if self.masked_attn_implementation == "eager":
                        additive_masks = build_group_decode_additive_masks(
                            hot_per_group,
                            kv_len=kv_len,
                            key_boundary=self.prefill_len,
                            device=self.device,
                        )
                        logits = self._decode_token(
                            gold_continuation[t - 1], additive_masks=additive_masks
                        )
                    elif self.masked_attn_implementation == "flash_attention_2":
                        binary_masks = build_group_decode_binary_masks(
                            hot_per_group,
                            kv_len=kv_len,
                            key_boundary=self.prefill_len,
                            device=self.device,
                        )
                        logits = self._decode_token(
                            gold_continuation[t - 1], additive_masks=binary_masks
                        )
                    else:
                        block_masks = build_group_block_masks(
                            hot_per_group, kv_len=kv_len, key_boundary=self.prefill_len,
                            q_len=1, q_start=q_start, device=self.device,
                        )
                        logits = self._decode_token(
                            gold_continuation[t - 1], block_masks=block_masks
                        )
                    ref_step = ref_logits[t].unsqueeze(0)
                    masked_step = logits.unsqueeze(0)
                    if decode_temperature != 1.0:
                        ref_step = ref_step / float(decode_temperature)
                        masked_step = masked_step / float(decode_temperature)
                    kls.append(token_kl_divergence(ref_step, masked_step))
                    if return_query_norms:
                        # L2 norm of the per-step logit deviation as the
                        # P11 Jacobian proxy.
                        diff = (
                            ref_logits[t].float() - logits.float()
                        )
                        query_norms.append(float(diff.norm().item()))
        self._reset_cache()
        if return_query_norms:
            return kls, query_norms
        return kls

    def masked_decode_autoregressive(
        self,
        hot_per_group: Sequence[Sequence[int]],
        k_gen: int,
    ) -> list[int]:
        """Greedy-decode ``k_gen`` tokens under masked KV access.

        Unlike :meth:`masked_decode` (which is teacher-forced for the KL
        audit), this routes the argmax of each masked step's logits back
        as the next input -- the actual user-facing generation under a
        compressed cache. Used by the P20b / E end-to-end-accuracy probe:
        compare these tokens against ``full_kv_reference``'s tokens to
        get a reference-fidelity match rate per priority.

        Single-token (``q_len = 1``) decode for the same Inductor /
        Triton reasons as :meth:`masked_decode`.
        """
        # The prefill prediction is shared with full-KV (cache state =
        # prefill, no masking applied at the prefill step), so step 0 is
        # identical. We seed the generation with the prefill argmax and
        # then start masked decoding from step 1.
        generated = [int(self._prefill_last_logits.argmax())]
        use_phi3_eager = self.masked_attn_implementation == "eager"
        with temporary_attn_implementation(self.model, self.masked_attn_implementation):
            with temporary_phi3_eager_attention(self.model, enabled=use_phi3_eager):
                for t in range(1, k_gen):
                    kv_len, q_start = decode_step_params(self.prefill_len, t)
                    if self.masked_attn_implementation == "eager":
                        masks = build_group_decode_additive_masks(
                            hot_per_group, kv_len=kv_len,
                            key_boundary=self.prefill_len, device=self.device,
                        )
                        logits = self._decode_token(
                            generated[-1], additive_masks=masks,
                        )
                    elif self.masked_attn_implementation == "flash_attention_2":
                        masks = build_group_decode_binary_masks(
                            hot_per_group, kv_len=kv_len,
                            key_boundary=self.prefill_len, device=self.device,
                        )
                        logits = self._decode_token(
                            generated[-1], additive_masks=masks,
                        )
                    else:
                        block_masks = build_group_block_masks(
                            hot_per_group, kv_len=kv_len,
                            key_boundary=self.prefill_len,
                            q_len=1, q_start=q_start, device=self.device,
                        )
                        logits = self._decode_token(
                            generated[-1], block_masks=block_masks,
                        )
                    generated.append(int(logits.argmax()))
        self._reset_cache()
        return generated
