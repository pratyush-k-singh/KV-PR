"""Direct measurement of the masked-decode operator norm rho_R (P20).

The existing rho-hat_R proxy in `flex_decode.FlexCachedDecodeSession.masked_decode`
records per-step `||logits_masked - logits_ref||_2` and takes the geometric
mean of consecutive ratios. The proxy is a one-trajectory empirical
estimate; reviewers can object that a deviation-ratio is not the same as
the operator's spectral norm.

This module measures rho_R *directly* at chosen masked-decode steps by:

  1. Snapshotting the per-layer (K_t, V_t) contributions that step t
     appended to the cache.
  2. Defining the perturbation map
        v in R^d  ->  logits at step t+1 with cache last-position
                      replaced by baseline + eps * reshape(v).
     Here d = sum over layers of (K_size + V_size) = the dimension of
     step t's contribution to the cache.
  3. Power-iterating on J^T J via central-difference JVPs and autograd
     VJPs to converge on the top singular vector, then reporting
     ||J v_top|| as rho_R at step t.

Step t's contribution is the *only* thing step t+1 sees that wasn't
already in the cache before step t, so this captures exactly the
compounding factor the proxy approximates. The teacher-forced decode is
preserved (gold_continuation[t] is still fed as step t+1's input);
only the cache contents are perturbed.

Per-step cost (n_iter power iterations): one cached step t+1 plus
2 * n_iter forward calls (for the central-difference JVPs) plus
n_iter backward passes (for the VJPs). Wall-time roughly 30-40x a
single masked-decode step. Measuring at K_meas <= K-1 chosen steps
gives total overhead ~K_meas * 30x the normal masked decode.

Use `CachePerturbationJacobian(session).top_sigma_at_step(...)` for a
single step; `compute_per_step_sigmas(...)` for a list of steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from .flex_decode import (
    FlexCachedDecodeSession,
    build_group_decode_additive_masks,
    build_group_decode_binary_masks,
    decode_step_params,
    temporary_attn_implementation,
    temporary_phi3_eager_attention,
)
from .flex_mask import build_group_block_masks, install_per_layer_block_mask_hooks
from .logit_kl import install_per_layer_mask_hooks
from .power_iteration import power_iterate_top_sigma


def _cache_layer_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (K, V) for a layer from a HF cache, agnostic to its internal layout."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    raise RuntimeError(
        f"Cache type {type(cache).__name__} does not expose per-layer K/V "
        "via key_cache/value_cache or layers[].keys/values"
    )


def _set_cache_layer_last_pos(
    cache, layer_idx: int, K_new_last: torch.Tensor, V_new_last: torch.Tensor
) -> None:
    """Replace the last sequence position of layer ``layer_idx``'s K/V with
    a new differentiable slice, preserving the gradient lineage of
    ``K_new_last`` / ``V_new_last``.

    Uses ``torch.cat`` so the slice's autograd lineage is retained
    (in-place ``t[..., -1:, :] = v`` uses ``copy_`` which detaches).
    """
    K_full, V_full = _cache_layer_kv(cache, layer_idx)
    K_new = torch.cat([K_full[..., :-1, :], K_new_last], dim=-2)
    V_new = torch.cat([V_full[..., :-1, :], V_new_last], dim=-2)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache[layer_idx] = K_new
        cache.value_cache[layer_idx] = V_new
    elif hasattr(cache, "layers"):
        cache.layers[layer_idx].keys = K_new
        cache.layers[layer_idx].values = V_new


class CachePerturbationJacobian:
    """Direct rho_R via cache-perturbation power iteration on a session.

    The session must have a fully populated cache from a completed
    masked-decode run; the caller is responsible for resetting the
    session's cache state between Jacobian queries.
    """

    def __init__(
        self,
        session: FlexCachedDecodeSession,
        *,
        eps: float = 0.05,
        n_iter: int = 10,
        eps_floor: float = 1e-12,
    ):
        self.session = session
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.eps_floor = float(eps_floor)

    def _baseline_kv_at_pos(
        self, cache, pos: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Snapshot the (K, V) at the last position (= pos-1) per layer."""
        n_layers = self._count_layers(cache)
        baseline: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(n_layers):
            K_full, V_full = _cache_layer_kv(cache, layer)
            if K_full.shape[-2] != pos:
                raise RuntimeError(
                    f"Cache layer {layer} has length {K_full.shape[-2]} but "
                    f"expected {pos} (cache state at step t -> pos = "
                    "prefill_len + step_t)"
                )
            K_last = K_full[..., -1:, :].detach().clone()
            V_last = V_full[..., -1:, :].detach().clone()
            baseline.append((K_last, V_last))
        return baseline

    def _count_layers(self, cache) -> int:
        if hasattr(cache, "key_cache"):
            return len(cache.key_cache)
        if hasattr(cache, "layers"):
            return len(cache.layers)
        raise RuntimeError("Cannot count cache layers")

    def _restore_kv_at_pos(
        self, cache, pos: int,
        baseline: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Restore the cache's last position to the baseline (no grad)."""
        for layer, (K_last, V_last) in enumerate(baseline):
            _set_cache_layer_last_pos(cache, layer, K_last, V_last)

    @staticmethod
    def _perturbation_shapes(
        baseline: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list[tuple[torch.Size, torch.Size]], int]:
        shapes = [(K.shape, V.shape) for K, V in baseline]
        total = sum(K.numel() + V.numel() for K, V in baseline)
        return shapes, int(total)

    def _apply_perturbation(
        self, cache,
        baseline: list[tuple[torch.Tensor, torch.Tensor]],
        perturbation: torch.Tensor,
    ) -> None:
        """Write baseline + perturbation (sliced + reshaped per layer) into the
        cache's last position. Differentiable in ``perturbation``."""
        offset = 0
        for layer, (K_base, V_base) in enumerate(baseline):
            k_n = K_base.numel()
            v_n = V_base.numel()
            K_pert = perturbation[offset:offset + k_n].reshape(K_base.shape)
            offset += k_n
            V_pert = perturbation[offset:offset + v_n].reshape(V_base.shape)
            offset += v_n
            K_new = K_base + K_pert.to(K_base.dtype)
            V_new = V_base + V_pert.to(V_base.dtype)
            _set_cache_layer_last_pos(cache, layer, K_new, V_new)

    def _decode_one_step_with_grad(
        self,
        token_id: int,
        *,
        hot_per_group: Sequence[Sequence[int]],
        kv_len: int,
        q_start: int,
    ) -> torch.Tensor:
        """One masked decode step under autograd (no inference_mode); returns
        the last-position logits as a differentiable tensor on device."""
        sess = self.session
        tok = torch.tensor(
            [[int(token_id)]],
            dtype=sess.prompt_ids.dtype,
            device=sess.prompt_ids.device,
        )
        impl = sess.masked_attn_implementation
        handles = []
        if impl == "eager":
            additive_masks = build_group_decode_additive_masks(
                hot_per_group, kv_len=kv_len,
                key_boundary=sess.prefill_len, device=sess.device,
            )
            handles = install_per_layer_mask_hooks(
                sess.model, additive_masks, sess.layer_to_group
            )
            masks_kw: dict = {}
        elif impl == "flash_attention_2":
            binary_masks = build_group_decode_binary_masks(
                hot_per_group, kv_len=kv_len,
                key_boundary=sess.prefill_len, device=sess.device,
            )
            handles = install_per_layer_mask_hooks(
                sess.model, binary_masks, sess.layer_to_group
            )
            masks_kw = {}
        else:
            block_masks = build_group_block_masks(
                hot_per_group, kv_len=kv_len,
                key_boundary=sess.prefill_len,
                q_len=1, q_start=q_start, device=sess.device,
            )
            handles = install_per_layer_block_mask_hooks(
                sess.model, block_masks, sess.layer_to_group
            )
            masks_kw = {}
        try:
            out = sess.model(
                input_ids=tok, past_key_values=sess.cache,
                use_cache=True, return_dict=True, **masks_kw,
            )
        finally:
            for h in handles:
                h.remove()
        sess.cache = out.past_key_values
        return out.logits[0, -1, :].float()

    def top_sigma_at_step(
        self,
        step_t: int,
        gold_continuation: Sequence[int],
        hot_per_group: Sequence[Sequence[int]],
        *,
        rng: Optional[torch.Generator] = None,
    ) -> dict:
        """Compute top sigma of the operator
            perturbation in cache last-position (= step t's contribution)
            ->
            logits at step t+1.

        The session's cache must be in state "after step t" (length =
        prefill_len + step_t) at call time. After this method returns the
        cache is left in the same state (cropped back to prefill_len +
        step_t with baseline K/V restored).

        Returns dict with keys: top_sigma, jv_norm_history, baseline_logit_norm.
        """
        sess = self.session
        pos = sess.prefill_len + step_t
        seq_len_now = self._cache_seq_len(sess.cache)
        if seq_len_now != pos:
            raise RuntimeError(
                f"top_sigma_at_step called with cache at length {seq_len_now}, "
                f"expected {pos} (= prefill_len {sess.prefill_len} + step_t "
                f"{step_t}). Call masked_decode first and crop only as needed."
            )
        if step_t >= len(gold_continuation):
            raise ValueError(
                f"step_t {step_t} requires gold_continuation[{step_t}] for "
                "step t+1; got length {len(gold_continuation)}"
            )

        baseline = self._baseline_kv_at_pos(sess.cache, pos)
        shapes, total_dim = self._perturbation_shapes(baseline)
        device = baseline[0][0].device

        next_token = int(gold_continuation[step_t])
        kv_len_next, q_start_next = decode_step_params(sess.prefill_len, step_t + 1)

        use_phi3_eager = sess.masked_attn_implementation == "eager"

        def run_step_with_perturbation(perturbation: torch.Tensor) -> torch.Tensor:
            """Returns step t+1's logits with cache last-pos = baseline + perturbation.
            Restores cache to baseline state before returning."""
            with temporary_attn_implementation(sess.model, sess.masked_attn_implementation):
                with temporary_phi3_eager_attention(sess.model, enabled=use_phi3_eager):
                    self._apply_perturbation(sess.cache, baseline, perturbation)
                    logits = self._decode_one_step_with_grad(
                        next_token, hot_per_group=hot_per_group,
                        kv_len=kv_len_next, q_start=q_start_next,
                    )
                    # The forward appended K_{t+1}, V_{t+1} to the cache; crop
                    # back so subsequent perturbation calls start from the same
                    # cache length.
                    sess.cache.crop(pos)
                    # And restore the baseline values at position step_t.
                    self._restore_kv_at_pos(sess.cache, pos, baseline)
            return logits

        # Baseline logits (zero perturbation) for reporting.
        with torch.no_grad():
            zero = torch.zeros(total_dim, device=device, dtype=torch.float32)
            baseline_logits = run_step_with_perturbation(zero)
        baseline_norm = float(baseline_logits.norm().item())

        result = power_iterate_top_sigma(
            run_step_with_perturbation,
            total_dim=total_dim, device=device,
            n_iter=self.n_iter, eps=self.eps, eps_floor=self.eps_floor,
            rng=rng,
        )
        result["baseline_logit_norm"] = baseline_norm
        return result

    @staticmethod
    def _cache_seq_len(cache) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            return int(getter())
        if hasattr(cache, "key_cache") and cache.key_cache:
            return int(cache.key_cache[0].shape[-2])
        if hasattr(cache, "layers") and cache.layers:
            return int(cache.layers[0].keys.shape[-2])
        raise RuntimeError("Cannot determine cache sequence length")


def compute_per_step_sigmas(
    session: FlexCachedDecodeSession,
    hot_per_group: Sequence[Sequence[int]],
    gold_continuation: Sequence[int],
    *,
    measure_steps: Sequence[int],
    n_iter: int = 10,
    eps: float = 1e-3,
    rng: Optional[torch.Generator] = None,
) -> dict[int, dict]:
    """Re-run the masked decode to each requested step and measure top sigma.

    ``measure_steps`` must be sorted ascending and each value must be in
    ``[1, len(gold_continuation) - 2]`` so step t+1 exists.

    Caller owns the session and is responsible for resetting it
    (``session._reset_cache()``) after this returns.
    """
    measure_steps = sorted({int(s) for s in measure_steps})
    if not measure_steps:
        return {}
    k_dec = len(gold_continuation)
    for s in measure_steps:
        if not (1 <= s <= k_dec - 2):
            raise ValueError(
                f"measure_step {s} must be in [1, {k_dec - 2}] so step t+1 "
                f"is available (gold_continuation has length {k_dec})"
            )

    use_phi3_eager = session.masked_attn_implementation == "eager"
    jac = CachePerturbationJacobian(session, eps=eps, n_iter=n_iter)
    results: dict[int, dict] = {}

    # Run the masked decode one step at a time, computing sigma at each
    # measure_step before continuing.
    session._reset_cache()
    last_measured = 0
    for target in measure_steps:
        # Advance cache from current state up to length prefill_len + target.
        with torch.no_grad():
            with temporary_attn_implementation(
                session.model, session.masked_attn_implementation
            ):
                with temporary_phi3_eager_attention(
                    session.model, enabled=use_phi3_eager
                ):
                    for t in range(last_measured + 1, target + 1):
                        kv_len, q_start = decode_step_params(session.prefill_len, t)
                        impl = session.masked_attn_implementation
                        if impl == "eager":
                            masks = build_group_decode_additive_masks(
                                hot_per_group, kv_len=kv_len,
                                key_boundary=session.prefill_len, device=session.device,
                            )
                            session._decode_token(
                                gold_continuation[t - 1], additive_masks=masks
                            )
                        elif impl == "flash_attention_2":
                            masks = build_group_decode_binary_masks(
                                hot_per_group, kv_len=kv_len,
                                key_boundary=session.prefill_len, device=session.device,
                            )
                            session._decode_token(
                                gold_continuation[t - 1], additive_masks=masks
                            )
                        else:
                            block_masks = build_group_block_masks(
                                hot_per_group, kv_len=kv_len,
                                key_boundary=session.prefill_len,
                                q_len=1, q_start=q_start, device=session.device,
                            )
                            session._decode_token(
                                gold_continuation[t - 1], block_masks=block_masks
                            )
        # Cache is now in state "after step target".
        results[target] = jac.top_sigma_at_step(
            target, gold_continuation, hot_per_group, rng=rng,
        )
        last_measured = target

    session._reset_cache()
    return results
