"""Effective-context support — the saturation predictor.

The lead hypothesis of the effective-context-saturation spec
(`docs/superpowers/specs/2026-05-17-effective-context-saturation.md` §5–§6):
layer-budget allocation stops mattering once the uniform per-group budget
exceeds a group's *effective-context support* — the number of tokens the group
actually draws attention mass from.

This module estimates that support from **full-KV** (uncompressed) attention
distributions. The non-circularity rule is load-bearing: support must never be
measured on the compressed/masked attention nor on the SnapKV priority that
drives compression — only on the attention the model computes when nothing is
masked. This module is pure NumPy and takes attention distributions as input;
recomputing them from Q/K is the caller's job (windowed_priority).

Three estimators, each with closed-form values on canonical distributions
(used as the spec's "known-distribution check"):

- ``mass_coverage`` — smallest ``k`` whose top-``k`` mass reaches ``tau``.
- ``participation_ratio`` — ``1 / sum(p**2)``.
- ``entropy`` — ``exp(H(p))``.

On a one-hot distribution all three give 1; on a uniform-over-``k``
distribution ``participation_ratio`` and ``entropy`` give ``k`` and
``mass_coverage`` gives ``ceil(tau * k)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ESTIMATORS = ("mass_coverage", "participation_ratio", "entropy")

# Tolerance for the mass-coverage cumulative-sum threshold. Attention inputs
# are float32 (rounding ~1e-7) and are renormalised before the cumsum, so a
# top element worth exactly ``tau`` can read as just-under. Compare against
# ``tau - _MASS_EPS`` with a tolerance safely above float32 precision; 1e-6 is
# far below any meaningful mass gap, so the integer support count is unaffected.
_MASS_EPS = 1e-6


def _normalised(p) -> np.ndarray:
    """Validate and row-normalise an attention array (last axis = keys)."""
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError("attention array must have at least 1 dimension")
    if not np.isfinite(arr).all():
        raise ValueError("attention array contains non-finite values")
    if (arr < 0).any():
        raise ValueError("attention array contains negative values")
    sums = arr.sum(axis=-1, keepdims=True)
    if (sums <= 0).any():
        raise ValueError("attention array has a row that sums to zero")
    return arr / sums


def distribution_support(
    p, *, estimator: str = "mass_coverage", tau: float = 0.95
) -> np.ndarray:
    """Effective support along the last axis.

    ``p`` may have any number of leading axes (e.g. ``[head, query_row, T]``);
    the result has shape ``p.shape[:-1]``. The input need not be normalised —
    it is row-normalised first. ``tau`` is used only by ``mass_coverage``.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}"
        )
    arr = _normalised(p)

    if estimator == "participation_ratio":
        return 1.0 / np.sum(arr * arr, axis=-1)

    if estimator == "entropy":
        # x*log(x) -> 0 as x -> 0; evaluate log only where p > 0.
        safe = np.where(arr > 0, arr, 1.0)
        entropy = -np.sum(np.where(arr > 0, arr * np.log(safe), 0.0), axis=-1)
        return np.exp(entropy)

    # mass_coverage
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1] for mass_coverage, got {tau}")
    descending = -np.sort(-arr, axis=-1)
    cumulative = np.cumsum(descending, axis=-1)
    reached = cumulative >= tau - _MASS_EPS
    first_index = np.argmax(reached, axis=-1)
    keys = arr.shape[-1]
    # argmax returns 0 when nothing is reached; guard with the any() mask.
    return np.where(reached.any(axis=-1), first_index + 1, keys).astype(np.float64)


def block_support(
    attention_block, *, estimator: str = "mass_coverage", tau: float = 0.95
) -> tuple[float, float]:
    """``(mean, max)`` support over every row of a windowed attention block.

    ``attention_block`` is ``[..., T]`` — any leading axes (head, query row)
    are flattened together; the mean is the typical support and the max is the
    bottleneck row (the spec's "a group is unsaturated if *any* tracked query
    needs more than the budget").
    """
    supports = distribution_support(attention_block, estimator=estimator, tau=tau)
    return float(np.mean(supports)), float(np.max(supports))


@dataclass(frozen=True)
class GroupSupport:
    """Per-layer-group effective support (`effective_support` output)."""

    per_group_mean: list[float]
    per_group_max: list[float]
    estimator: str
    tau: float


def group_support_from_layers(
    layer_mean: dict[int, float],
    layer_max: dict[int, float],
    groups: list[list[int]],
    *,
    estimator: str = "mass_coverage",
    tau: float = 0.95,
) -> GroupSupport:
    """Aggregate pre-computed per-layer support stats into per-group support.

    For callers that compute `block_support` per layer in a streaming loop —
    a 131k run must not hold every layer's ``[H, w, T]`` attention block in
    memory at once. ``layer_mean`` / ``layer_max`` map a layer index to its
    mean / bottleneck support. `effective_support` is the all-in-one
    equivalent when the attention blocks are already in hand.
    """
    if not groups:
        raise ValueError("groups must be a non-empty layer partition")

    per_group_mean: list[float] = []
    per_group_max: list[float] = []
    for group in groups:
        if not group:
            raise ValueError("each group must contain at least one layer")
        for layer in group:
            if layer not in layer_mean or layer not in layer_max:
                raise ValueError(
                    f"group references layer {layer} absent from per-layer "
                    f"support stats (have {sorted(layer_mean)})"
                )
        per_group_mean.append(
            float(np.mean([layer_mean[layer] for layer in group]))
        )
        per_group_max.append(
            float(np.max([layer_max[layer] for layer in group]))
        )

    return GroupSupport(
        per_group_mean=per_group_mean,
        per_group_max=per_group_max,
        estimator=estimator,
        tau=float(tau),
    )


def effective_support(
    per_layer_attention: dict[int, np.ndarray],
    groups: list[list[int]],
    *,
    estimator: str = "mass_coverage",
    tau: float = 0.95,
) -> GroupSupport:
    """Per-group effective support from per-layer full-KV attention blocks.

    ``per_layer_attention`` maps a layer index to its windowed attention block
    ``[..., T]``. ``groups`` is a contiguous layer partition (see
    `layer_groups.make_layer_groups`). Aggregation follows spec §6: support is
    computed per attention row, meaned over rows within a layer, then meaned
    over layers within a group; ``per_group_max`` takes the bottleneck (max)
    instead of the mean at both stages.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(
            f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}"
        )
    if not groups:
        raise ValueError("groups must be a non-empty layer partition")

    layer_mean: dict[int, float] = {}
    layer_max: dict[int, float] = {}
    for group in groups:
        for layer in group:
            if layer in layer_mean:
                continue
            if layer not in per_layer_attention:
                raise ValueError(
                    f"group references layer {layer} absent from "
                    f"per_layer_attention (have {sorted(per_layer_attention)})"
                )
            layer_mean[layer], layer_max[layer] = block_support(
                per_layer_attention[layer], estimator=estimator, tau=tau
            )
    return group_support_from_layers(
        layer_mean, layer_max, groups, estimator=estimator, tau=tau
    )
