"""Published-method allocator vectors — PyramidKV / inverse-pyramid / proportional.

Encodes published layer-budget allocation schemes as fixed budget vectors that
can be evaluated under the same audit testbed as uniform / mincost / single-
dominant. The revised audit thesis (status §20–§21): cheap cost proxies are
blind to a structured allocation tail on Qwen tail tasks; testing the field's
actual allocators answers whether their schemes land in that tail or land near
uniform on it.

All allocators are pure CPU and floor-respecting. They take ``(num_groups,
total, floors)`` plus method-specific parameters and return a floor-respecting
largest-remainder-rounded budget vector that sums exactly to ``total``.

References:
- PyramidKV (Cai et al., 2024): monotone-decreasing per-layer KV budget,
  steepest at low layers. The published default ratio is ~1.5:1 (B_max:B_min)
  — at the group level that is ``alpha=0.2`` here.
- PyramidInfer (Yang et al., 2024) / "inverse pyramid": monotone-increasing
  per-layer budget — preferred where later layers are more compression-
  sensitive (the §3 output-deviation rung found Llama-1B's upper layers more
  sensitive at sub-8k, the natural target for an inverse pyramid).
- AdaKV (Feng et al., 2024): head-adaptive KV budget. The group-level analog
  is ``proportional_alloc`` driven by an externally supplied weight vector
  (the producer can feed it per-group priority-mass, cold-attention demand,
  or any other allocation signal).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.dart_pagedkv.budget import largest_remainder_round


def _check_inputs(num_groups: int, total: int, floors: Sequence[int]) -> int:
    """Common parameter validation; returns ``sum(floors)``."""
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if len(floors) != num_groups:
        raise ValueError(
            f"floors length {len(floors)} != num_groups {num_groups}"
        )
    floor_sum = sum(int(f) for f in floors)
    if total < floor_sum:
        raise ValueError(f"total {total} < sum(floors) {floor_sum}")
    return floor_sum


def pyramid_alloc(
    num_groups: int, total: int, floors: Sequence[int], *, alpha: float = 0.2
) -> list[int]:
    """PyramidKV-style monotone-decreasing budget across groups.

    Group 0 (low layers) receives the most discretionary budget; group G-1
    (high layers) receives the least. The discretionary share scales as
    ``1 + alpha*(1 − 2g/(G−1))`` for ``g ∈ [0, G−1]`` — linear in ``g``,
    centred on the mean discretionary share. With ``alpha=0`` the result is
    uniform; with ``alpha=0.2`` (PyramidKV's published shape) the
    group-0:group-(G−1) discretionary ratio is ``1.2:0.8 = 1.5:1``.
    ``alpha`` must be in ``[0, 1)`` so that the last-group share stays
    strictly positive at the limit.
    """
    floor_sum = _check_inputs(num_groups, total, floors)
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    floors_list = [int(f) for f in floors]
    if num_groups == 1:
        return [int(total)]
    discretionary = total - floor_sum
    mean_disc = discretionary / num_groups
    fractional = [float(f) for f in floors_list]
    for g in range(num_groups):
        shape = 1.0 + alpha * (1.0 - 2.0 * g / (num_groups - 1))
        fractional[g] = floors_list[g] + mean_disc * shape
    return largest_remainder_round(fractional, total, floors_list)


def inverse_pyramid_alloc(
    num_groups: int, total: int, floors: Sequence[int], *, alpha: float = 0.2
) -> list[int]:
    """Monotone-increasing budget — the PyramidKV reverse (PyramidInfer-style).

    Group 0 receives the least discretionary budget; group G-1 the most.
    Mirror image of :func:`pyramid_alloc` with the same ``alpha`` semantics.
    """
    floor_sum = _check_inputs(num_groups, total, floors)
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    floors_list = [int(f) for f in floors]
    if num_groups == 1:
        return [int(total)]
    discretionary = total - floor_sum
    mean_disc = discretionary / num_groups
    fractional = [float(f) for f in floors_list]
    for g in range(num_groups):
        shape = 1.0 + alpha * (2.0 * g / (num_groups - 1) - 1.0)
        fractional[g] = floors_list[g] + mean_disc * shape
    return largest_remainder_round(fractional, total, floors_list)


def proportional_alloc(
    num_groups: int, total: int, floors: Sequence[int],
    weights: Sequence[float],
) -> list[int]:
    """Budget proportional to externally supplied per-group ``weights``.

    The AdaKV-style group-level allocator. Weights come from the producer
    (per-group priority-mass, cold-attention demand, or any other allocation
    signal). Weights must be non-negative and sum to a positive number; zero-
    weight groups still receive their floor. The discretionary budget is
    divided proportionally and floor-respecting largest-remainder rounding
    makes the result sum to ``total``.
    """
    _check_inputs(num_groups, total, floors)
    weights_f = [float(w) for w in weights]
    if len(weights_f) != num_groups:
        raise ValueError(
            f"weights length {len(weights_f)} != num_groups {num_groups}"
        )
    if any(w < 0 for w in weights_f):
        raise ValueError(f"weights must be non-negative, got {weights_f}")
    weight_sum = sum(weights_f)
    if weight_sum <= 0:
        raise ValueError("at least one weight must be > 0")
    floors_list = [int(f) for f in floors]
    discretionary = total - sum(floors_list)
    fractional = [
        float(f) + discretionary * w / weight_sum
        for f, w in zip(floors_list, weights_f)
    ]
    return largest_remainder_round(fractional, total, floors_list)


def adakv_priority_weights(
    group_priorities: Sequence[Sequence[float]], total: int,
    floors: Sequence[int],
) -> list[float]:
    """Per-group priority-mass weight for AdaKV-style proportional allocation.

    For each group, takes the sum of the top ``B_g`` priority values where
    ``B_g`` is the uniform per-group budget — i.e. "how much priority-mass
    would this group capture under uniform allocation". Groups whose top-K
    priority mass dominates earn proportionally more budget under
    :func:`proportional_alloc`. The shape is uniform-anchored on purpose:
    a degenerate per-group priority (random / all-equal) returns equal
    weights, so the allocator falls back to uniform rather than
    drifting on noise.
    """
    if not group_priorities:
        raise ValueError("group_priorities must be non-empty")
    floors_list = [int(f) for f in floors]
    discretionary = total - sum(floors_list)
    if discretionary < 0:
        raise ValueError(
            f"total {total} below sum(floors) {sum(floors_list)} — "
            "AdaKV weights are undefined"
        )
    weights: list[float] = []
    for g, priority in enumerate(group_priorities):
        per_group_budget = floors_list[g] + discretionary // len(group_priorities)
        topk = min(per_group_budget, len(priority))
        sorted_desc = sorted(priority, reverse=True)
        weights.append(float(sum(sorted_desc[:topk])))
    return weights


PUBLISHED_ALLOCATORS = {
    "pyramid": pyramid_alloc,
    "inverse_pyramid": inverse_pyramid_alloc,
}
