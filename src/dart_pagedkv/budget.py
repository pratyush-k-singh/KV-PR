"""Budget-vector primitives: validation, deterministic rounding, interpolation.

A budget vector allocates a fixed total hot-tier budget ``B`` across ``G``
layer-groups. Each entry is a non-negative integer at or above a per-group
protected floor; the entries sum exactly to ``B``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def validate_budget(b: Sequence[float], total: int, floors: Sequence[int]) -> None:
    """Raise ``ValueError`` unless ``b`` is a valid budget vector.

    Valid means: same length as ``floors``; every entry integral and at or
    above its floor; entries sum exactly to ``total``.
    """
    if len(b) != len(floors):
        raise ValueError(f"budget length {len(b)} != floors length {len(floors)}")
    for i, value in enumerate(b):
        if value != int(value):
            raise ValueError(f"budget entry {i} is not an integer: {value!r}")
    for i, (value, floor) in enumerate(zip(b, floors)):
        if value < floor:
            raise ValueError(f"budget entry {i} ({value}) is below its floor ({floor})")
    actual = sum(int(value) for value in b)
    if actual != total:
        raise ValueError(f"budget sums to {actual}, expected {total}")


def largest_remainder_round(
    fractional: Sequence[float], total: int, floors: Sequence[int]
) -> list[int]:
    """Round a fractional budget to integers summing exactly to ``total``.

    The rounding is applied to the non-floor budget: floors are subtracted,
    the remainder is rounded by the largest-remainder (Hamilton) method with
    ties broken to the lowest index, and floors are added back. Every entry of
    the result is at or above its floor. The procedure is fully deterministic
    and introduces no data-dependent adaptivity.
    """
    if len(fractional) != len(floors):
        raise ValueError(
            f"fractional length {len(fractional)} != floors length {len(floors)}"
        )
    floor_sum = sum(int(f) for f in floors)
    if floor_sum > total:
        raise ValueError(f"floors sum to {floor_sum}, exceeding total {total}")

    remainder_total = total - floor_sum
    lowers: list[int] = []
    frac_parts: list[float] = []
    for i, (value, floor) in enumerate(zip(fractional, floors)):
        rem = float(value) - float(floor)
        if rem < -1e-9:
            raise ValueError(
                f"fractional entry {i} ({value}) is below its floor ({floor})"
            )
        rem = max(rem, 0.0)
        low = math.floor(rem)
        lowers.append(low)
        frac_parts.append(rem - low)

    deficit = remainder_total - sum(lowers)
    if not 0 <= deficit <= len(fractional):
        raise ValueError(
            f"fractional vector does not sum near total ({total}); deficit={deficit}"
        )

    order = sorted(range(len(fractional)), key=lambda i: (-frac_parts[i], i))
    round_up = set(order[:deficit])
    return [
        int(floors[i]) + lowers[i] + (1 if i in round_up else 0)
        for i in range(len(fractional))
    ]


def interpolate_budget(
    b_adv: Sequence[int],
    b_rob: Sequence[int],
    lam: float,
    total: int,
    floors: Sequence[int],
) -> list[int]:
    """Blend an advice budget and a robust budget at trust weight ``lam``.

    Returns ``round(lam * b_adv + (1 - lam) * b_rob)``. ``lam`` is clamped to
    ``[0, 1]``; at ``lam == 1`` the result is ``b_adv``, at ``lam == 0`` it is
    ``b_rob``. ``b_adv`` and ``b_rob`` must both be valid budgets for the same
    ``total`` and ``floors``.
    """
    if len(b_adv) != len(b_rob):
        raise ValueError(f"b_adv length {len(b_adv)} != b_rob length {len(b_rob)}")
    lam = min(1.0, max(0.0, float(lam)))
    fractional = [
        lam * float(adv) + (1.0 - lam) * float(rob)
        for adv, rob in zip(b_adv, b_rob)
    ]
    return largest_remainder_round(fractional, total, floors)


def uniform_budget(num_groups: int, total: int, floors: Sequence[int]) -> list[int]:
    """Allocate ``total`` evenly across ``num_groups`` groups above their floors."""
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if len(floors) != num_groups:
        raise ValueError(
            f"floors length {len(floors)} != num_groups {num_groups}"
        )
    floor_sum = sum(int(f) for f in floors)
    share = (total - floor_sum) / num_groups
    fractional = [float(f) + share for f in floors]
    return largest_remainder_round(fractional, total, floors)
