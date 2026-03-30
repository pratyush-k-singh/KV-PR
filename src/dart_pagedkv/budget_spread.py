"""A predeclared spread of budget vectors that covers the simplex usefully.

For the Stage-0 bridge experiment we evaluate the formal cost vs teacher-forced
fidelity over a *fixed*, pre-registered set of budget vectors. This module
constructs that set deterministically from ``num_groups``, ``total``, and the
floors -- so the spread can be regenerated identically across runs and the
Gate-0 Spearman correlation is meaningful relative to a stable test
distribution.

The spread contains: the uniform allocation, one *single-dominant* vector per
group (that group absorbs all discretionary budget, others sit at their
floors), and -- when ``num_groups >= 3`` -- one *single-starved* vector per
group (that group sits at its floor, the rest share the discretionary budget
evenly). Duplicates are removed.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.dart_pagedkv.budget import (
    largest_remainder_round,
    uniform_budget,
    validate_budget,
)


def predeclared_spread(
    num_groups: int, total: int, floors: Sequence[int]
) -> list[list[int]]:
    """Return the deterministic, deduplicated predeclared budget-vector spread."""
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if len(floors) != num_groups:
        raise ValueError(
            f"floors length {len(floors)} != num_groups {num_groups}"
        )
    floors = [int(f) for f in floors]
    spread: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(b: list[int]) -> None:
        key = tuple(b)
        if key in seen:
            return
        validate_budget(b, total=total, floors=floors)
        seen.add(key)
        spread.append(b)

    add(uniform_budget(num_groups, total, floors))

    if num_groups == 1:
        return spread

    floor_sum = sum(floors)
    discretionary = total - floor_sum
    if discretionary < 0:
        raise ValueError(f"floors sum to {floor_sum}, exceeding total {total}")

    # Single-dominant: group g absorbs all discretionary budget.
    for g in range(num_groups):
        b = list(floors)
        b[g] = floors[g] + discretionary
        add(b)

    # Single-starved: group g sits at its floor, the rest share evenly. With
    # num_groups == 2 this collapses onto the single-dominant set.
    if num_groups >= 3:
        for g in range(num_groups):
            others = [i for i in range(num_groups) if i != g]
            share = discretionary / len(others)
            fractional = [float(f) for f in floors]
            for o in others:
                fractional[o] = floors[o] + share
            add(largest_remainder_round(fractional, total, floors))

    return spread


def ratio_allocation_grid(
    prompt_len: int,
    ratios: Sequence[float],
    num_groups: int,
    floors: Sequence[int],
) -> list[dict]:
    """Per-ratio predeclared allocation grids for the long-context probe.

    The long-context decode-probe has two budget axes (spec
    2026-05-17-longcontext-decode-probe §4): the *total* hot-token
    budget — a retention ``ratio`` of the prompt length — and the
    *allocation* of that total across layer-groups (`predeclared_spread`).

    For each ``ratio`` the total is ``round(ratio * prompt_len)``,
    clamped up to ``sum(floors)`` so the floors are always satisfiable;
    `predeclared_spread` then gives that total's deduplicated allocation
    grid. Returns one ``{"ratio", "total", "allocations"}`` entry per
    ratio — the two axes kept explicit so every trajectory is indexed
    by ``(ratio, allocation)`` and never collapsed to a tier mean.
    """
    if prompt_len < 1:
        raise ValueError(f"prompt_len must be >= 1, got {prompt_len}")
    floor_sum = sum(int(f) for f in floors)
    grid: list[dict] = []
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError(f"ratios must be > 0, got {ratio}")
        total = max(int(round(float(ratio) * prompt_len)), floor_sum)
        grid.append({
            "ratio": float(ratio),
            "total": total,
            "allocations": predeclared_spread(num_groups, total, floors),
        })
    return grid
