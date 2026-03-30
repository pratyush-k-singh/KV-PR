"""Partition transformer layers into contiguous groups for budget allocation."""

from __future__ import annotations


def make_layer_groups(num_layers: int, num_groups: int) -> list[list[int]]:
    """Partition ``range(num_layers)`` into ``num_groups`` contiguous groups.

    Group sizes are as even as possible; when ``num_layers`` is not divisible
    by ``num_groups`` the earlier groups absorb the remainder, so group sizes
    are non-increasing.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if not 1 <= num_groups <= num_layers:
        raise ValueError(f"num_groups must be in [1, {num_layers}], got {num_groups}")

    base, remainder = divmod(num_layers, num_groups)
    groups: list[list[int]] = []
    start = 0
    for group_idx in range(num_groups):
        size = base + (1 if group_idx < remainder else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def layer_to_group(groups: list[list[int]]) -> dict[int, int]:
    """Invert a layer-group partition into a layer-index -> group-index map."""
    mapping: dict[int, int] = {}
    for group_idx, layers in enumerate(groups):
        for layer in layers:
            mapping[layer] = group_idx
    return mapping
