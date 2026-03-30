"""Two-tier (hot/cold) split: turn a budget vector into per-group hot sets.

Given a budget vector over layer-groups and each group's frozen token-priority
order, this produces the hot token set for every group. The cold tier is the
complement -- tokens retained but not in the HBM working set.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.dart_pagedkv.filler import protected_positions, select_hot


def group_hot_sets(
    budget: Sequence[int],
    group_priorities: Sequence[Sequence[float]],
    seq_len: int,
    n_sink: int,
    n_recent: int,
) -> list[list[int]]:
    """Return the sorted hot token indices for each layer-group.

    ``budget[g]`` is the hot-tier budget for group ``g``; ``group_priorities[g]``
    is that group's frozen per-token priority (length ``seq_len``). Protected
    sink/recent positions are forced hot in every group.
    """
    if len(budget) != len(group_priorities):
        raise ValueError(
            f"budget length {len(budget)} != group_priorities length "
            f"{len(group_priorities)}"
        )
    for g, priority in enumerate(group_priorities):
        if len(priority) != seq_len:
            raise ValueError(
                f"group_priorities[{g}] has length {len(priority)}, expected "
                f"seq_len {seq_len}"
            )
    protected = protected_positions(seq_len, n_sink, n_recent)
    return [
        select_hot(group_priorities[g], budget[g], protected)
        for g in range(len(budget))
    ]


def cold_indices(hot: Sequence[int], seq_len: int) -> list[int]:
    """Return the sorted cold-tier indices: ``[0, seq_len)`` minus ``hot``."""
    hot_set = {int(h) for h in hot}
    return [i for i in range(seq_len) if i not in hot_set]
