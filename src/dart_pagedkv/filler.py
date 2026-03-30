"""The frozen monotone token-priority filler.

Given a fixed per-token priority order (frozen at prefill) and a per-group
budget, the filler returns the canonical hot set: the top-budget tokens by
priority, with protected sink/recent positions forced in. The priority order
is frozen, so top-k sets nest as the budget grows -- the property that makes a
budget vector a well-defined metrical-task-system state.
"""

from __future__ import annotations

from collections.abc import Sequence


def protected_positions(seq_len: int, n_sink: int, n_recent: int) -> list[int]:
    """Return the sorted sink + recent positions that must stay hot.

    Sinks are ``[0, n_sink)``; recent positions are the final ``n_recent`` of
    the sequence. Counts are clamped to ``[0, seq_len]`` and any overlap is
    deduplicated.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be >= 0, got {seq_len}")
    n_sink = max(0, min(int(n_sink), seq_len))
    n_recent = max(0, min(int(n_recent), seq_len))
    positions = set(range(n_sink))
    positions.update(range(seq_len - n_recent, seq_len))
    return sorted(positions)


def select_hot(
    priority: Sequence[float], budget: int, protected: Sequence[int] = ()
) -> list[int]:
    """Return the sorted hot-token indices for one group at a given budget.

    Protected positions are forced in (and count against the budget); the rest
    of the budget is filled by the highest-``priority`` positions, with ties
    broken to the lowest index. Because ``priority`` is a fixed order, the
    selected sets nest as ``budget`` grows.
    """
    n = len(priority)
    budget = max(0, min(int(budget), n))
    protected_in = sorted({int(p) for p in protected if 0 <= int(p) < n})

    hot = list(protected_in[:budget])
    if len(hot) >= budget:
        return sorted(hot)

    hot_set = set(hot)
    rest = [i for i in range(n) if i not in hot_set]
    rest.sort(key=lambda i: (-float(priority[i]), i))
    for i in rest:
        hot.append(i)
        if len(hot) >= budget:
            break
    return sorted(hot)
