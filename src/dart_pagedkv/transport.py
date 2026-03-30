"""DART/MTS adaptive transport over the budget simplex.

The λ-update arbitrates between an ADV (advice) recommendation and a ROB
(robust) recommendation using observed service cost as the signal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .budget import interpolate_budget


@dataclass(frozen=True)
class TransportConfig:
    """Static configuration for a DART transport simulation."""

    rob_budget: tuple[int, ...]
    total: int
    floors: tuple[int, ...]
    eta: float = 1.0
    lam_init: float = 0.5
    movement_weight: float = 0.0
    eta_decay: float = 0.0  # 0 = constant η; positive ⇒ η_t = η/(1 + decay·t)


@dataclass
class TransportTrajectory:
    """Per-round history produced by ``simulate_dart``."""

    lam_history: list[float] = field(default_factory=list)
    budgets_played: list[tuple[int, ...]] = field(default_factory=list)
    costs_observed: list[float] = field(default_factory=list)
    logit_kl_played: list[float] = field(default_factory=list)
    movement_costs: list[float] = field(default_factory=list)


def simulate_dart(
    prompts: Sequence[str],
    records_by_prompt: dict[str, list[dict]],
    adv_fn: Callable[[str], Sequence[int]],
    config: TransportConfig,
    cost_key: str = "cost_max",
    probe_interval: int = 1,
) -> TransportTrajectory:
    """Run DART transport over a prompt workload using pre-recorded outcomes.

    For each prompt:
      1. ADV recommends a budget vector.
      2. ``b_play = round(λ b_adv + (1-λ) b_rob)`` is computed and snapped to
         the nearest recorded budget for this prompt.
      3. ``c_play`` and ``logit_kl_played`` are looked up from the records.
      4. On probe rounds (every ``probe_interval``-th, including round 0),
         fresh ADV/ROB ``cost_key`` values are observed and cached as the
         current regret ``c_adv - c_rob``. Non-probe rounds reuse the most
         recently observed regret without refreshing the signal.
      5. ``λ`` updates every round using the current cached regret.

    ``cost_key`` selects which record field DART observes as cost: the
    default ``"cost_max"`` is the bottleneck service cost (what a real
    online system would have cheaply); ``"logit_kl"`` is the oracle
    fidelity signal used to test the observability gate (F).
    """
    traj = TransportTrajectory()
    lam = config.lam_init
    last_budget: tuple[int, ...] | None = None
    cached_regret: float | None = None

    for t, pid in enumerate(prompts):
        records = records_by_prompt[pid]
        recorded_budgets = [tuple(r["budget"]) for r in records]

        b_adv = tuple(int(x) for x in adv_fn(pid))
        b_interp = tuple(interpolate_budget(
            b_adv, config.rob_budget, lam, config.total, config.floors,
        ))
        _, b_play = snap_to_recorded(b_interp, recorded_budgets)
        play_record = records[recorded_budgets.index(b_play)]

        m = 0.0 if last_budget is None else float(
            sum(abs(a - b) for a, b in zip(b_play, last_budget))
        )

        traj.lam_history.append(lam)
        traj.budgets_played.append(b_play)
        traj.costs_observed.append(float(play_record["cost_max"]))
        traj.logit_kl_played.append(float(play_record["logit_kl"]))
        traj.movement_costs.append(m)

        if t % probe_interval == 0:
            # Probe round: observe fresh ADV/ROB counterfactual cost.
            _, b_adv_snap = snap_to_recorded(b_adv, recorded_budgets)
            _, b_rob_snap = snap_to_recorded(config.rob_budget, recorded_budgets)
            c_adv = float(records[recorded_budgets.index(b_adv_snap)][cost_key])
            c_rob = float(records[recorded_budgets.index(b_rob_snap)][cost_key])
            cached_regret = c_adv - c_rob

        # λ-update uses current cached regret (fresh on probe rounds, stale
        # otherwise). If no probe has happened yet, λ is left unchanged.
        if cached_regret is not None:
            eta_t = config.eta / (1.0 + config.eta_decay * t) if config.eta_decay > 0 else config.eta
            lam = max(0.0, min(1.0, lam - eta_t * cached_regret))
        last_budget = b_play

    return traj


def snap_to_recorded(
    candidate: Sequence[int], recorded: Sequence[Sequence[int]]
) -> tuple[int, tuple[int, ...]]:
    """Return ``(index, recorded_budget)`` minimizing L1 distance to ``candidate``.

    Ties are broken to the lowest index for determinism.
    """
    best_idx = 0
    best_d: int | None = None
    best_b: tuple[int, ...] | None = None
    for i, b in enumerate(recorded):
        d = sum(abs(int(c) - int(r)) for c, r in zip(candidate, b))
        if best_d is None or d < best_d:
            best_d = d
            best_idx = i
            best_b = tuple(int(r) for r in b)
    assert best_b is not None, "recorded must be non-empty"
    return best_idx, best_b


def update_lam(lam: float, c_adv: float, c_rob: float, eta: float) -> float:
    """One round of the DART λ-update.

    The update is ``λ_{t+1} = clip(λ_t - η · (c_adv - c_rob), 0, 1)``. When
    ``c_adv < c_rob`` ADV's pure recommendation would have been cheaper, so
    ``λ`` increases (more trust). Symmetric for ``c_adv > c_rob``. Equal
    costs leave ``λ`` unchanged.
    """
    new_lam = lam - eta * (c_adv - c_rob)
    return max(0.0, min(1.0, new_lam))
