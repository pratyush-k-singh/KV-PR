#!/usr/bin/env python3
"""Long-context decode-probe — trajectory analyzer.

The CPU analyze half. Loads the producer's ``trajectories.json`` (per-prompt
per-``(ratio, allocation)`` records of ``cost`` AND ``per_step_kl``, plus the
per-prompt ``effective_support``) and derives the effective-context-saturation
metrics (spec `docs/superpowers/specs/2026-05-17-effective-context-saturation.md`).

Per ``(prompt, ratio)`` it identifies four scenarios honestly:

  * ``uniform``       — the uniform allocation of that ratio's total.
  * ``mincost_bad``   — argmin **cost** (the cost-following advisor's pick;
                        NOT assumed dominate-g0).
  * ``oracle_bstar``  — argmin total per-step KL.
  * ``worst_kl``      — argmax total per-step KL.

For each it takes the **per-token mean** logit-KL over masked decode steps
1..k_dec-1 (step 0 = shared prefill = 0; §14.2 sum trap avoided). Metrics:

  * ``oracle_gap = uniform_kl - oracle_kl`` — the **primary saturation
    metric**: does a better-than-uniform allocation exist? (spec §7).
  * ``worst_gap  = worst_kl  - oracle_kl`` — adversarial tail risk.
  * ``mincost_vs_uniform``                 — the cost-degeneracy diagnostic
    (≈0 at long context — the cost argmin *is* uniform).
  * ``min_margin = min_g(B_g - S_g)``      — the saturation predictor: the
    uniform per-group budget minus the group's effective-context support.
    Computed with mean support (``min_margin_mean``) and bottleneck/max
    support (``min_margin_max``). H1: ``oracle_gap`` collapses once
    ``min_margin`` crosses ≥ 0.

Collapsed ``(tier, ratio)`` cells (``ratio*T <= floor_sum``) are excluded from
aggregate means. Writes ``stats.json`` + ``report.md``. Pure CPU; no torch.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.dart_pagedkv.budget import uniform_budget
from src.dart_pagedkv.filler import protected_positions


# --- summary-distribution helpers (status §17 — paper-grade rigor) -------


def bootstrap_median_ci(
    xs: list[float], *, level: float = 0.95, n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Bootstrap percentile CI for the median of ``xs``.

    Resamples with replacement ``n_resamples`` times; returns the ``(lo, hi)``
    percentiles of the resampled medians at confidence ``level``. Returns
    ``None`` if ``xs`` is empty. Single-prompt cells return a degenerate
    ``(x, x)`` interval — informative as "we have no width estimate here".
    """
    if not xs:
        return None
    rng = random.Random(seed)
    n = len(xs)
    medians: list[float] = []
    for _ in range(n_resamples):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    alpha = 1.0 - level
    lo_idx = max(0, int(round(n_resamples * alpha / 2)))
    hi_idx = min(n_resamples - 1, int(round(n_resamples * (1 - alpha / 2))) - 1)
    return float(medians[lo_idx]), float(medians[hi_idx])


def tail_rate(xs: list[float], *, threshold: float) -> float | None:
    """Fraction of values strictly greater than ``threshold``.

    The "how often does a prompt hit the heavy tail" diagnostic — a cell with
    mean 0.5 but tail_rate@1.0 = 30% has a third of prompts above 1.0, very
    different from a cell where every prompt is around 0.5.
    """
    if not xs:
        return None
    return float(sum(1 for x in xs if x > threshold)) / len(xs)


def heavy_tail_index(xs: list[float]) -> float | None:
    """``max / median`` ratio (None when median ≤ 0 or input is empty).

    ``≈1`` → uniform cell (every prompt similar). ``≫1`` → the cell's
    `oracle_gap` is a sparse heavy tail riding on a small median.
    """
    if not xs:
        return None
    med = statistics.median(xs)
    if med <= 0:
        return None
    return float(max(xs)) / float(med)


def mean_per_step_kl(per_step_kl: list[float]) -> float:
    """Mean over masked decode steps (skip step 0 = shared prefill = 0)."""
    tail = per_step_kl[1:]
    return float(statistics.fmean(tail)) if tail else 0.0


def protected_floors(seq_len: int, num_groups: int, n_sink: int, n_recent: int) -> list[int]:
    """Replicate `longcontext_decode_probe.protected_budget_floors` without
    pulling its torch dependencies."""
    return [len(protected_positions(seq_len, n_sink, n_recent))] * num_groups


def classify_allocation(
    alloc: list[int], floor: int, tol_uniform: int = 1, tol_floor: int = 5,
) -> str:
    """Classify a budget vector by its shape.

    Returns one of ``uniform``, ``dominant_g<i>``, ``starved_g<i>``, ``other``.
    The classification underpins the **concentration index** (status §21):
    on diffuse-attention models like Phi-3.5-mini, ``dominant_g<i>`` beats
    ``uniform`` under SnapKV but loses to it under random — the inversion
    confirms the priority×allocation interaction is a model property.

    ``tol_uniform`` allows largest-remainder rounding wobble (≤1 per group).
    ``tol_floor`` allows a near-floor entry to count as "starved" if it is
    within a few tokens of the protected floor (post-rounding).
    """
    if not alloc:
        return "other"
    if max(alloc) - min(alloc) <= tol_uniform:
        return "uniform"
    near_floor = [a <= floor + tol_floor for a in alloc]
    if sum(near_floor) == len(alloc) - 1:
        return f"dominant_g{near_floor.index(False)}"
    if sum(near_floor) == 1:
        # Genuine single-starved: the non-floor entries must be roughly equal,
        # matching `predeclared_spread`'s single-starved shape. Otherwise the
        # vector is just lumpy and should be classified as ``other``.
        other_vals = [a for a, nf in zip(alloc, near_floor) if not nf]
        if other_vals and (max(other_vals) - min(other_vals)) <= tol_uniform + 1:
            return f"starved_g{near_floor.index(True)}"
    return "other"


def select_scenarios_for_ratio(
    trajectories: list[dict[str, Any]], uniform_allocation: list[int]
) -> dict[str, dict[str, Any]]:
    """Identify the four scenario trajectories for one ratio.

    ``trajectories`` is the list of allocation records sharing a single
    ratio (each with ``allocation``, ``cost``, ``per_step_kl``).
    ``uniform_allocation`` is the canonical uniform budget for the
    ratio's total (matched by exact equality). The ``uniform`` value may
    be ``None`` if no allocation matches it exactly (should not happen
    for `predeclared_spread` output).
    """
    if not trajectories:
        raise ValueError("empty trajectory list for ratio")
    uniform_tuple = tuple(int(x) for x in uniform_allocation)
    uniform = next(
        (t for t in trajectories if tuple(t["allocation"]) == uniform_tuple),
        None,
    )
    return {
        "uniform": uniform,
        "mincost_bad": min(trajectories, key=lambda t: t["cost"]),
        "oracle_bstar": min(trajectories, key=lambda t: sum(t["per_step_kl"])),
        "worst_kl": max(trajectories, key=lambda t: sum(t["per_step_kl"])),
    }


def _scenario_record(traj: dict[str, Any] | None) -> dict[str, Any]:
    if traj is None:
        return {"allocation": None, "kl_per_step": None}
    return {
        "allocation": list(traj["allocation"]),
        "kl_per_step": mean_per_step_kl(traj["per_step_kl"]),
        "cost": traj.get("cost"),
    }


def _delta(a: float | None, b: float | None) -> float | None:
    """``a - b``, propagating ``None``."""
    return None if a is None or b is None else a - b


def _support_margins(
    effective_support: dict[str, Any] | None,
    uniform_budget_per_group: list[int],
) -> tuple[float | None, float | None]:
    """``min_g(B_g - S_g)`` for mean and bottleneck (max) support.

    ``B_g`` is the uniform per-group budget; ``S_g`` is the group's
    effective-context support. A negative margin means the uniform budget
    does NOT cover the group's support — the unsaturated regime where
    allocation is predicted to matter. Returns ``(None, None)`` when the
    producer wrote no ``effective_support`` (old runs).
    """
    if not effective_support:
        return None, None
    s_mean = effective_support.get("per_group")
    s_max = effective_support.get("per_group_max")
    num_groups = len(uniform_budget_per_group)
    if (not s_mean or not s_max
            or len(s_mean) != num_groups or len(s_max) != num_groups):
        return None, None
    margin_mean = min(
        uniform_budget_per_group[g] - s_mean[g] for g in range(num_groups)
    )
    margin_max = min(
        uniform_budget_per_group[g] - s_max[g] for g in range(num_groups)
    )
    return float(margin_mean), float(margin_max)


def _pick_support(
    prompt: dict[str, Any], estimator: str | None, tau: float | None
) -> dict[str, Any] | None:
    """Select the effective-support entry to use as the saturation predictor.

    ``estimator=None`` returns the producer's primary ``effective_support``
    (the only field present on pre-grid runs). Otherwise the matching
    entry is taken from the per-prompt ``effective_support_grid`` (the
    multi-estimator pre-compute) — ``tau`` is matched only for
    ``mass_coverage``; ``participation_ratio`` / ``entropy`` are tau-free.
    Returns ``None`` if a specific estimator is requested but absent from
    the grid (the report then shows no margins, rather than silently
    using the wrong support).
    """
    if estimator is None:
        return prompt.get("effective_support")
    for entry in prompt.get("effective_support_grid") or []:
        if entry.get("estimator") != estimator:
            continue
        if estimator == "mass_coverage" and tau is not None and entry.get("tau") != tau:
            continue
        return entry
    return None


def analyze_prompt(
    prompt: dict[str, Any], num_groups: int, n_sink: int, n_recent: int,
    *, support_estimator: str | None = None, support_tau: float | None = None,
) -> dict[str, Any]:
    """Per-prompt analysis: per-ratio scenario picks, deltas and saturation margins."""
    tokens = int(prompt["tokens"])
    floors = protected_floors(tokens, num_groups, n_sink, n_recent)
    floor_sum = int(sum(floors))
    effective_support = _pick_support(prompt, support_estimator, support_tau)
    by_ratio: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for traj in prompt.get("trajectories", []):
        by_ratio[traj["ratio"]].append(traj)
    ratio_summaries = {rs["ratio"]: rs for rs in prompt.get("ratio_summaries", [])}

    per_ratio: list[dict[str, Any]] = []
    floor_per_group = floors[0] if floors else 0
    for ratio, trajectories in sorted(by_ratio.items()):
        total = int(trajectories[0]["total"])
        uniform_alloc = uniform_budget(num_groups, total, floors)
        scenarios = select_scenarios_for_ratio(trajectories, uniform_alloc)
        u = _scenario_record(scenarios["uniform"])
        m = _scenario_record(scenarios["mincost_bad"])
        o = _scenario_record(scenarios["oracle_bstar"])
        w = _scenario_record(scenarios["worst_kl"])
        rs = ratio_summaries.get(ratio, {})
        collapsed = bool(rs.get("collapsed", total <= floor_sum))
        margin_mean, margin_max = _support_margins(effective_support, uniform_alloc)
        # worst_gap == best_vs_worst delta; kept under both names — the old
        # key for backward compatibility, the new one for the saturation read.
        worst_gap = _delta(w["kl_per_step"], o["kl_per_step"])
        # Named published allocators (status §17, opt-in via producer's
        # --extra-allocators). Map each tag to its KL — vs-uniform delta is
        # computed lazily by the renderer.
        named: dict[str, dict[str, Any]] = {}
        for traj in trajectories:
            tag = traj.get("allocator_tag")
            if not tag:
                continue
            named[tag] = _scenario_record(traj)
        # Concentration / spread indices — the mechanistic signal (status
        # §21 mechanism). Best single-dominant vs uniform → ``conc_idx``;
        # best single-starved vs uniform → ``spread_idx``. Positive means
        # the imbalanced allocation beats uniform.
        dom_kls: list[float] = []
        star_kls: list[float] = []
        for traj in trajectories:
            atype = classify_allocation(traj["allocation"], floor_per_group)
            if atype.startswith("dominant_g"):
                dom_kls.append(mean_per_step_kl(traj["per_step_kl"]))
            elif atype.startswith("starved_g"):
                star_kls.append(mean_per_step_kl(traj["per_step_kl"]))
        best_dominant_kl = min(dom_kls) if dom_kls else None
        best_starved_kl = min(star_kls) if star_kls else None
        u_kl = u["kl_per_step"]
        if u_kl is not None and u_kl > 1e-9:
            conc_idx = (
                None if best_dominant_kl is None
                else (u_kl - best_dominant_kl) / u_kl
            )
            spread_idx = (
                None if best_starved_kl is None
                else (u_kl - best_starved_kl) / u_kl
            )
        else:
            conc_idx = None
            spread_idx = None
        per_ratio.append({
            "ratio": ratio,
            "total": total,
            "discretionary": rs.get("discretionary", total - floor_sum),
            "collapsed": collapsed,
            "n_distinct_allocations": rs.get(
                "n_distinct_allocations",
                len({tuple(t["allocation"]) for t in trajectories}),
            ),
            "uniform": u,
            "mincost_bad": m,
            "oracle_bstar": o,
            "worst_kl": w,
            "named_allocators": named,
            "uniform_budget_per_group": list(uniform_alloc),
            "mincost_vs_uniform_per_step_delta": _delta(
                m["kl_per_step"], u["kl_per_step"]
            ),
            "best_vs_worst_per_step_delta": worst_gap,
            "oracle_gap_per_step": _delta(u["kl_per_step"], o["kl_per_step"]),
            "worst_gap_per_step": worst_gap,
            "best_dominant_kl_per_step": best_dominant_kl,
            "best_starved_kl_per_step": best_starved_kl,
            "conc_idx_per_step": conc_idx,
            "spread_idx_per_step": spread_idx,
            "min_margin_mean": margin_mean,
            "min_margin_max": margin_max,
        })
    return {
        "prompt_id": prompt.get("prompt_id"),
        "task": prompt.get("task"),
        "tier_length": prompt.get("tier_length"),
        "tokens": tokens,
        "floor_sum": floor_sum,
        "effective_support": effective_support,
        "per_ratio": per_ratio,
    }


_BUCKET_FIELDS = (
    "mincost_vs_uniform", "best_vs_worst", "oracle_gap",
    "uniform_kl", "mincost_kl", "min_margin_mean", "min_margin_max",
    "conc_idx", "spread_idx",
)


def aggregate(per_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (tier_length, ratio) aggregate, EXCLUDING collapsed cells."""
    buckets: dict[tuple[int, float], dict[str, list[float]]] = defaultdict(
        lambda: {field: [] for field in _BUCKET_FIELDS}
    )
    named_buckets: dict[
        tuple[int, float], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    counts: dict[tuple[int, float], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "collapsed": 0, "meaningful": 0}
    )
    for prompt in per_prompt:
        for cell in prompt["per_ratio"]:
            key = (int(prompt["tier_length"]), float(cell["ratio"]))
            counts[key]["total"] += 1
            if cell["collapsed"]:
                counts[key]["collapsed"] += 1
                continue
            counts[key]["meaningful"] += 1
            bucket = buckets[key]
            # Named-allocator KLs per cell (status §17 audit control).
            uniform_kl = cell["uniform"]["kl_per_step"]
            for tag, rec in (cell.get("named_allocators") or {}).items():
                kl = rec.get("kl_per_step")
                if kl is None or uniform_kl is None:
                    continue
                named_buckets[key][f"{tag}_kl"].append(kl)
                named_buckets[key][f"{tag}_vs_uniform"].append(kl - uniform_kl)

            def _push(field: str, value: float | None) -> None:
                if value is not None:
                    bucket[field].append(value)

            _push("mincost_vs_uniform", cell["mincost_vs_uniform_per_step_delta"])
            _push("best_vs_worst", cell["best_vs_worst_per_step_delta"])
            _push("oracle_gap", cell["oracle_gap_per_step"])
            _push("uniform_kl", cell["uniform"]["kl_per_step"])
            _push("mincost_kl", cell["mincost_bad"]["kl_per_step"])
            _push("min_margin_mean", cell["min_margin_mean"])
            _push("min_margin_max", cell["min_margin_max"])
            _push("conc_idx", cell.get("conc_idx_per_step"))
            _push("spread_idx", cell.get("spread_idx_per_step"))

    def _mean(xs: list[float]) -> float | None:
        return float(statistics.fmean(xs)) if xs else None

    def _median(xs: list[float]) -> float | None:
        return float(statistics.median(xs)) if xs else None

    def _max(xs: list[float]) -> float | None:
        return float(max(xs)) if xs else None

    rows: list[dict[str, Any]] = []
    for key in sorted(buckets.keys() | counts.keys()):
        tier_length, ratio = key
        bucket = buckets.get(key, {field: [] for field in _BUCKET_FIELDS})
        count = counts[key]
        og = bucket["oracle_gap"]
        # Bootstrap percentile CI on the per-prompt oracle_gap median —
        # answers "does the per-step lever's median CI exclude 0?" without
        # needing a normality assumption. 2000 resamples is plenty at n=24.
        ci = bootstrap_median_ci(og, level=0.95, n_resamples=2000)
        rows.append({
            "tier_length": tier_length,
            "ratio": ratio,
            "n_prompts_total": count["total"],
            "n_prompts_collapsed": count["collapsed"],
            "n_prompts_meaningful": count["meaningful"],
            "mean_mincost_vs_uniform_per_step": _mean(bucket["mincost_vs_uniform"]),
            "mean_best_vs_worst_per_step": _mean(bucket["best_vs_worst"]),
            "mean_oracle_gap_per_step": _mean(og),
            # oracle_gap distribution — a heavy tail (rare prompts with huge
            # allocation headroom) shows up as median << mean << max.
            "median_oracle_gap_per_step": _median(og),
            "max_oracle_gap_per_step": _max(og),
            "oracle_gap_median_ci95_lo": ci[0] if ci else None,
            "oracle_gap_median_ci95_hi": ci[1] if ci else None,
            # Heavy-tail diagnostic + tail rates at three thresholds.
            "oracle_gap_heavy_tail_index": heavy_tail_index(og),
            "oracle_gap_tail_rate_0p10": tail_rate(og, threshold=0.10),
            "oracle_gap_tail_rate_0p50": tail_rate(og, threshold=0.50),
            "oracle_gap_tail_rate_1p00": tail_rate(og, threshold=1.00),
            # Mechanistic indices (status §21): per-cell median across prompts.
            "conc_idx_median": _median(bucket["conc_idx"]),
            "conc_idx_mean": _mean(bucket["conc_idx"]),
            "spread_idx_median": _median(bucket["spread_idx"]),
            "spread_idx_mean": _mean(bucket["spread_idx"]),
            "mean_uniform_kl_per_step": _mean(bucket["uniform_kl"]),
            "mean_mincost_kl_per_step": _mean(bucket["mincost_kl"]),
            "mean_min_margin_mean": _mean(bucket["min_margin_mean"]),
            "mean_min_margin_max": _mean(bucket["min_margin_max"]),
            "named_allocator_means": {
                field: _mean(values)
                for field, values in sorted(named_buckets.get(key, {}).items())
            },
        })
    return rows


def _fmt(x: Any) -> str:
    return "—" if x is None else f"{x:.4f}"


def render_report(stats: dict[str, Any]) -> str:
    """Tidy per-(tier,ratio) Markdown tables for the analyses dir."""
    lines: list[str] = []
    lines.append("# Long-context decode-probe analysis\n")
    lines.append(f"- model: `{stats.get('model')}`")
    lines.append(f"- manifest: `{stats.get('manifest')}`")
    lines.append(f"- ratios: {stats.get('ratios')}")
    lines.append(f"- k_dec: {stats.get('k_dec')}  obs_window: {stats.get('obs_window')}  "
                 f"num_groups: {stats.get('num_groups')}  "
                 f"n_sink: {stats.get('n_sink')}  n_recent: {stats.get('n_recent')}")
    support_estimator = stats.get("support_estimator")
    if support_estimator:
        lines.append(f"- support estimator: `{support_estimator}`  "
                     f"tau: {stats.get('support_tau')}")
    priority = stats.get("priority")
    if priority:
        lines.append(f"- within-group priority: `{priority}`")
    lines.append("")
    lines.append("Per-step (per-token mean) KL across decode steps 1..K-1.")
    lines.append("`collapsed` cells (ratio*T <= floor_sum) are excluded from "
                 "aggregate means — averaging deltas over zero-variance rows "
                 "would dilute the real signal.")
    lines.append("")
    lines.append("## Aggregate per (tier, ratio)")
    lines.append("")
    lines.append("`oracle_gap = uniform_kl - oracle_kl` is the primary "
                 "sensitivity metric (does a better-than-uniform allocation "
                 "exist); `mincost−uniform` is the cost-degeneracy diagnostic "
                 "(≈0 when the cost argmin is the uniform vector). "
                 "`og_med`/`og_max` are the median and max of `oracle_gap` "
                 "over the cell's meaningful prompts — `og_med ≪ oracle_gap` "
                 "means the allocation gap is a heavy tail, not a broad shift.")
    lines.append("")
    lines.append("| tier | ratio | n_total | n_collapsed | n_meaningful | "
                 "oracle_gap | og_med | og_max | mincost−uniform | worst−oracle | "
                 "uniform_kl | mincost_kl |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in stats["aggregate_per_tier_ratio"]:
        lines.append(
            f"| {row['tier_length']} | {row['ratio']:.4f} | "
            f"{row['n_prompts_total']} | {row['n_prompts_collapsed']} | "
            f"{row['n_prompts_meaningful']} | "
            f"{_fmt(row['mean_oracle_gap_per_step'])} | "
            f"{_fmt(row['median_oracle_gap_per_step'])} | "
            f"{_fmt(row['max_oracle_gap_per_step'])} | "
            f"{_fmt(row['mean_mincost_vs_uniform_per_step'])} | "
            f"{_fmt(row['mean_best_vs_worst_per_step'])} | "
            f"{_fmt(row['mean_uniform_kl_per_step'])} | "
            f"{_fmt(row['mean_mincost_kl_per_step'])} |"
        )
    lines.append("")
    lines.extend(_render_tail_section(stats["aggregate_per_tier_ratio"]))
    lines.extend(_render_named_allocator_section(stats["aggregate_per_tier_ratio"]))
    lines.extend(_render_saturation_section(stats["aggregate_per_tier_ratio"]))
    return "\n".join(lines) + "\n"


def _render_named_allocator_section(rows: list[dict[str, Any]]) -> list[str]:
    """Named-allocator KLs per cell (status §17 published-allocator audit)."""
    tags: dict[str, None] = {}
    for row in rows:
        for field in (row.get("named_allocator_means") or {}):
            if field.endswith("_kl"):
                tag = field[:-3]
                tags[tag] = None
    if not tags:
        return []
    tag_list = list(tags)
    lines = ["## Named allocators", ""]
    lines.append(
        "Per-cell published-allocator KLs. `<tag>−uniform` is positive when "
        "the named allocator is **worse** than uniform; the §21 audit reads "
        "as: how close do published methods land to uniform on the cells "
        "where the layer-allocation lever is real?"
    )
    lines.append("")
    header = ["tier", "ratio"]
    for tag in tag_list:
        header.append(f"{tag}_kl")
        header.append(f"{tag}−uniform")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for row in rows:
        means = row.get("named_allocator_means") or {}
        cells = [str(row["tier_length"]), f"{row['ratio']:.4f}"]
        for tag in tag_list:
            cells.append(_fmt(means.get(f"{tag}_kl")))
            cells.append(_fmt(means.get(f"{tag}_vs_uniform")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _render_tail_section(rows: list[dict[str, Any]]) -> list[str]:
    """Bootstrap CI + heavy-tail/tail-rate diagnostics for `oracle_gap`."""
    lines = ["## Oracle-gap distribution diagnostics", ""]
    lines.append("Bootstrap 95% CI on the **median** per-prompt `oracle_gap` "
                 "(2000 resamples, percentile method) is the rigorous "
                 "'is the lever there' read on small n. `tail_rate@T` is the "
                 "fraction of meaningful prompts with `oracle_gap > T`; "
                 "`heavy_tail_index = og_max / og_med` ≫ 1 means the cell "
                 "is a sparse heavy tail.")
    lines.append("")
    lines.append("| tier | ratio | og_med | CI95 lo | CI95 hi | heavy_tail_index | "
                 "tail@0.1 | tail@0.5 | tail@1.0 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['tier_length']} | {row['ratio']:.4f} | "
            f"{_fmt(row['median_oracle_gap_per_step'])} | "
            f"{_fmt(row['oracle_gap_median_ci95_lo'])} | "
            f"{_fmt(row['oracle_gap_median_ci95_hi'])} | "
            f"{_fmt(row['oracle_gap_heavy_tail_index'])} | "
            f"{_fmt(row['oracle_gap_tail_rate_0p10'])} | "
            f"{_fmt(row['oracle_gap_tail_rate_0p50'])} | "
            f"{_fmt(row['oracle_gap_tail_rate_1p00'])} |"
        )
    lines.append("")
    return lines


def _render_saturation_section(rows: list[dict[str, Any]]) -> list[str]:
    """The effective-context-saturation table — oracle_gap vs support margin."""
    lines = ["## Effective-context saturation", ""]
    has_margin = any(r["mean_min_margin_mean"] is not None for r in rows)
    if not has_margin:
        lines.append("`effective_support` absent from trajectories.json — "
                     "re-run the producer (build step 2) to populate the "
                     "saturation predictor.")
        lines.append("")
        return lines
    lines.append("`oracle_gap = uniform_kl - oracle_kl` is the primary "
                 "sensitivity metric (does a better-than-uniform allocation "
                 "exist). `min_margin = min_g(B_g - S_g)` is the saturation "
                 "predictor: uniform per-group budget minus effective-context "
                 "support. **H1: `oracle_gap` collapses to the noise floor "
                 "once `min_margin` crosses ≥ 0** (saturated regime).")
    lines.append("")
    lines.append("| tier | ratio | min_margin (mean S) | min_margin (max S) | "
                 "oracle_gap | worst_gap | regime |")
    lines.append("|---:|---:|---:|---:|---:|---:|:--|")
    for row in rows:
        margin_mean = row["mean_min_margin_mean"]
        regime = "—"
        if margin_mean is not None:
            regime = "saturated" if margin_mean >= 0 else "unsaturated"
        lines.append(
            f"| {row['tier_length']} | {row['ratio']:.4f} | "
            f"{_fmt(margin_mean)} | {_fmt(row['mean_min_margin_max'])} | "
            f"{_fmt(row['mean_oracle_gap_per_step'])} | "
            f"{_fmt(row['mean_best_vs_worst_per_step'])} | {regime} |"
        )
    lines.append("")
    lines.append("Crossing read (spec §9): the gate passes if `oracle_gap` is "
                 "material in `unsaturated` cells, near the noise floor in "
                 "`saturated` cells, and monotone in `min_margin` across the "
                 "pooled cells.")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="longcontext-decode-probe-analyze")
    p.add_argument("--in-dir", type=Path, required=True,
                   help="Analyses dir containing trajectories.json + eval_config.json.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write stats.json + report.md (defaults to --in-dir).")
    p.add_argument("--support-estimator", default=None,
                   choices=["mass_coverage", "participation_ratio", "entropy"],
                   help="Pick this estimator from effective_support_grid for the "
                        "saturation predictor (default: the producer's primary). "
                        "Lets the §9 predictor be re-calibrated without re-running "
                        "the GPU producer.")
    p.add_argument("--support-tau", type=float, default=None,
                   help="Mass-coverage tau to pick from the grid (mass_coverage only).")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    in_dir = args.in_dir
    out_dir = args.out_dir if args.out_dir is not None else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = in_dir / "trajectories.json"
    if not trajectories_path.exists():
        raise FileNotFoundError(f"{trajectories_path} not found")
    data = json.loads(trajectories_path.read_text(encoding="utf-8"))
    num_groups = int(data["num_groups"])
    n_sink = int(data["n_sink"])
    n_recent = int(data["n_recent"])

    per_prompt = []
    for prompt in data["prompts"]:
        if prompt.get("status") != "ok":
            continue
        per_prompt.append(analyze_prompt(
            prompt, num_groups, n_sink, n_recent,
            support_estimator=args.support_estimator,
            support_tau=args.support_tau,
        ))

    # Which (estimator, tau) the saturation predictor actually used: the
    # analyzer override if given, else the producer's primary.
    if args.support_estimator is not None:
        used_estimator = args.support_estimator
        used_tau = args.support_tau if args.support_estimator == "mass_coverage" else None
    else:
        used_estimator = data.get("support_estimator")
        used_tau = data.get("support_tau")

    aggregate_rows = aggregate(per_prompt)
    stats = {
        "manifest": data.get("manifest"),
        "model": data.get("model"),
        "k_dec": data.get("k_dec"),
        "obs_window": data.get("obs_window"),
        "num_groups": num_groups,
        "n_sink": n_sink,
        "n_recent": n_recent,
        "ratios": data.get("ratios"),
        "support_estimator": used_estimator,
        "support_tau": used_tau,
        "priority": data.get("priority"),
        "per_prompt": per_prompt,
        "aggregate_per_tier_ratio": aggregate_rows,
    }

    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(stats), encoding="utf-8")
    print(f"[lc-analyze] wrote {out_dir}/stats.json + report.md "
          f"({len(per_prompt)} prompts, {len(aggregate_rows)} (tier,ratio) cells)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
