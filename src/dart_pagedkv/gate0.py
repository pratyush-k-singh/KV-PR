"""Gate-0 metrics: Spearman rank correlation, per-bucket signs, full report.

The Stage-0 bridge correlates the formal cost (cold attention demand) with a
teacher-forced fidelity rung across a predeclared budget-vector spread. A
non-trivial positive Spearman rho, positive across most buckets and not
inverted on any of them, is the precondition for proceeding to Stage 1.
Numeric thresholds (default ``rho >= 0.45``) are pre-registered before each
run, never tuned post-hoc.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, with NaN -> 0 and n<2 -> 0 fallbacks."""
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    if arr_x.shape != arr_y.shape:
        raise ValueError(f"shape mismatch: {arr_x.shape} vs {arr_y.shape}")
    if arr_x.size < 2:
        return 0.0
    rho, _ = spearmanr(arr_x, arr_y)
    value = float(rho)
    if not np.isfinite(value):
        return 0.0
    return value


def bucket_signs(
    cost: Sequence[float],
    fidelity: Sequence[float],
    buckets: Sequence[str],
) -> dict[str, float]:
    """Per-bucket Spearman rho. Buckets with fewer than 2 elements yield 0.0."""
    cost_arr = np.asarray(cost, dtype=np.float64)
    fidelity_arr = np.asarray(fidelity, dtype=np.float64)
    bucket_arr = np.asarray([str(b) for b in buckets])
    if not (cost_arr.shape == fidelity_arr.shape == bucket_arr.shape):
        raise ValueError(
            f"shape mismatch: cost {cost_arr.shape}, fidelity "
            f"{fidelity_arr.shape}, buckets {bucket_arr.shape}"
        )

    result: dict[str, float] = {}
    for label in sorted(set(bucket_arr.tolist())):
        mask = bucket_arr == label
        if int(mask.sum()) < 2:
            result[label] = 0.0
            continue
        result[label] = spearman_rho(cost_arr[mask], fidelity_arr[mask])
    return result


def gate0_report(
    cost: Sequence[float],
    fidelity: Sequence[float],
    buckets: Sequence[str] | None = None,
    *,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Overall Spearman + optional per-bucket signs + pass-vs-threshold."""
    overall = spearman_rho(cost, fidelity)
    report: dict[str, Any] = {
        "n": int(len(cost)),
        "overall_rho": overall,
        "threshold": float(threshold),
        "overall_pass": bool(overall >= float(threshold)),
    }
    if buckets is not None:
        signs = bucket_signs(cost, fidelity, buckets)
        n_buckets = max(len(signs), 1)
        report["bucket_rhos"] = signs
        report["positive_buckets"] = sum(1 for v in signs.values() if v > 0)
        report["negative_buckets"] = sum(1 for v in signs.values() if v < 0)
        report["zero_buckets"] = sum(1 for v in signs.values() if v == 0)
        report["positive_fraction"] = report["positive_buckets"] / n_buckets
    return report
