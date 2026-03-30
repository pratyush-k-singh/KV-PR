#!/usr/bin/env python3
"""Analyze the P24a (Qwen GQA-emulation) and P28 (TOVA) results jointly.

P24a question: does *more* KV-head sharing on a drift-regime architecture
(Qwen3-1.7B, GQA-8) flip it to the bias regime, or does the regime stay
fixed while alpha grows monotonically as it did on Phi P24b?

P28 question: does TOVA (last-query-row priority) match the snapkv pattern
on bias-regime Phi (catastrophic) and the random-like pattern on drift-
regime LLaMA/Qwen (cheap)? This is the operational version of the
Proposition-1 prediction: any priority biasing the retained-set centroid
in value-space relative to random should behave like snapkv on bias-
dominated architectures.

Writes:
  docs/paper/p24a_qwen_gqa_intervention.md
  docs/paper/p28_tova_sweep.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PULL = ROOT / "analyses" / "2026-05-21_p24a_p28_p20_pull"
OUT_P24A = HERE / "p24a_qwen_gqa_intervention.md"
OUT_P28 = ROOT / "docs" / "paper" / "p28_tova_sweep.md"


def is_uniform(a):
    return a and (max(a) - min(a)) <= 1


def fit_alpha_beta(per_step_mean: np.ndarray):
    """Linear fit y = alpha + beta * t on t = 1..15."""
    t = np.arange(1, len(per_step_mean))
    y = per_step_mean[1:]
    A = np.vstack([np.ones_like(t, dtype=float), t.astype(float)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def load_cell_trajectories(dir_path: Path):
    """Return (per_ratio_arrays, eval_config) for one cell.
    per_ratio_arrays: {ratio: list of per_step_kl arrays for uniform allocs}."""
    full = dir_path / "trajectories.json"
    if not full.exists():
        return None, None
    data = json.load(open(full))
    by_ratio: dict[float, list[list[float]]] = {}
    for prompt in data.get("prompts", []):
        if prompt.get("status") != "ok":
            continue
        for t in prompt.get("trajectories", []):
            if not is_uniform(t.get("allocation", [])):
                continue
            ps = t.get("per_step_kl")
            if not ps or len(ps) < 16:
                continue
            by_ratio.setdefault(t["ratio"], []).append(ps)
    return by_ratio, data


def summarize_cell(by_ratio):
    """Return {ratio: {alpha, beta, mean_kl, n}}."""
    out = {}
    for r, arrs in by_ratio.items():
        M = np.array(arrs)
        m = M.mean(axis=0)
        alpha, beta = fit_alpha_beta(m)
        out[round(r, 4)] = {
            "n": len(M),
            "alpha": alpha,
            "beta": beta,
            "mean_kl": float(m[1:].mean()),
        }
    return out


def analyze_p24a():
    factors = [1, 2, 4, 8]
    priorities = ["snapkv", "random"]
    rows: dict[tuple, dict] = {}
    for pri in priorities:
        for f in factors:
            target = (
                f"2026-05-21_p24a_gqa_qwen_s43_n2_taskshard_v1_p24a_qwen3_1p7b_"
                f"{pri}_n2_g{f}_l32768_merged"
            )
            by_ratio, _ = load_cell_trajectories(PULL / target)
            if by_ratio is None:
                print(f"[p24a] MISSING {target}")
                continue
            for r, summary in summarize_cell(by_ratio).items():
                rows[(pri, f, r)] = summary
                print(
                    f"{pri:6} f={f} r={r:.2f}  n={summary['n']:3d}  "
                    f"alpha={summary['alpha']:.4f}  beta={summary['beta']:.5f}  "
                    f"meanKL={summary['mean_kl']:.4f}"
                )
    print()

    lines = [
        "# P24a: GQA-emulation on Qwen3-1.7B (L4/L5 symmetric to P24b)\n",
        "Symmetric to P24b's intervention on Phi: the same `gqa_emulation.py` "
        "hooks are applied to Qwen3-1.7B, averaging its 8 KV heads in groups "
        "of `factor`. Factor=1 reproduces the unmodified Qwen-GQA-2 (its "
        "natural ratio of 16 query heads to 8 KV heads); factor=2/4/8 force "
        "additional KV-head sharing.\n",
        "**Question.** If GQA absence were the architectural feature placing "
        "Phi in the bias regime, *more* KV-head sharing on a drift-regime "
        "architecture should leave it drift-dominated (since it is *gaining* "
        "the structural feature implicated). If, on the contrary, more "
        "KV-head sharing pushes Qwen into the bias regime, GQA presence is "
        "what places LLaMA/Qwen in the drift regime, contradicting the P24b "
        "conclusion. The paper-consistent prediction is no regime flip; "
        "alpha grows monotonically as it did on Phi (the bias channel widens "
        "in both architectures with more sharing) but beta stays in its "
        "drift-regime band.\n",
        "## Mean Masked-Decode KL by Factor and Ratio\n",
        "| factor | ratio | snapkv mean KL | random mean KL | snap-rand gap | snap/rand ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    ratios = sorted({k[2] for k in rows})
    for f in factors:
        for r in ratios:
            if ("snapkv", f, r) not in rows or ("random", f, r) not in rows:
                continue
            s = rows[("snapkv", f, r)]
            R = rows[("random", f, r)]
            gap = s["mean_kl"] - R["mean_kl"]
            ratio = s["mean_kl"] / R["mean_kl"] if R["mean_kl"] > 0 else float("nan")
            lines.append(
                f"| {f} | {r:.2f} | {s['mean_kl']:.4f} | {R['mean_kl']:.4f} | "
                f"{gap:+.4f} | {ratio:.2f}x |"
            )
    lines.append("")

    lines.append("## alpha (bias coefficient) and beta (drift coefficient)\n")
    lines.append(
        "Linear fit `KL_pi(t) = alpha + beta * t` across masked-decode steps "
        "1..15, cross-prompt mean.\n"
    )
    lines.append("| factor | ratio | snapkv alpha | snapkv beta | random alpha | random beta | alpha gap |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for f in factors:
        for r in ratios:
            if ("snapkv", f, r) not in rows or ("random", f, r) not in rows:
                continue
            s = rows[("snapkv", f, r)]
            R = rows[("random", f, r)]
            ag = s["alpha"] - R["alpha"]
            lines.append(
                f"| {f} | {r:.2f} | {s['alpha']:.4f} | {s['beta']:.5f} | "
                f"{R['alpha']:.4f} | {R['beta']:.5f} | {ag:+.4f} |"
            )
    lines.append("")

    # Verdict on the regime: drift-dominated cells have beta > some threshold
    # while bias-dominated have beta ~ 0 with large alpha. Cross-cell read.
    if ("snapkv", 1, 0.04) in rows:
        baseline_beta_snap = rows[("snapkv", 1, 0.04)]["beta"]
        baseline_alpha_snap = rows[("snapkv", 1, 0.04)]["alpha"]
        eight_beta_snap = rows[("snapkv", 8, 0.04)]["beta"]
        eight_alpha_snap = rows[("snapkv", 8, 0.04)]["alpha"]
        lines.append("## Headline at r=0.04 (the tightest budget)\n")
        lines.append(
            f"- snapkv alpha at factor=1: {baseline_alpha_snap:.4f}, beta: {baseline_beta_snap:.5f}"
        )
        lines.append(
            f"- snapkv alpha at factor=8: {eight_alpha_snap:.4f}, beta: {eight_beta_snap:.5f}"
        )
        lines.append("")

    OUT_P24A.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[p24a] wrote {OUT_P24A}")
    return rows


def analyze_p28():
    """TOVA priority on three models across four tier lengths.
    Compare to existing snapkv / random / accumulated data from the paper's
    central DB. The headline questions:
      1. Does TOVA match snapkv on Phi (bias regime) -> catastrophic?
      2. Does TOVA behave like random/snapkv on LLaMA/Qwen (drift regime)?
      3. Where does TOVA fit in the selector's two-priority decision?
    """
    models = [("phi35", "Phi-3.5-mini"), ("llama32_1b", "LLaMA-3.2-1B"),
              ("qwen3_1p7b", "Qwen3-1.7B")]
    tiers = [2048, 8192, 32768, 131072]
    rows: dict[tuple, dict] = {}
    for slug, _ in models:
        for tier in tiers:
            target = (
                f"2026-05-21_p28_tova_s43_taskshard_v1_p28_{slug}_tova_"
                f"l{tier}_merged"
            )
            by_ratio, _ = load_cell_trajectories(PULL / target)
            if by_ratio is None:
                print(f"[p28] MISSING {target}")
                continue
            for r, summary in summarize_cell(by_ratio).items():
                rows[(slug, tier, r)] = summary
                print(
                    f"{slug:12} tier={tier:6} r={r:.2f}  n={summary['n']:3d}  "
                    f"alpha={summary['alpha']:.4f}  beta={summary['beta']:.5f}  "
                    f"meanKL={summary['mean_kl']:.4f}"
                )
    print()

    lines = [
        "# P28: TOVA priority across 3 models x 4 tier lengths (L7 partial close)\n",
        "TOVA = SnapKV with `obs_window = 1` (last-query row only). Pull at "
        "`analyses/2026-05-21_p24a_p28_p20_pull/...p28_*_tova_l*_merged/`.\n",
        "**Question.** Proposition 1 predicts that any priority whose "
        "construction biases the retained-set centroid in value-space "
        "relative to random selection should match snapkv's pattern on "
        "bias-dominated architectures. TOVA's column-sum aggregation uses "
        "the last query row only -- conceptually narrower than snapkv's "
        "32-row window but still attention-derived, so the same retained-"
        "set biasing applies. If TOVA on Phi is catastrophic like snapkv, "
        "the mechanism's prediction holds for a new priority. If TOVA on "
        "Phi is cheap like random, the mechanism is more priority-specific "
        "than the paper claims.\n",
        "## Mean masked-decode KL per (model, tier, ratio)\n",
        "Cross-prompt mean over uniform allocations.\n",
        "| model | tier | r=0.04 | r=0.08 | r=0.16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for slug, name in models:
        for tier in tiers:
            cells = {r: rows[(slug, tier, r)]
                     for r in (0.04, 0.08, 0.16)
                     if (slug, tier, r) in rows}
            if not cells:
                continue
            line = f"| {name} | {tier} |"
            for r in (0.04, 0.08, 0.16):
                if r in cells:
                    line += f" {cells[r]['mean_kl']:.4f} (n={cells[r]['n']}) |"
                else:
                    line += " - |"
            lines.append(line)
    lines.append("")

    lines.append("## alpha and beta per (model, tier) at r=0.04\n")
    lines.append("Tightest budget; the bias coefficient is regime-defining.\n")
    lines.append("| model | tier | n | alpha | beta | mean KL |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for slug, name in models:
        for tier in tiers:
            if (slug, tier, 0.04) not in rows:
                continue
            s = rows[(slug, tier, 0.04)]
            lines.append(
                f"| {name} | {tier} | {s['n']} | {s['alpha']:.4f} | "
                f"{s['beta']:.5f} | {s['mean_kl']:.4f} |"
            )
    lines.append("")

    OUT_P28.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[p28] wrote {OUT_P28}")
    return rows


if __name__ == "__main__":
    print("=== P24a Qwen GQA-emulation ===")
    p24a = analyze_p24a()
    print("\n=== P28 TOVA sweep ===")
    p28 = analyze_p28()
