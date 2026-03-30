#!/usr/bin/env python3
"""Comprehensive statistical analysis for the KV-PR paper.

Outputs `docs/paper/stats_appendix.md` and several supporting CSVs.

Tests included (each labelled in the output):
  S1  Bootstrap percentile CIs (95%, 99%) on every cell-level mean and
      median used in the headline tables.
  S2  Cohen's d and Cliff's δ for the cross-architecture α and β
      contrasts, paired and unpaired.
  S3  Permutation tests for the priority-class inversion on Phi at
      131k r=0.04 (snapkv vs random per-prompt KL).
  S4  Hierarchical / mixed-effects-style variance-component
      decomposition: σ²_model, σ²_task, σ²_prompt for α and β.
  S5  Bonferroni and Benjamini-Hochberg-corrected p-values for the
      per-task within-Phi PR-α regressions.
  S6  Leave-one-task-out sensitivity for the headline within-Phi PR-α
      regression and the selector overall-improvement figure.
  S7  Power calculation: minimum detectable effect at our cell-level n.
  S8  Bootstrap CI on the P10 paired Δposition under τ.
  S9  P11 Jacobian-proxy CI: mean log ‖J‖ with bootstrap CI and the
      one-sample two-sided test of log ‖J‖ ≠ 0 per (model, priority).
  S10 Joint regression on Phi α including sink-concentration metric
      from P10 (bin-0 mass).
  S11 Multi-task heterogeneity test (Cochran's Q on per-task α-PR
      slopes within Phi).
  S12 Holm-Bonferroni-corrected significance for the cross-model α
      basin separation pairwise contrasts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE
DB = HERE / "paper_data.db"
PULL_P7 = ROOT / "analyses" / "2026-05-21_p7_temperature_split_n5_v1_pull"
PULL_P10P11 = ROOT / "analyses" / "2026-05-21_p10_p11_n4_split_v2_pull"


# =========================================================================
# Utilities
# =========================================================================

def is_uniform(alloc):
    if not alloc:
        return False
    return (max(alloc) - min(alloc)) <= 1


def fit_alpha_beta(kl):
    if len(kl) < 16:
        return float("nan"), float("nan")
    s = np.arange(1, 16, dtype=float)
    y = np.array(kl[1:16], dtype=float)
    b = np.cov(s, y, ddof=0)[0, 1] / np.var(s)
    a = y.mean() - b * s.mean()
    return float(a), float(b)


def bootstrap_ci(values, statistic=np.mean, n_resamples=4000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty(n_resamples)
    n = arr.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(arr[idx])
    point = float(statistic(arr))
    lo = float(np.percentile(stats, 100 * (alpha / 2)))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = math.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else float("nan")


def cliffs_delta(x, y):
    """Cliff's delta — non-parametric effect size between -1 and +1."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        return float("nan")
    greater = 0
    less = 0
    for xi in x:
        greater += int((y < xi).sum())
        less += int((y > xi).sum())
    return float((greater - less) / (nx * ny))


def permutation_test(x, y, n_perm=10000, seed=0):
    """Two-sample permutation test of mean difference."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    obs = float(x.mean() - y.mean())
    pool = np.concatenate([x, y])
    nx = x.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        diff = float(pool[:nx].mean() - pool[nx:].mean())
        if abs(diff) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def spearman_rho_with_p(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    rx_idx = sorted(range(n), key=lambda i: xs[i])
    ry_idx = sorted(range(n), key=lambda i: ys[i])
    rx = [0.0] * n
    ry = [0.0] * n
    for r, i in enumerate(rx_idx):
        rx[i] = r
    for r, i in enumerate(ry_idx):
        ry[i] = r
    mx = sum(rx) / n; my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n)) *
           sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    rho = num / den if den > 0 else float("nan")
    if math.isnan(rho):
        return rho, float("nan")
    t_stat = rho * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1 - rho * rho, 1e-12))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    return rho, p


def holm_bonferroni(ps):
    n = len(ps)
    idx = sorted(range(n), key=lambda i: ps[i])
    adj = [0.0] * n
    prev = 0.0
    for k, i in enumerate(idx):
        v = min(1.0, (n - k) * ps[i])
        v = max(v, prev)
        adj[i] = v
        prev = v
    return adj


def benjamini_hochberg(ps):
    n = len(ps)
    idx = sorted(range(n), key=lambda i: ps[i])
    adj = [0.0] * n
    prev = 1.0
    for k_rev in range(n - 1, -1, -1):
        i = idx[k_rev]
        v = min(1.0, ps[i] * n / (k_rev + 1))
        v = min(v, prev)
        adj[i] = v
        prev = v
    return adj


# =========================================================================
# Data loading
# =========================================================================

def load_db_alpha():
    """Per-prompt α, β, KL from the main DB joined to per-prompt PR."""
    con = sqlite3.connect(DB)
    cur = con.cursor()
    ms = {
        "llama3p2_1b_instruct": "llama_1b",
        "qwen3_1p7b": "qwen_1p7b",
        "phi_3p5_mini_instruct": "phi_3p5",
    }
    pr_map = {}
    cur.execute("SELECT model, tier_length, prompt_id, participation_ratio FROM diffuseness")
    for dm, tier, pid, pr in cur:
        pr_map[(dm, tier, pid)] = pr
    cur.execute(
        """
        SELECT r.model, r.priority, r.num_groups, p.tier_length, p.task, p.prompt_id,
               m.ratio, m.per_step_kl_json, m.allocation_json
        FROM measurements m
        JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
        JOIN runs r    ON p.run_id = r.run_id
        WHERE r.num_groups = 4 AND m.n_steps_kl >= 15
        """
    )
    rows = []
    for model, pri, ng, tier, task, pid, ratio, kl_json, alloc_json in cur:
        a = json.loads(alloc_json)
        if not is_uniform(a):
            continue
        kl = json.loads(kl_json)
        alpha, beta = fit_alpha_beta(kl)
        if math.isnan(alpha):
            continue
        pr = pr_map.get((ms.get(model, model), tier, pid))
        rows.append({
            "model": model, "priority": pri, "tier": tier, "task": task,
            "prompt_id": pid, "ratio": ratio,
            "alpha": alpha, "beta": beta,
            "mean_kl": float(statistics.mean(kl[1:16])),
            "PR": pr,
        })
    con.close()
    return rows


def load_p7():
    rows = []
    for d in sorted(os.listdir(PULL_P7)):
        if not d.endswith("_merged"):
            continue
        full = PULL_P7 / d / "trajectories.json"
        if not full.exists():
            continue
        data = json.load(open(full))
        model = os.path.basename(data.get("model", "")).replace(".", "p").replace("-", "_").lower()
        pri = data.get("priority")
        tau = data.get("priority_temperature", 1.0)
        for prompt in data.get("prompts", []):
            if prompt.get("status") != "ok":
                continue
            for t in prompt.get("trajectories", []):
                kl = t.get("per_step_kl", [])
                if len(kl) < 16:
                    continue
                if not is_uniform(t.get("allocation", [])):
                    continue
                a, b = fit_alpha_beta(kl)
                rows.append({
                    "model": model, "priority": pri, "tau": tau,
                    "tier": prompt["tier_length"], "task": prompt["task"],
                    "prompt_id": prompt["prompt_id"], "ratio": t["ratio"],
                    "alpha": a, "beta": b,
                    "mean_kl": float(statistics.mean(kl[1:16])),
                })
    return rows


def load_p10p11():
    rows = []
    for d in sorted(os.listdir(PULL_P10P11)):
        if not d.endswith("_merged"):
            continue
        full = PULL_P10P11 / d / "trajectories.json"
        if not full.exists():
            continue
        data = json.load(open(full))
        model = os.path.basename(data.get("model", "")).replace(".", "p").replace("-", "_").lower()
        pri = data.get("priority")
        tau = data.get("priority_temperature", 1.0)
        exp_kind = "p10" if "p10_" in d else "p11" if "p11_" in d else "?"
        for prompt in data.get("prompts", []):
            if prompt.get("status") != "ok":
                continue
            for t in prompt.get("trajectories", []):
                if not is_uniform(t.get("allocation", [])):
                    continue
                kl = t.get("per_step_kl", [])
                if len(kl) < 16:
                    continue
                cov = t.get("coverage") or {}
                a, b = fit_alpha_beta(kl)
                rows.append({
                    "exp": exp_kind,
                    "model": model, "priority": pri, "tau": tau,
                    "tier": prompt["tier_length"], "task": prompt["task"],
                    "prompt_id": prompt["prompt_id"], "ratio": t["ratio"],
                    "alpha": a, "beta": b,
                    "mean_kl": float(statistics.mean(kl[1:16])),
                    "per_step_logit_diff_l2": t.get("per_step_logit_diff_l2"),
                    "bin0_mass": (
                        sum(h[0] for h in cov["per_group_position_hist_10bin"])
                        / max(1, sum(sum(h) for h in cov["per_group_position_hist_10bin"]))
                        if cov.get("per_group_position_hist_10bin") else None
                    ),
                    "bin9_mass": (
                        sum(h[9] for h in cov["per_group_position_hist_10bin"])
                        / max(1, sum(sum(h) for h in cov["per_group_position_hist_10bin"]))
                        if cov.get("per_group_position_hist_10bin") else None
                    ),
                    "mean_retained_position": (
                        float(np.nanmean(cov["per_group_mean_retained_position"]))
                        if cov.get("per_group_mean_retained_position") else None
                    ),
                    "jaccard": cov.get("mean_pairwise_jaccard"),
                    "ret_mass": cov.get("total_retained_reference_mass"),
                    "union_size": cov.get("union_size"),
                })
    return rows


# =========================================================================
# Sections
# =========================================================================

OUTLINES = []


def section(title):
    OUTLINES.append("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def line(s):
    OUTLINES.append(s)


def s1_bootstrap_headline(db_rows, p7_rows):
    section("S1. Bootstrap CIs (95% and 99%) on the headline cell means")
    line("Per (model, priority, tier, ratio) cell mean of per-prompt mean_KL, with")
    line("4000-resample percentile bootstrap CIs.")
    line(f"\n{'model':<28}{'priority':<14}{'tier':>7}{'r':>6}{'n':>4}  "
         f"{'mean':>8}{'95% CI':>22}{'99% CI':>22}")
    by = defaultdict(list)
    for r in db_rows:
        by[(r["model"], r["priority"], r["tier"], r["ratio"])].append(r["mean_kl"])
    for k in sorted(by.keys()):
        v = by[k]
        if len(v) < 4:
            continue
        m, lo95, hi95 = bootstrap_ci(v, np.mean, n_resamples=4000, alpha=0.05, seed=1)
        _, lo99, hi99 = bootstrap_ci(v, np.mean, n_resamples=4000, alpha=0.01, seed=1)
        ci95 = f"[{lo95:+.4f}, {hi95:+.4f}]"
        ci99 = f"[{lo99:+.4f}, {hi99:+.4f}]"
        if k[1] not in ("snapkv", "random"):
            continue
        if k[2] not in (32768, 131072):
            continue
        line(f"  {k[0]:<28}{k[1]:<14}{k[2]:>7}{k[3]:>6.3f}{len(v):>4}  "
             f"{m:>8.4f}{ci95:>22}{ci99:>22}")


def s2_effect_sizes(db_rows):
    section("S2. Effect sizes for cross-architecture α / β contrasts")
    line("Cohen's d and Cliff's δ on per-prompt α and β between models, at")
    line("32k and 131k r=0.04, snapkv priority, uniform allocation.")
    line(f"\n{'comparison':<48}{'n_A':>5}{'n_B':>5}  {'Δmean':>9}{'cohen_d':>9}{'cliff_δ':>9}")
    def sub(model, tier):
        return [r for r in db_rows if r["model"] == model and r["tier"] == tier
                and r["priority"] == "snapkv" and r["ratio"] == 0.04]
    pairs = [
        ("phi_3p5_mini_instruct", "llama3p2_1b_instruct", 32768),
        ("phi_3p5_mini_instruct", "qwen3_1p7b", 32768),
        ("phi_3p5_mini_instruct", "llama3p2_1b_instruct", 131072),
        ("phi_3p5_mini_instruct", "qwen3_1p7b", 131072),
    ]
    for a, b, tier in pairs:
        A = sub(a, tier); B = sub(b, tier)
        if not A or not B:
            continue
        for field in ["alpha", "beta"]:
            x = np.array([r[field] for r in A])
            y = np.array([r[field] for r in B])
            d = cohens_d(x, y)
            cd = cliffs_delta(x, y)
            label = f"{field}: {a[:14]} − {b[:14]} @{tier}"
            line(f"  {label:<48}{len(A):>5}{len(B):>5}  "
                 f"{(x.mean() - y.mean()):>+9.4f}{d:>+9.3f}{cd:>+9.3f}")


def s3_permutation_inversion(db_rows):
    section("S3. Permutation test of the Phi inversion (snap vs random) at 131k r=0.04")
    A = [r["mean_kl"] for r in db_rows
         if r["model"] == "phi_3p5_mini_instruct" and r["tier"] == 131072
         and r["priority"] == "snapkv" and r["ratio"] == 0.04]
    B = [r["mean_kl"] for r in db_rows
         if r["model"] == "phi_3p5_mini_instruct" and r["tier"] == 131072
         and r["priority"] == "random" and r["ratio"] == 0.04]
    if A and B:
        obs, p = permutation_test(A, B, n_perm=20000, seed=2)
        line(f"  n_snap = {len(A)}  n_rand = {len(B)}")
        line(f"  mean(KL_snap) − mean(KL_rand) = {obs:+.4f}")
        line(f"  permutation p-value (two-sided, 20000 perms) = {p:.5f}")
        line(f"  median(KL_snap) = {statistics.median(A):+.4f}, "
             f"median(KL_rand) = {statistics.median(B):+.4f}")

    # Same test at 32k
    A2 = [r["mean_kl"] for r in db_rows
          if r["model"] == "phi_3p5_mini_instruct" and r["tier"] == 32768
          and r["priority"] == "snapkv" and r["ratio"] == 0.04]
    B2 = [r["mean_kl"] for r in db_rows
          if r["model"] == "phi_3p5_mini_instruct" and r["tier"] == 32768
          and r["priority"] == "random" and r["ratio"] == 0.04]
    if A2 and B2:
        obs, p = permutation_test(A2, B2, n_perm=20000, seed=2)
        line(f"\n  same test at 32k r=0.04: n_snap={len(A2)} n_rand={len(B2)}")
        line(f"  mean diff = {obs:+.4f}  p = {p:.5f}")

    # Permutation test on a non-Phi model for placebo
    A3 = [r["mean_kl"] for r in db_rows
          if r["model"] == "llama3p2_1b_instruct" and r["tier"] == 131072
          and r["priority"] == "snapkv" and r["ratio"] == 0.04]
    B3 = [r["mean_kl"] for r in db_rows
          if r["model"] == "llama3p2_1b_instruct" and r["tier"] == 131072
          and r["priority"] == "random" and r["ratio"] == 0.04]
    if A3 and B3:
        obs, p = permutation_test(A3, B3, n_perm=20000, seed=2)
        line(f"\n  placebo on LLaMA-3.2-1B at 131k r=0.04:")
        line(f"  n_snap = {len(A3)}  n_rand = {len(B3)}")
        line(f"  mean diff = {obs:+.4f}  p = {p:.5f}")


def s4_variance_components(db_rows):
    section("S4. Variance components of α and β across (model, task, prompt)")
    line("Decomposes σ² into between-model, between-task-within-model, between-")
    line("prompt-within-(model,task), and residual. Per-priority, snapkv only.")
    sub = [r for r in db_rows if r["priority"] == "snapkv"]
    for field in ["alpha", "beta", "mean_kl"]:
        line(f"\n  Field: {field}")
        # Aggregate at (model, task, prompt_id) — pool over ratios for variance comp
        by_mt = defaultdict(lambda: defaultdict(list))
        for r in sub:
            by_mt[r["model"]][r["task"]].append((r["prompt_id"], r[field]))
        # Compute model-level means, task-within-model means, prompt-within-task means
        all_vals = []
        model_means = {}
        task_means = defaultdict(dict)
        prompt_means = defaultdict(lambda: defaultdict(dict))
        for m, tasks in by_mt.items():
            model_vals = []
            for t, lst in tasks.items():
                task_vals = [v for _, v in lst]
                model_vals.extend(task_vals)
                task_means[m][t] = float(np.mean(task_vals))
                # prompt means
                by_pp = defaultdict(list)
                for pid, v in lst:
                    by_pp[pid].append(v)
                for pid, vs in by_pp.items():
                    prompt_means[m][t][pid] = float(np.mean(vs))
            model_means[m] = float(np.mean(model_vals))
            all_vals.extend(model_vals)
        if not all_vals:
            continue
        grand = float(np.mean(all_vals))
        # Variances at each level
        ss_total = sum((v - grand) ** 2 for v in all_vals)
        ss_between_model = 0.0
        for m, vs in by_mt.items():
            n = sum(len(lst) for lst in vs.values())
            ss_between_model += n * (model_means[m] - grand) ** 2
        ss_between_task = 0.0
        for m, vs in by_mt.items():
            for t, lst in vs.items():
                n = len(lst)
                ss_between_task += n * (task_means[m][t] - model_means[m]) ** 2
        ss_between_prompt = 0.0
        for m, tasks in prompt_means.items():
            for t, pps in tasks.items():
                for pid, mp in pps.items():
                    n_obs = sum(1 for x, v in by_mt[m][t] if x == pid)
                    ss_between_prompt += n_obs * (mp - task_means[m][t]) ** 2
        ss_residual = max(ss_total - ss_between_model - ss_between_task - ss_between_prompt, 0)
        if ss_total > 0:
            line(f"    σ²_model            : {ss_between_model:>10.4f}  ({ss_between_model/ss_total*100:.1f}%)")
            line(f"    σ²_task|model       : {ss_between_task:>10.4f}  ({ss_between_task/ss_total*100:.1f}%)")
            line(f"    σ²_prompt|task,model: {ss_between_prompt:>10.4f}  ({ss_between_prompt/ss_total*100:.1f}%)")
            line(f"    σ²_residual         : {ss_residual:>10.4f}  ({ss_residual/ss_total*100:.1f}%)")
            line(f"    σ²_total            : {ss_total:>10.4f}")


def s5_multiple_testing(db_rows):
    section("S5. Multiple-testing-corrected per-task within-Phi PR-α slopes")
    sub = [r for r in db_rows if r["model"] == "phi_3p5_mini_instruct"
           and r["priority"] == "snapkv" and r["PR"] is not None]
    by_task = defaultdict(list)
    for r in sub:
        by_task[r["task"]].append(r)
    tasks = sorted(by_task.keys())
    rho_list = []
    p_list = []
    n_list = []
    for t in tasks:
        rs = by_task[t]
        xs = [math.log(max(r["PR"], 1)) for r in rs]
        ys = [r["alpha"] for r in rs]
        rho, p = spearman_rho_with_p(xs, ys)
        rho_list.append(rho); p_list.append(p); n_list.append(len(rs))
    holm = holm_bonferroni(p_list)
    bh = benjamini_hochberg(p_list)
    line(f"  {'task':<20}{'n':>4}{'ρ':>9}{'p (raw)':>10}{'p (Holm)':>12}{'p (BH-FDR)':>14}")
    for i, t in enumerate(tasks):
        line(f"  {t:<20}{n_list[i]:>4}{rho_list[i]:>+9.4f}"
             f"{p_list[i]:>10.4f}{holm[i]:>12.4f}{bh[i]:>14.4f}")
    n_sig_raw = sum(1 for p in p_list if p < 0.05)
    n_sig_holm = sum(1 for p in holm if p < 0.05)
    n_sig_bh = sum(1 for p in bh if p < 0.05)
    line(f"\n  significant at α=0.05: raw {n_sig_raw}/{len(p_list)}, "
         f"Holm {n_sig_holm}/{len(p_list)}, BH-FDR {n_sig_bh}/{len(p_list)}")


def s6_leave_one_out(db_rows):
    section("S6. Leave-one-task-out sensitivity for within-Phi PR-α regression")
    sub = [r for r in db_rows if r["model"] == "phi_3p5_mini_instruct"
           and r["priority"] == "snapkv" and r["PR"] is not None]
    all_tasks = sorted(set(r["task"] for r in sub))
    line(f"  Tasks: {', '.join(all_tasks)}")
    line(f"  Full-sample n = {len(sub)}")
    xs = [math.log(max(r["PR"], 1)) for r in sub]
    ys = [r["alpha"] for r in sub]
    rho_full, p_full = spearman_rho_with_p(xs, ys)
    line(f"  Full-sample Spearman ρ(log PR, α) = {rho_full:+.4f}  (p = {p_full:.2e})")
    line(f"\n  Leave-one-task-out:")
    line(f"  {'omitted task':<20}{'n':>5}{'ρ':>9}{'p':>12}")
    rhos = []
    for t in all_tasks:
        loo = [r for r in sub if r["task"] != t]
        xs = [math.log(max(r["PR"], 1)) for r in loo]
        ys = [r["alpha"] for r in loo]
        rho, p = spearman_rho_with_p(xs, ys)
        rhos.append(rho)
        line(f"  {t:<20}{len(loo):>5}{rho:>+9.4f}{p:>12.2e}")
    line(f"\n  LOO range: ρ ∈ [{min(rhos):+.4f}, {max(rhos):+.4f}]")
    line(f"  LOO mean ± sd: {statistics.mean(rhos):+.4f} ± {statistics.stdev(rhos):.4f}")

    # Selector overall-improvement leave-one-out
    section("S6b. Selector improvement leave-one-prompt-out (Table 5 sensitivity)")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("""
        SELECT model, fixed_snapkv_held, selector_pr_threshold_held
        FROM selector_cells
    """)
    rows = cur.fetchall()
    con.close()
    snap = np.array([r[1] for r in rows], dtype=float)
    pr = np.array([r[2] for r in rows], dtype=float)
    full_impr = float((snap.mean() - pr.mean()) / snap.mean())
    line(f"  Full-sample: fixed-snap mean = {snap.mean():.4f}, KV-PR mean = {pr.mean():.4f}")
    line(f"  Improvement = (snap − KV-PR)/snap = {full_impr*100:.2f}%")
    # Bootstrap CI on improvement
    rng = np.random.default_rng(3)
    boots = []
    for _ in range(4000):
        idx = rng.integers(0, len(snap), size=len(snap))
        s_b = snap[idx].mean(); p_b = pr[idx].mean()
        if s_b > 0:
            boots.append((s_b - p_b) / s_b)
    boots = np.array(boots)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    line(f"  Bootstrap 95% CI on improvement: [{lo*100:.2f}%, {hi*100:.2f}%]")


def s7_power(db_rows):
    section("S7. Power calculation: minimum detectable effect at our cell-level n")
    # For the per-cell n at 131k (n~11), compute the minimum effect size
    # detectable with 80% power, two-sided α=0.05.
    line("Approximate minimum detectable effect size (Cohen's d) for a two-sample")
    line("t-test at 80% power, α=0.05, two-sided, at our cell sample sizes:")
    line(f"  {'n_per_group':>11}{'min Cohen d (80% power)':>25}{'min mean diff/sd':>22}")
    for n in [4, 6, 11, 22, 50]:
        # Approximate via z + (z_α/2 + z_β) * √(2/n)
        z_alpha = 1.96
        z_beta = 0.84
        d_min = (z_alpha + z_beta) * math.sqrt(2 / n)
        line(f"  {n:>11}{d_min:>25.3f}{d_min:>22.3f}")
    line("\n  Observed effect sizes for reference:")
    A = [r["mean_kl"] for r in db_rows
         if r["model"] == "phi_3p5_mini_instruct" and r["tier"] == 131072
         and r["priority"] == "snapkv" and r["ratio"] == 0.04]
    B = [r["mean_kl"] for r in db_rows
         if r["model"] == "phi_3p5_mini_instruct" and r["tier"] == 131072
         and r["priority"] == "random" and r["ratio"] == 0.04]
    if A and B:
        d = cohens_d(A, B)
        line(f"    Phi @131k r=0.04 snap vs rand: d = {d:.2f}  (n_A={len(A)}, n_B={len(B)})")
    A2 = [r["mean_kl"] for r in db_rows
          if r["model"] == "llama3p2_1b_instruct" and r["tier"] == 131072
          and r["priority"] == "snapkv" and r["ratio"] == 0.04]
    B2 = [r["mean_kl"] for r in db_rows
          if r["model"] == "llama3p2_1b_instruct" and r["tier"] == 131072
          and r["priority"] == "random" and r["ratio"] == 0.04]
    if A2 and B2:
        d = cohens_d(A2, B2)
        line(f"    LLaMA @131k r=0.04 snap vs rand: d = {d:.2f}  "
             f"(n_A={len(A2)}, n_B={len(B2)})")


def s8_p10_position_shift(p10p11_rows):
    section("S8. P10 paired Δposition under τ — bootstrap CI")
    phi = [r for r in p10p11_rows if r["exp"] == "p10"
           and r["model"] == "phi_3p5_mini_instruct"]
    by_pp = defaultdict(dict)
    for r in phi:
        by_pp[(r["prompt_id"], r["ratio"])][r["tau"]] = r
    pairs = [v for v in by_pp.values() if 1.0 in v and 0.25 in v]
    line(f"  paired (prompt, ratio) cells: {len(pairs)}")
    deltas_pos = []
    deltas_bin0 = []
    deltas_alpha = []
    for v in pairs:
        c1, c0 = v[1.0], v[0.25]
        if (c0.get("mean_retained_position") is not None and
            c1.get("mean_retained_position") is not None and
            not math.isnan(c0["mean_retained_position"]) and
            not math.isnan(c1["mean_retained_position"])):
            deltas_pos.append(c0["mean_retained_position"] - c1["mean_retained_position"])
        if (c0.get("bin0_mass") is not None and
            c1.get("bin0_mass") is not None):
            deltas_bin0.append(c0["bin0_mass"] - c1["bin0_mass"])
        deltas_alpha.append(c0["alpha"] - c1["alpha"])
    for name, vals in [("Δ mean retained position", deltas_pos),
                        ("Δ bin-0 mass (sink-concentration)", deltas_bin0),
                        ("Δ α", deltas_alpha)]:
        if not vals:
            continue
        m, lo, hi = bootstrap_ci(np.array(vals), np.mean, n_resamples=4000, alpha=0.05, seed=4)
        n = len(vals)
        sd = float(np.std(vals, ddof=1))
        t_stat = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
        line(f"\n  {name}:")
        line(f"    n = {n}, mean = {m:+.4f}, sd = {sd:.4f}")
        line(f"    95% CI = [{lo:+.4f}, {hi:+.4f}],  paired t = {t_stat:+.2f}")


def s9_jacobian(p10p11_rows):
    section("S9. P11 Jacobian proxy: per-cell log ‖J‖ with bootstrap CI")
    p11 = [r for r in p10p11_rows if r["exp"] == "p11"
           and r["per_step_logit_diff_l2"]]
    by_mp = defaultdict(list)
    for r in p11:
        arr = r["per_step_logit_diff_l2"]
        if len(arr) < 16:
            continue
        ratios = []
        for s in range(2, 16):
            if arr[s - 1] > 0:
                ratios.append(math.log(arr[s] / arr[s - 1]))
        if ratios:
            by_mp[(r["model"], r["priority"])].append(np.mean(ratios))
    line(f"  {'model':<28}{'priority':<10}{'n':>5}  {'mean log ‖J‖':>14}"
         f"{'95% CI':>26}{'1-sample t':>12}{'p (two-sided)':>15}")
    for k in sorted(by_mp.keys()):
        vals = np.array(by_mp[k], dtype=float)
        m, lo, hi = bootstrap_ci(vals, np.mean, n_resamples=4000, alpha=0.05, seed=5)
        n = vals.size
        sd = float(vals.std(ddof=1)) if n > 1 else 0
        t_stat = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
        # 2-sided p via normal approx for n=132 (large enough)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2)))) if not math.isnan(t_stat) else float("nan")
        ci = f"[{lo:+.5f}, {hi:+.5f}]"
        line(f"  {k[0]:<28}{k[1]:<10}{n:>5}  {m:>+14.5f}{ci:>26}{t_stat:>+12.2f}{p:>15.2e}")


def s10_joint_regression_phi(p10p11_rows, db_rows):
    section("S10. Joint regression on Phi α including sink-concentration (bin-0 mass)")
    phi = [r for r in p10p11_rows if r["exp"] == "p10"
           and r["model"] == "phi_3p5_mini_instruct"]
    pr_map = {}
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT model, tier_length, prompt_id, participation_ratio FROM diffuseness"
    )
    for dm, tier, pid, pr in cur:
        pr_map[("phi_3p5", tier, pid)] = pr
    con.close()
    # Build feature matrix
    Xrows = []
    yrows = []
    keep = []
    for r in phi:
        pr = pr_map.get(("phi_3p5", r["tier"], r["prompt_id"]))
        if pr is None or r["bin0_mass"] is None or r["jaccard"] is None or r["ret_mass"] is None:
            continue
        Xrows.append([1.0, math.log(max(pr, 1)), r["bin0_mass"], r["jaccard"], r["ret_mass"]])
        yrows.append(r["alpha"])
        keep.append(r)
    X = np.array(Xrows); y = np.array(yrows)
    if X.shape[0] < 10:
        line("  insufficient overlap with PR data; skipping")
        return
    # OLS fits with nested models
    def fit(cols):
        Xs = X[:, cols]
        b, *_ = np.linalg.lstsq(Xs, y, rcond=None)
        y_hat = Xs @ b
        ssr = float(((y - y_hat) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ssr / sst if sst > 0 else float("nan")
        return b, r2
    headers = ["intercept", "log_PR", "bin0_mass", "jaccard", "ret_mass"]
    nested = [
        ("log_PR only",                [0, 1]),
        ("+ bin0_mass",                [0, 1, 2]),
        ("+ jaccard",                  [0, 1, 2, 3]),
        ("+ ret_mass",                 [0, 1, 2, 3, 4]),
        ("bin0_mass only",             [0, 2]),
        ("bin0_mass + jaccard",        [0, 2, 3]),
    ]
    line(f"  n = {len(y)} (uniform-allocation, Phi snapkv, both τ pooled)")
    line(f"\n  {'model':<28}{'R²':>8}  coefficients")
    for name, cols in nested:
        b, r2 = fit(cols)
        coefs = ", ".join(f"{headers[c]}={b[i]:+.3f}" for i, c in enumerate(cols))
        line(f"  {name:<28}{r2:>+8.4f}  {coefs}")


def s11_heterogeneity_phi(db_rows):
    section("S11. Cochran's Q heterogeneity of per-task PR-α slopes within Phi")
    sub = [r for r in db_rows if r["model"] == "phi_3p5_mini_instruct"
           and r["priority"] == "snapkv" and r["PR"] is not None]
    by_task = defaultdict(list)
    for r in sub:
        by_task[r["task"]].append(r)
    slopes = []
    ses = []
    tasks = sorted(by_task.keys())
    for t in tasks:
        rs = by_task[t]
        n = len(rs)
        if n < 5:
            continue
        xs = np.array([math.log(max(r["PR"], 1)) for r in rs])
        ys = np.array([r["alpha"] for r in rs])
        b = np.cov(xs, ys, ddof=0)[0, 1] / xs.var()
        a = ys.mean() - b * xs.mean()
        yhat = a + b * xs
        resid = ys - yhat
        se = resid.std(ddof=2) / math.sqrt(((xs - xs.mean()) ** 2).sum()) if n > 2 else float("nan")
        slopes.append(b); ses.append(se)
    weights = [1.0 / (se ** 2) for se in ses if not math.isnan(se) and se > 0]
    valid_slopes = [b for b, se in zip(slopes, ses) if not math.isnan(se) and se > 0]
    if len(weights) < 3:
        line("  insufficient tasks for heterogeneity test")
        return
    grand = sum(w * b for w, b in zip(weights, valid_slopes)) / sum(weights)
    Q = sum(w * (b - grand) ** 2 for w, b in zip(weights, valid_slopes))
    df = len(weights) - 1
    # Approximate chi-square p-value via series
    # P(X^2 > Q) for df degrees of freedom; use scipy if available else approx
    try:
        from scipy.stats import chi2
        p = float(1.0 - chi2.cdf(Q, df))
    except Exception:
        # crude normal approximation for Q/df
        p = float("nan")
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    line(f"  n_tasks with sufficient n: {len(weights)}")
    line(f"  grand-mean slope (inverse-variance weighted): {grand:+.4f}")
    line(f"  Cochran's Q = {Q:.3f}  df = {df}  p = {p:.4f}  I² = {I2:.1f}%")
    line("  Per-task slopes:")
    line(f"  {'task':<20}{'slope':>10}{'SE':>10}")
    for b, se, t in zip(slopes, ses, tasks):
        if math.isnan(se) or se <= 0:
            continue
        line(f"  {t:<20}{b:>+10.4f}{se:>10.4f}")


def s12_pairwise_basin_separation(db_rows):
    section("S12. Pairwise α-basin separation (Holm-corrected)")
    sub = [r for r in db_rows if r["priority"] == "snapkv" and r["ratio"] == 0.04
           and r["tier"] == 32768]
    by_m = defaultdict(list)
    for r in sub:
        by_m[r["model"]].append(r["alpha"])
    models = sorted(by_m.keys())
    raw_p = []
    pairs = []
    obs = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            x = by_m[a]; y = by_m[b]
            if len(x) < 4 or len(y) < 4:
                continue
            _, p = permutation_test(x, y, n_perm=10000, seed=6)
            raw_p.append(p)
            pairs.append((a, b))
            obs.append(float(np.mean(x) - np.mean(y)))
    adj = holm_bonferroni(raw_p)
    line(f"  {'pair':<60}{'Δmean':>10}{'raw p':>10}{'Holm':>10}")
    for (a, b), d, p, ap in zip(pairs, obs, raw_p, adj):
        line(f"  {a[:28]} vs {b[:28]:<28}{d:>+10.4f}{p:>10.5f}{ap:>10.5f}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("[stats] loading DB rows...")
    db_rows = load_db_alpha()
    print(f"[stats]   {len(db_rows)} DB rows (uniform, k_dec>=15)")
    print("[stats] loading P7 rows...")
    p7_rows = load_p7()
    print(f"[stats]   {len(p7_rows)} P7 rows")
    print("[stats] loading P10/P11 rows...")
    p10p11 = load_p10p11()
    print(f"[stats]   {len(p10p11)} P10/P11 rows")

    s1_bootstrap_headline(db_rows, p7_rows)
    s2_effect_sizes(db_rows)
    s3_permutation_inversion(db_rows)
    s4_variance_components(db_rows)
    s5_multiple_testing(db_rows)
    s6_leave_one_out(db_rows)
    s7_power(db_rows)
    s8_p10_position_shift(p10p11)
    s9_jacobian(p10p11)
    s10_joint_regression_phi(p10p11, db_rows)
    s11_heterogeneity_phi(db_rows)
    s12_pairwise_basin_separation(db_rows)

    out_path = OUT / "stats_appendix.md"
    header = (
        "# Statistical appendix for the KV-PR paper\n\n"
        "Generated by `docs/paper/run_full_stats.py`. Numbers are "
        "from the full DB ingest plus the P7 / P10 / P11 pulls. "
        "All p-values are two-sided unless noted.\n\n"
    )
    body = "\n".join(OUTLINES) + "\n"
    out_path.write_text(header + body, encoding="utf-8")
    print(f"[stats] wrote {out_path}")


if __name__ == "__main__":
    main()
