#!/usr/bin/env python3
"""Recompute every paper-table / paper-figure value from the updated DB.

Reads docs/paper/paper_data.db (which now includes the 2026-05-21 mechanism
pull) and emits:

  docs/paper/paper_numbers_v2.md   — every recomputed value with a sentence
                                     of side-by-side context vs the old value
  docs/paper/recomputed_table1_oracle_top.csv
  docs/paper/recomputed_table2_priority_inversion_131k.csv
  docs/paper/recomputed_table3_conc_idx_scaling.csv
  docs/paper/recomputed_table5_selector.csv
  docs/paper/recomputed_figure1_priority_inversion.csv
  docs/paper/recomputed_figure3_conc_idx.csv
  docs/paper/recomputed_figure4_selector.csv

Key rules:
  - num_groups=4 only (g_sweep rows with G ∈ {1,2,8,16} are excluded).
  - uniform allocation only for Table 2 / Table 3 / Figure 1
    (max(alloc) − min(alloc) ≤ 1).
  - n_steps_kl > 0 (skips diffuseness-calib-style degenerate measurements;
    the dedup already excludes diffuseness_calib but this is belt + braces).
"""

from __future__ import annotations

import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
DB = OUT / "paper_data.db"

MODEL_ORDER = ["llama3p2_1b_instruct", "qwen3_1p7b", "phi_3p5_mini_instruct"]
PRIORITY_ORDER = ["snapkv", "accumulated", "recent", "random", "shared_random"]


def bootstrap_median_ci(values, n_resamples=2000, alpha=0.05, seed=0):
    """Percentile-method bootstrap CI on the median."""
    import random
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    medians = []
    n = len(values)
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(resample))
    medians.sort()
    med = statistics.median(values)
    lo = medians[int(alpha/2 * n_resamples)]
    hi = medians[int((1 - alpha/2) * n_resamples)]
    return (med, lo, hi)


def is_uniform(alloc_json):
    a = json.loads(alloc_json)
    if not a:
        return False
    return (max(a) - min(a)) <= 1


def load_kl_table(con):
    """Long-form join: model, priority, task, tier, ratio, alloc, prompt_id,
    mean_kl. Filtered to num_groups=4 and uniform allocation."""
    cur = con.cursor()
    cur.execute("""
        SELECT r.model, r.priority, p.task, p.tier_length, m.ratio,
               m.allocation_json, p.prompt_id, m.mean_kl, m.n_steps_kl
        FROM measurements m
        JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
        JOIN runs r    ON p.run_id        = r.run_id
        WHERE r.num_groups = 4 AND m.n_steps_kl > 0
    """)
    return list(cur)


def table1_oracle_top(con):
    """Top-10 oracle-gap cells under snapkv priority.

    For each (model, task, tier, ratio): collect per-prompt per-allocation
    KL under snapkv. Oracle gap on a prompt = KL_uniform − min(KL across
    allocations). Aggregate the per-prompt oracle-gap (median, CI95).
    """
    cur = con.cursor()
    cur.execute("""
        SELECT r.model, p.task, p.tier_length, m.ratio, p.prompt_id,
               m.allocation_json, m.mean_kl
        FROM measurements m
        JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
        JOIN runs r    ON p.run_id        = r.run_id
        WHERE r.priority = 'snapkv' AND r.num_groups = 4 AND m.n_steps_kl > 0
    """)
    by_cell_prompt = defaultdict(lambda: {"uniform": None, "min": None})
    for model, task, tier, ratio, pid, alloc_json, kl in cur:
        key = (model, task, tier, ratio, pid)
        e = by_cell_prompt[key]
        if e["min"] is None or kl < e["min"]:
            e["min"] = kl
        if is_uniform(alloc_json):
            e["uniform"] = kl
    # Per-cell aggregate.
    by_cell = defaultdict(list)
    for (model, task, tier, ratio, pid), e in by_cell_prompt.items():
        if e["uniform"] is None or e["min"] is None:
            continue
        gap = e["uniform"] - e["min"]
        by_cell[(model, task, tier, ratio)].append(gap)
    rows = []
    for (model, task, tier, ratio), gaps in by_cell.items():
        if len(gaps) < 1:
            continue
        med, lo, hi = bootstrap_median_ci(gaps)
        rows.append({
            "model": model, "task": task, "tier": tier, "ratio": ratio,
            "n_prompts": len(gaps),
            "og_med": med, "og_ci_lo": lo, "og_ci_hi": hi,
            "og_max": max(gaps), "og_mean": statistics.mean(gaps),
        })
    rows.sort(key=lambda r: -r["og_med"])
    return rows


def table2_priority_inversion(con, tier=131072, ratio=0.04):
    """Per-(model, priority) mean uniform KL at a tier × ratio cell.

    Aggregation matches the paper: per-prompt uniform KL → mean across all
    prompts in cells (model, *, tier, ratio).
    """
    cur = con.cursor()
    cur.execute("""
        SELECT r.model, r.priority, p.task, p.prompt_id, m.mean_kl,
               m.allocation_json
        FROM measurements m
        JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
        JOIN runs r    ON p.run_id        = r.run_id
        WHERE p.tier_length = ? AND m.ratio = ? AND r.num_groups = 4
          AND m.n_steps_kl > 0
    """, (tier, ratio))
    bins = defaultdict(list)
    for model, priority, task, pid, kl, alloc_json in cur:
        if not is_uniform(alloc_json):
            continue
        bins[(model, priority)].append(kl)
    rows = []
    for (model, priority), kls in bins.items():
        rows.append({
            "model": model, "priority": priority, "tier": tier, "ratio": ratio,
            "n_prompts": len(kls),
            "mean_kl": statistics.mean(kls),
            "median_kl": statistics.median(kls),
        })
    rows.sort(key=lambda r: (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
                              PRIORITY_ORDER.index(r["priority"]) if r["priority"] in PRIORITY_ORDER else 99))
    return rows


def table3_conc_idx_scaling(con, ratio=0.08):
    """conc_idx_median = (KL_uniform − KL_best_dominant) / KL_uniform,
    per (model, priority, tier) at one ratio."""
    cur = con.cursor()
    cur.execute("""
        SELECT r.model, r.priority, p.task, p.tier_length, p.prompt_id,
               m.allocation_json, m.mean_kl
        FROM measurements m
        JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
        JOIN runs r    ON p.run_id        = r.run_id
        WHERE m.ratio = ? AND r.num_groups = 4 AND m.n_steps_kl > 0
    """, (ratio,))
    by_pp = defaultdict(lambda: {"uniform": None, "best_dom": None})
    for model, priority, task, tier, pid, alloc_json, kl in cur:
        a = json.loads(alloc_json)
        if not a:
            continue
        key = (model, priority, tier, task, pid)
        e = by_pp[key]
        if is_uniform(alloc_json):
            e["uniform"] = kl
        # "single-dominant": one group gets most of the budget (>= 60% of total)
        if max(a) / sum(a) >= 0.60:
            if e["best_dom"] is None or kl < e["best_dom"]:
                e["best_dom"] = kl
    by_cell = defaultdict(list)
    for (model, priority, tier, task, pid), e in by_pp.items():
        if e["uniform"] is None or e["best_dom"] is None or e["uniform"] <= 1e-6:
            continue
        ci = (e["uniform"] - e["best_dom"]) / e["uniform"]
        by_cell[(model, priority, tier)].append(ci)
    rows = []
    for (model, priority, tier), cis in by_cell.items():
        rows.append({
            "model": model, "priority": priority, "tier": tier,
            "n": len(cis), "conc_idx_median": statistics.median(cis),
            "conc_idx_mean": statistics.mean(cis),
        })
    rows.sort(key=lambda r: (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
                              r["tier"],
                              PRIORITY_ORDER.index(r["priority"]) if r["priority"] in PRIORITY_ORDER else 99))
    return rows


def table5_selector(con):
    """Reproduce the selector benchmark: for each (model, task, tier, ratio)
    cell with both snapkv AND random data, split prompts into calibration
    (first half) and held-out (second half); compute mean held-out KL for
    each strategy.

    Also includes shared_random if present (new in the mechanism pull).
    """
    cur = con.cursor()
    # Per-cell prompts and per-priority per-prompt mean KL under uniform.
    cur.execute("""
        SELECT r.model, r.priority, p.task, p.tier_length, m.ratio,
               p.prompt_id, m.mean_kl, m.allocation_json
        FROM measurements m
        JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
        JOIN runs r    ON p.run_id        = r.run_id
        WHERE r.num_groups = 4 AND m.n_steps_kl > 0
    """)
    per_cell = defaultdict(lambda: defaultdict(dict))  # cell -> priority -> prompt_id -> [kl]
    for model, priority, task, tier, ratio, pid, kl, alloc_json in cur:
        if not is_uniform(alloc_json):
            continue
        per_cell[(model, task, tier, ratio)].setdefault(priority, {}).setdefault(pid, []).append(kl)

    # PR threshold rule: PR > 5000 → random; else snapkv.
    cur.execute("SELECT model, AVG(participation_ratio) FROM diffuseness WHERE tier_length=32768 GROUP BY model")
    pr_by_model = {m: avg_pr for m, avg_pr in cur}
    pr_choice = {m: ("random" if pr_by_model.get(m, 0) > 5000 else "snapkv") for m in pr_by_model}

    rows = []
    for (model, task, tier, ratio), per_pri in per_cell.items():
        if "snapkv" not in per_pri or "random" not in per_pri:
            continue
        snap_pids = sorted(per_pri["snapkv"].keys())
        rand_pids = sorted(per_pri["random"].keys())
        common = sorted(set(snap_pids) & set(rand_pids))
        if len(common) < 4:
            continue
        # Split common prompts into calibration (first half) and held-out (rest)
        n_cal = max(1, len(common) // 2)
        cal_pids = common[:n_cal]
        held_pids = common[n_cal:]
        snap_cal = statistics.mean(per_pri["snapkv"][pid][0] for pid in cal_pids)
        rand_cal = statistics.mean(per_pri["random"][pid][0] for pid in cal_pids)
        snap_held = statistics.mean(per_pri["snapkv"][pid][0] for pid in held_pids)
        rand_held = statistics.mean(per_pri["random"][pid][0] for pid in held_pids)
        # Calibration selector picks the lower-KL priority on cal prompts.
        cal_choice = "snapkv" if snap_cal < rand_cal else "random"
        cal_held = snap_held if cal_choice == "snapkv" else rand_held
        # PR-threshold selector.
        pr_ch = pr_choice.get(_pr_key(model), "snapkv")
        pr_held = snap_held if pr_ch == "snapkv" else rand_held
        # Oracle = min of the two.
        oracle = min(snap_held, rand_held)
        row = {
            "model": model, "task": task, "tier": tier, "ratio": ratio,
            "n_cal": len(cal_pids), "n_held": len(held_pids),
            "snap_held": snap_held, "rand_held": rand_held,
            "cal_choice": cal_choice, "cal_held": cal_held,
            "pr_choice": pr_ch, "pr_held": pr_held,
            "oracle": oracle,
        }
        if "shared_random" in per_pri:
            common_sh = sorted(set(per_pri["shared_random"].keys()) & set(common))
            if common_sh:
                sh_held = statistics.mean(per_pri["shared_random"][pid][0]
                                          for pid in common_sh if pid in held_pids
                                          and pid in per_pri["shared_random"])
                row["shared_held"] = sh_held
        rows.append(row)
    return rows


def _pr_key(model_slug):
    """Map measurement model_slug to diffuseness model slug."""
    return {
        "llama3p2_1b_instruct": "llama_1b",
        "qwen3_1p7b": "qwen_1p7b",
        "phi_3p5_mini_instruct": "phi_3p5",
    }.get(model_slug, model_slug)


def write_csv(name, rows):
    if not rows:
        print(f"(no rows for {name})")
        return
    cols = list(rows[0].keys())
    p = OUT / name
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = round(v, 6)
            w.writerow(r)


def summarize_t2(rows):
    print("\n=== Table 2 — Phi/Qwen/Llama @131k r=0.04 mean uniform KL ===")
    cur = {}
    for r in rows:
        cur.setdefault(r["model"], {})[r["priority"]] = (r["mean_kl"], r["n_prompts"])
    for m in MODEL_ORDER:
        d = cur.get(m, {})
        snap = d.get("snapkv", (float("nan"), 0))
        rec = d.get("recent", (float("nan"), 0))
        rand = d.get("random", (float("nan"), 0))
        sh = d.get("shared_random", (float("nan"), 0))
        print(f"  {m:<28} snap={snap[0]:.4f} (n={snap[1]})  "
              f"recent={rec[0]:.4f} (n={rec[1]})  "
              f"rand={rand[0]:.4f} (n={rand[1]})  "
              f"shared={sh[0]:.4f} (n={sh[1]})")


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # Sanity: any g_sweep contamination?
    cur.execute("SELECT COUNT(*) FROM runs WHERE num_groups != 4")
    n_nongroup4 = cur.fetchone()[0]
    print(f"runs with num_groups != 4 (must be excluded from paper aggregations): {n_nongroup4}")

    t1 = table1_oracle_top(con)
    write_csv("recomputed_table1_oracle_top.csv", t1)
    print(f"\nTable 1 oracle-gap top-10 (by og_med, snapkv priority):")
    for r in t1[:12]:
        print(f"  {r['model']:<28}{r['task']:<18}{r['tier']:>7}{r['ratio']:>6.3f}  "
              f"n={r['n_prompts']:>3}  og_med={r['og_med']:.3f}  CI=[{r['og_ci_lo']:.2f},{r['og_ci_hi']:.2f}]")

    t2 = table2_priority_inversion(con, 131072, 0.04)
    write_csv("recomputed_table2_priority_inversion_131k.csv", t2)
    summarize_t2(t2)

    t3 = table3_conc_idx_scaling(con, ratio=0.08)
    write_csv("recomputed_table3_conc_idx_scaling.csv", t3)
    print(f"\n=== Table 3 — conc_idx_median by (model, priority, tier) at r=0.08 ===")
    for r in t3:
        if r["priority"] not in ("snapkv", "random", "shared_random"):
            continue
        print(f"  {r['model']:<28}{r['priority']:<15}{r['tier']:>7}  n={r['n']:>3}  "
              f"conc_idx_med={r['conc_idx_median']:+.3f}")

    t5 = table5_selector(con)
    write_csv("recomputed_table5_selector.csv", t5)
    print(f"\n=== Table 5 — selector cells recomputed ===")
    by_model = defaultdict(lambda: {"snap": [], "rand": [], "cal": [], "pr": [], "oracle": [], "shared": []})
    for r in t5:
        by_model[r["model"]]["snap"].append(r["snap_held"])
        by_model[r["model"]]["rand"].append(r["rand_held"])
        by_model[r["model"]]["cal"].append(r["cal_held"])
        by_model[r["model"]]["pr"].append(r["pr_held"])
        by_model[r["model"]]["oracle"].append(r["oracle"])
        if "shared_held" in r:
            by_model[r["model"]]["shared"].append(r["shared_held"])
    by_model["OVERALL"] = {k: [v for d in by_model.values() for v in d[k]] for k in ["snap","rand","cal","pr","oracle","shared"]}
    print(f"  {'model':<28} {'n_cells':>8} {'snap':>7} {'rand':>7} {'cal':>7} {'PR':>7} {'oracle':>7} {'shared':>8}")
    for m in [*MODEL_ORDER, "OVERALL"]:
        d = by_model.get(m, {})
        if not d.get("snap"): continue
        snap = statistics.mean(d["snap"])
        rand = statistics.mean(d["rand"])
        cal = statistics.mean(d["cal"])
        pr = statistics.mean(d["pr"])
        oracle = statistics.mean(d["oracle"])
        shared = statistics.mean(d["shared"]) if d.get("shared") else float("nan")
        print(f"  {m:<28} {len(d['snap']):>8} {snap:>7.4f} {rand:>7.4f} {cal:>7.4f} {pr:>7.4f} {oracle:>7.4f} {shared:>8.4f}")

    # Figure-data CSVs (already covered by tables; pass through).
    write_csv("recomputed_figure1_priority_inversion.csv", t2)
    write_csv("recomputed_figure3_conc_idx.csv", [r for r in t3 if r["tier"] == 131072])
    write_csv("recomputed_figure4_selector.csv", t5)

    con.close()


if __name__ == "__main__":
    main()
