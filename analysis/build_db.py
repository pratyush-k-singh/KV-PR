#!/usr/bin/env python3
"""Build the paper's central experiment database.

Walks every paper-canonical ``*_merged/`` dir under analyses/ (the dirs whose
data feeds master_audit) and writes:

  docs/paper/paper_data.db                — SQLite, queryable
  docs/paper/paper_grid_coverage.csv      — (model, task, tier, prompt_id) ×
                                            priority indicator grid (NULL where
                                            not measured)
  docs/paper/paper_grid_results.csv       — same rows, but each priority cell
                                            holds the per-prompt mean held-out
                                            KL at the lowest-KL ratio
  docs/paper/paper_measurements_full.csv  — narrow long-form: one row per
                                            (run, prompt, ratio, allocation)
                                            measurement, with mean_kl and the
                                            full per_step_kl array
  docs/paper/paper_cell_stats.csv         — cell-level aggregate (1359 rows
                                            from master_audit/stats.json)
  docs/paper/paper_selector_cells.csv     — 264 selector cells from
                                            master_audit/priority_selector.json
  docs/paper/paper_diffuseness.csv        — per-prompt PR/H/mass_cov from
                                            master_audit/attention_diffuseness_v2.json

Run:
  python docs/paper/build_central_db.py

No GPU, ~10s for the full walk on a warm cache.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ROOT / "analyses"
OUT_DIR = Path(__file__).resolve().parent
DB_PATH = OUT_DIR / "paper_data.db"
MASTER_AUDIT = ANALYSES / "2026-05-19_master_audit"

# Paper-canonical pull collections. Restrict here so the DB contains "only
# data used for the final paper results". Excluded by design:
#   - 2026-05-17_saturation_*  (pre-pilot Stage-A; k_dec=8, n=1, snapkv-only;
#                               superseded by sparse_breadth; zero rows in
#                               master_audit diffuseness)
#   - 2026-05-18_tail_repl_n24_k8_pull           (k_dec=8 mismatch with the
#   - 2026-05-18_priority_sweep_qwen_tail_k8_pull paper's k_dec=16 protocol)
# The diffuseness PR contributions from tail_repl still land in the DB via
# the master_audit/attention_diffuseness_v2.json import, since PR is k_dec
# independent — only the per-step KL trajectory data is dropped.
# Ingestion order matters: targeted_n12 first because it is a SUPERSET of the
# 8 sparse_breadth cells it overlaps (12 prompts × standard 12 allocations +
# 3 extra published allocators / ratio). After targeted_n12 is ingested, the
# sparse_breadth ingestion deduplicates against the (prompt_id, model,
# priority, tier_length) tuples already present, dropping the 48 byte-identical
# prompt-rows that would otherwise inflate cell-level n by 50–100%.
#
# The 2026-05-21 mechanism pull uses seed s=43 (existing pulls use s=42), so
# its prompts are an independent sample — no overlap, no dedup needed. It
# contributes:
#   - the new shared_random priority (paper §5.4 mechanism test)
#   - a G-sweep (num_groups ∈ {1,2,8,16}); g_sweep rows must be filtered out
#     of any aggregation that assumes G=4 (see views below)
#   - additional snapkv / indep-random data on the same (model,task,tier)
#     grid the paper already covers, with the new coverage telemetry block
PAPER_PULL_ROOTS = [
    "2026-05-19_targeted_n12_v1_pull",
    "2026-05-19_endpoint_131k_n1_llama_v1_pull",
    "2026-05-19_endpoint_131k_n1_phi_h200_v1_pull",
    "2026-05-19_endpoint_131k_n1_qwen_v1_pull",
    "2026-05-18_sparse_breadth_n6_v1_llama_pull",
    "2026-05-18_sparse_breadth_n6_v1_phi35_pull",
    "2026-05-18_sparse_breadth_n6_v1_qwen_pull",
    "2026-05-21_mechanism_s43_n4_v8_pull",
]
# Excluded:
#   - 2026-05-19_diffuseness_calib_n2_v1_pull: k_dec=1, n_steps_kl=0 for every
#     trajectory; its data was a full-KV attention capture for PR computation,
#     not a real KL measurement. PR is preserved via the master_audit import.
#   - 2026-05-19_endpoint_131k_n1_v1_pull: duplicates the 22 Llama+Qwen 131k
#     prompts already present in the per-model llama_v1_pull and qwen_v1_pull
#     pulls (Phi 131k is in its own h200 pull).


def _model_slug(model_path: str) -> str:
    name = Path(str(model_path)).name
    return name.replace(".", "p").replace("-", "_").lower()


def _task_from_prompts(prompts: list[dict]) -> str:
    from collections import Counter
    c = Counter(p.get("task", "unknown") for p in prompts if p.get("status") == "ok")
    if not c:
        return "unknown"
    return c.most_common(1)[0][0]


def _tier_from_prompts(prompts: list[dict]) -> int | None:
    from collections import Counter
    c = Counter(p.get("tier_length") for p in prompts if p.get("status") == "ok")
    if not c:
        return None
    return c.most_common(1)[0][0]


def discover_merged_dirs() -> list[Path]:
    out: list[Path] = []
    for pull in PAPER_PULL_ROOTS:
        root = ANALYSES / pull
        if not root.is_dir():
            continue
        # Either the pull dir is itself the merged dir (rare), or it has
        # nested *_merged subdirs.
        if (root / "trajectories.json").exists() and root.name.endswith("_merged"):
            out.append(root)
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and entry.name.endswith("_merged") and (entry / "trajectories.json").exists():
                out.append(entry)
            elif entry.is_dir() and not entry.name.endswith("_merged"):
                # one-level deeper (e.g. endpoint_131k_n1_v1_pull/.../2026-05-19_endpoint_..._merged)
                for sub in sorted(entry.iterdir()) if entry.is_dir() else []:
                    if sub.is_dir() and sub.name.endswith("_merged") and (sub / "trajectories.json").exists():
                        out.append(sub)
    return out


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_dir TEXT UNIQUE NOT NULL,
  pull_collection TEXT NOT NULL,
  model TEXT NOT NULL,
  model_path TEXT,
  priority TEXT NOT NULL,
  k_dec INTEGER,
  obs_window INTEGER,
  num_groups INTEGER,
  n_sink INTEGER,
  n_recent INTEGER,
  ratios_json TEXT,
  priority_seed INTEGER,
  accumulated_chunk_size INTEGER,
  extra_allocators TEXT,
  skip_support INTEGER,
  attn_backend TEXT,
  chat_template TEXT,
  prompts_attempted INTEGER,
  prompts_ok INTEGER,
  inferred_task TEXT,
  inferred_tier INTEGER
);

CREATE TABLE IF NOT EXISTS prompts (
  prompt_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(run_id),
  prompt_id TEXT NOT NULL,
  task TEXT NOT NULL,
  tier_length INTEGER NOT NULL,
  tokens INTEGER,
  status TEXT,
  floor_sum INTEGER,
  capture_prefill_s REAL,
  prefill_s REAL,
  ref_decode_s REAL,
  masked_decode_s REAL,
  peak_mem_gb REAL,
  UNIQUE(run_id, prompt_id)
);

CREATE TABLE IF NOT EXISTS measurements (
  measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
  prompt_row_id INTEGER NOT NULL REFERENCES prompts(prompt_row_id),
  ratio REAL NOT NULL,
  total_budget INTEGER NOT NULL,
  allocation_json TEXT NOT NULL,
  cost REAL,
  per_step_kl_json TEXT NOT NULL,
  mean_kl REAL,
  n_steps_kl INTEGER,
  -- Coverage telemetry (populated only by the 2026-05-21 mechanism pull)
  union_size INTEGER,
  intersection_size INTEGER,
  mean_pairwise_jaccard REAL,
  total_retained_reference_mass REAL,
  mean_retained_reference_mass REAL,
  union_fraction_of_context REAL,
  union_fraction_of_group_budget_sum REAL
);

CREATE TABLE IF NOT EXISTS diffuseness (
  diff_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  model TEXT NOT NULL,
  tier_length INTEGER NOT NULL,
  task TEXT NOT NULL,
  priority TEXT,
  prompt_id TEXT NOT NULL,
  tokens INTEGER,
  mass_cov_0p95 REAL,
  participation_ratio REAL,
  entropy REAL,
  UNIQUE(model, tier_length, task, prompt_id)
);

CREATE TABLE IF NOT EXISTS cell_stats (
  cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model TEXT NOT NULL,
  task TEXT NOT NULL,
  priority TEXT NOT NULL,
  ratio REAL NOT NULL,
  n INTEGER,
  og_mean REAL,
  og_med REAL,
  og_max REAL,
  og_ci_lo REAL,
  og_ci_hi REAL,
  heavy_tail_index REAL,
  tail_rate_0p10 REAL,
  tail_rate_0p50 REAL,
  tail_rate_1p00 REAL,
  uniform_kl_mean REAL,
  mincost_diff_mean REAL,
  conc_idx_median REAL,
  spread_idx_median REAL,
  worst_gap_mean REAL,
  source_file TEXT
);

CREATE TABLE IF NOT EXISTS selector_cells (
  sel_cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model TEXT NOT NULL,
  task TEXT NOT NULL,
  tier_length INTEGER NOT NULL,
  ratio REAL NOT NULL,
  n_calibration INTEGER,
  n_held INTEGER,
  snap_cal_mean REAL,
  rand_cal_mean REAL,
  fixed_snapkv_held REAL,
  fixed_random_held REAL,
  selector_calibration_choice TEXT,
  selector_calibration_held REAL,
  selector_pr_threshold_choice TEXT,
  selector_pr_threshold_held REAL,
  oracle_held REAL
);

CREATE INDEX IF NOT EXISTS idx_prompts_lookup ON prompts(task, tier_length, prompt_id);
CREATE INDEX IF NOT EXISTS idx_meas_prompt ON measurements(prompt_row_id);
CREATE INDEX IF NOT EXISTS idx_runs_model_priority ON runs(model, priority);
CREATE INDEX IF NOT EXISTS idx_diff_lookup ON diffuseness(model, tier_length, task, prompt_id);
CREATE INDEX IF NOT EXISTS idx_cell_stats_lookup ON cell_stats(model, task, priority, ratio);
"""


def mean_of_nonzero_tail(arr: list[float]) -> float:
    """Mean of per_step_kl from step 1 onwards (step 0 is shared prefill, KL=0)."""
    tail = arr[1:] if len(arr) > 1 else arr
    return statistics.mean(tail) if tail else 0.0


def ingest_run(con: sqlite3.Connection, merged_dir: Path) -> tuple[int, int]:
    """Ingest one merged dir. Returns (n_prompts, n_measurements) inserted."""
    traj_path = merged_dir / "trajectories.json"
    cfg_path = merged_dir / "eval_config.json"
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    # Run metadata.
    prompts_list = data.get("prompts", [])
    model_path = data.get("model", "")
    model = _model_slug(model_path)
    priority = data.get("priority", "snapkv")
    inferred_task = _task_from_prompts(prompts_list)
    inferred_tier = _tier_from_prompts(prompts_list)
    pull_collection = merged_dir.parent.name

    cur = con.cursor()
    rel_dir = str(merged_dir.relative_to(ROOT)).replace(os.sep, "/")
    cur.execute(
        """INSERT OR IGNORE INTO runs (
              source_dir, pull_collection, model, model_path, priority, k_dec,
              obs_window, num_groups, n_sink, n_recent, ratios_json,
              priority_seed, accumulated_chunk_size, extra_allocators,
              skip_support, attn_backend, chat_template, prompts_attempted,
              prompts_ok, inferred_task, inferred_tier
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            rel_dir, pull_collection, model, model_path, priority,
            data.get("k_dec"),
            data.get("obs_window"),
            data.get("num_groups"),
            data.get("n_sink"),
            data.get("n_recent"),
            json.dumps(data.get("ratios", [])),
            data.get("priority_seed"),
            data.get("accumulated_chunk_size"),
            ",".join(data.get("extra_allocators", []) or []),
            1 if data.get("skip_support") else 0,
            data.get("attn_backend"),
            data.get("chat_template"),
            data.get("prompts_attempted"),
            data.get("prompts_ok"),
            inferred_task,
            inferred_tier,
        ),
    )
    cur.execute("SELECT run_id FROM runs WHERE source_dir = ?", (rel_dir,))
    run_id = cur.fetchone()[0]

    n_prompts = 0
    n_meas = 0
    n_skipped_dupe = 0
    for p in prompts_list:
        if p.get("status") != "ok":
            continue
        # Cross-run dedup: skip this prompt if (prompt_id, model, priority,
        # tier_length, num_groups) already exists from an earlier-ingested pull.
        # `num_groups` must be in the key so the G-sweep cells (which reuse
        # the same prompt_ids at G ∈ {1,2,8,16} alongside G=4) don't
        # accidentally collapse into one another.
        num_groups = data.get("num_groups")
        cur.execute(
            """SELECT 1 FROM prompts p2 JOIN runs r2 ON p2.run_id=r2.run_id
               WHERE p2.prompt_id=? AND r2.model=? AND r2.priority=?
                 AND p2.tier_length=? AND r2.num_groups=?
                 AND r2.run_id != ? LIMIT 1""",
            (p["prompt_id"], model, priority, p.get("tier_length"),
             num_groups, run_id),
        )
        if cur.fetchone():
            n_skipped_dupe += 1
            continue
        cur.execute(
            """INSERT OR IGNORE INTO prompts (
                  run_id, prompt_id, task, tier_length, tokens, status,
                  floor_sum, capture_prefill_s, prefill_s, ref_decode_s,
                  masked_decode_s, peak_mem_gb
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, p["prompt_id"], p.get("task"), p.get("tier_length"),
                p.get("tokens"), p.get("status"), p.get("floor_sum"),
                p.get("capture_prefill_s"), p.get("prefill_s"),
                p.get("ref_decode_s"), p.get("masked_decode_s"),
                p.get("peak_mem_gb"),
            ),
        )
        cur.execute(
            "SELECT prompt_row_id FROM prompts WHERE run_id = ? AND prompt_id = ?",
            (run_id, p["prompt_id"]),
        )
        prompt_row_id = cur.fetchone()[0]
        n_prompts += 1
        for t in p.get("trajectories", []):
            kl = t.get("per_step_kl", [])
            cov = t.get("coverage") or {}
            cur.execute(
                """INSERT INTO measurements (
                      prompt_row_id, ratio, total_budget, allocation_json,
                      cost, per_step_kl_json, mean_kl, n_steps_kl,
                      union_size, intersection_size, mean_pairwise_jaccard,
                      total_retained_reference_mass, mean_retained_reference_mass,
                      union_fraction_of_context, union_fraction_of_group_budget_sum
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    prompt_row_id, t.get("ratio"), t.get("total"),
                    json.dumps(t.get("allocation", [])), t.get("cost"),
                    json.dumps(kl), mean_of_nonzero_tail(kl),
                    len(kl) - 1 if len(kl) > 1 else 0,
                    cov.get("union_size"), cov.get("intersection_size"),
                    cov.get("mean_pairwise_jaccard"),
                    cov.get("total_retained_reference_mass"),
                    cov.get("mean_retained_reference_mass"),
                    cov.get("union_fraction_of_context"),
                    cov.get("union_fraction_of_group_budget_sum"),
                ),
            )
            n_meas += 1
    return n_prompts, n_meas


def ingest_master_audit(con: sqlite3.Connection) -> tuple[int, int, int]:
    """Load the three master_audit JSON aggregates."""
    cur = con.cursor()

    # cell_stats (1359 rows).
    stats = json.loads((MASTER_AUDIT / "stats.json").read_text(encoding="utf-8"))
    cs_n = 0
    for r in stats.get("rows", []):
        cur.execute(
            """INSERT INTO cell_stats (
                  model, task, priority, ratio, n, og_mean, og_med, og_max,
                  og_ci_lo, og_ci_hi, heavy_tail_index, tail_rate_0p10,
                  tail_rate_0p50, tail_rate_1p00, uniform_kl_mean,
                  mincost_diff_mean, conc_idx_median, spread_idx_median,
                  worst_gap_mean, source_file
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                r.get("model"), r.get("task"), r.get("priority"),
                r.get("ratio"), r.get("n"), r.get("og_mean"), r.get("og_med"),
                r.get("og_max"), r.get("og_ci_lo"), r.get("og_ci_hi"),
                r.get("heavy_tail_index"), r.get("tail_rate_0p10"),
                r.get("tail_rate_0p50"), r.get("tail_rate_1p00"),
                r.get("uniform_kl_mean"), r.get("mincost_diff_mean"),
                r.get("conc_idx_median"), r.get("spread_idx_median"),
                r.get("worst_gap_mean"),
                "analyses/2026-05-19_master_audit/stats.json",
            ),
        )
        cs_n += 1

    # diffuseness (246 rows).
    diff = json.loads((MASTER_AUDIT / "attention_diffuseness_v2.json").read_text(encoding="utf-8"))
    diff_n = 0
    for r in diff.get("per_prompt_rows", []):
        cur.execute(
            """INSERT OR IGNORE INTO diffuseness (
                  source, model, tier_length, task, priority, prompt_id,
                  tokens, mass_cov_0p95, participation_ratio, entropy
               ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                r.get("source"), r.get("model"), r.get("tier"),
                r.get("task"), r.get("priority"), r.get("prompt_id"),
                r.get("tokens"), r.get("mass_cov_0p95"),
                r.get("participation_ratio"), r.get("entropy"),
            ),
        )
        diff_n += 1

    # selector_cells (264 rows).
    sel = json.loads((MASTER_AUDIT / "priority_selector.json").read_text(encoding="utf-8"))
    sel_n = 0
    for c in sel.get("cells", []):
        cur.execute(
            """INSERT INTO selector_cells (
                  model, task, tier_length, ratio, n_calibration, n_held,
                  snap_cal_mean, rand_cal_mean, fixed_snapkv_held,
                  fixed_random_held, selector_calibration_choice,
                  selector_calibration_held, selector_pr_threshold_choice,
                  selector_pr_threshold_held, oracle_held
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                c.get("model"), c.get("task"), c.get("tier"), c.get("ratio"),
                c.get("n_calibration"), c.get("n_held"),
                c.get("snap_cal_mean"), c.get("rand_cal_mean"),
                c.get("fixed_snapkv_held"), c.get("fixed_random_held"),
                c.get("selector_calibration_choice"),
                c.get("selector_calibration_held"),
                c.get("selector_pr_threshold_choice"),
                c.get("selector_pr_threshold_held"),
                c.get("oracle_held"),
            ),
        )
        sel_n += 1

    return cs_n, diff_n, sel_n


def export_csvs(con: sqlite3.Connection) -> None:
    """Write the human-readable CSV exports the user asked for."""

    cur = con.cursor()

    # 1) full long-form measurements (one row per experiment trajectory).
    with (OUT_DIR / "paper_measurements_full.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "source_dir", "pull_collection", "model", "priority",
            "num_groups", "task", "tier_length", "prompt_id", "tokens",
            "ratio", "total_budget", "allocation", "cost", "mean_kl",
            "n_steps_kl", "per_step_kl",
            "union_size", "intersection_size", "mean_pairwise_jaccard",
            "total_retained_reference_mass", "mean_retained_reference_mass",
            "union_fraction_of_context", "union_fraction_of_group_budget_sum",
        ])
        cur.execute("""
            SELECT r.run_id, r.source_dir, r.pull_collection, r.model,
                   r.priority, r.num_groups, p.task, p.tier_length,
                   p.prompt_id, p.tokens, m.ratio, m.total_budget,
                   m.allocation_json, m.cost, m.mean_kl, m.n_steps_kl,
                   m.per_step_kl_json, m.union_size, m.intersection_size,
                   m.mean_pairwise_jaccard, m.total_retained_reference_mass,
                   m.mean_retained_reference_mass,
                   m.union_fraction_of_context,
                   m.union_fraction_of_group_budget_sum
            FROM measurements m
            JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
            JOIN runs r ON p.run_id = r.run_id
            ORDER BY r.model, p.task, p.tier_length, p.prompt_id,
                     r.priority, m.ratio
        """)
        for row in cur:
            w.writerow(row)

    # 2) cell_stats export (1359 rows).
    with (OUT_DIR / "paper_cell_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [
            "model", "task", "priority", "ratio", "n", "og_mean", "og_med",
            "og_max", "og_ci_lo", "og_ci_hi", "heavy_tail_index",
            "tail_rate_0p10", "tail_rate_0p50", "tail_rate_1p00",
            "uniform_kl_mean", "mincost_diff_mean", "conc_idx_median",
            "spread_idx_median", "worst_gap_mean", "source_file",
        ]
        w.writerow(cols)
        cur.execute(f"SELECT {','.join(cols)} FROM cell_stats ORDER BY model, task, priority, ratio")
        for row in cur:
            w.writerow(row)

    # 3) selector_cells (264 rows).
    with (OUT_DIR / "paper_selector_cells.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [
            "model", "task", "tier_length", "ratio", "n_calibration", "n_held",
            "snap_cal_mean", "rand_cal_mean", "fixed_snapkv_held",
            "fixed_random_held", "selector_calibration_choice",
            "selector_calibration_held", "selector_pr_threshold_choice",
            "selector_pr_threshold_held", "oracle_held",
        ]
        w.writerow(cols)
        cur.execute(f"SELECT {','.join(cols)} FROM selector_cells ORDER BY model, task, tier_length, ratio")
        for row in cur:
            w.writerow(row)

    # 4) diffuseness (per-prompt PR/H/mass_cov, 246 rows).
    with (OUT_DIR / "paper_diffuseness.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [
            "source", "model", "tier_length", "task", "priority", "prompt_id",
            "tokens", "mass_cov_0p95", "participation_ratio", "entropy",
        ]
        w.writerow(cols)
        cur.execute(f"SELECT {','.join(cols)} FROM diffuseness ORDER BY model, tier_length, task, prompt_id")
        for row in cur:
            w.writerow(row)

    # 5) Coverage grid: one row per (model, task, tier, prompt_id) seen,
    #    columns = priorities; cell = number of ratios measured (NULL if not run).
    cur.execute("""
        SELECT p.prompt_id, p.task, p.tier_length, r.model, r.priority,
               p.tokens, COUNT(m.measurement_id) AS n_meas
        FROM prompts p
        JOIN runs r ON p.run_id = r.run_id
        LEFT JOIN measurements m ON m.prompt_row_id = p.prompt_row_id
        GROUP BY p.prompt_id, p.task, p.tier_length, r.model, r.priority
    """)
    grid: dict[tuple, dict] = {}
    for prompt_id, task, tier, model, priority, tokens, n_meas in cur:
        key = (model, task, tier, prompt_id)
        if key not in grid:
            grid[key] = {"tokens": tokens, "priorities": {}}
        grid[key]["priorities"][priority] = n_meas
    priorities_seen = sorted({p for v in grid.values() for p in v["priorities"]})
    with (OUT_DIR / "paper_grid_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "tier_length", "prompt_id", "tokens"]
                   + [f"n_meas_{p}" for p in priorities_seen])
        for (model, task, tier, prompt_id), info in sorted(grid.items()):
            row = [model, task, tier, prompt_id, info["tokens"]]
            for p in priorities_seen:
                row.append(info["priorities"].get(p, ""))
            w.writerow(row)

    # 6) Results grid: one row per (model, task, tier, prompt_id), columns =
    #    priority × ratio with mean_kl.
    cur.execute("""
        SELECT r.model, p.task, p.tier_length, p.prompt_id, r.priority,
               m.ratio, m.mean_kl
        FROM measurements m
        JOIN prompts p ON m.prompt_row_id = p.prompt_row_id
        JOIN runs r ON p.run_id = r.run_id
    """)
    results: dict[tuple, dict] = {}
    pr_set: set[str] = set()
    ratio_set: set[float] = set()
    for model, task, tier, prompt_id, priority, ratio, mean_kl in cur:
        key = (model, task, tier, prompt_id)
        results.setdefault(key, {})[(priority, ratio)] = mean_kl
        pr_set.add(priority)
        ratio_set.add(ratio)
    pr_order = sorted(pr_set)
    ratio_order = sorted(ratio_set)
    cols = [(p, r) for p in pr_order for r in ratio_order]
    with (OUT_DIR / "paper_grid_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "tier_length", "prompt_id"]
                   + [f"{p}_r{r:.4f}" for (p, r) in cols])
        for (model, task, tier, prompt_id), cells in sorted(results.items()):
            row = [model, task, tier, prompt_id]
            for k in cols:
                v = cells.get(k)
                row.append("" if v is None else f"{v:.6f}")
            w.writerow(row)


def write_readme(con: sqlite3.Connection, counts: dict) -> None:
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM runs"); n_runs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM prompts"); n_prompts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT prompt_id) FROM prompts"); n_unique_prompts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM measurements"); n_meas = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cell_stats"); n_cs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM diffuseness"); n_diff = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM selector_cells"); n_sel = cur.fetchone()[0]
    cur.execute("SELECT model, COUNT(*) FROM prompts p JOIN runs r ON p.run_id=r.run_id GROUP BY model")
    by_model = list(cur)
    cur.execute("""SELECT r.model, p.task, p.tier_length, COUNT(DISTINCT p.prompt_id)
                   FROM prompts p JOIN runs r ON p.run_id=r.run_id
                   GROUP BY r.model, p.task, p.tier_length
                   ORDER BY r.model, p.task, p.tier_length""")
    by_cell = list(cur)

    lines = [
        "# Paper Central Experiment Database",
        "",
        "Single SQLite + CSV bundle covering every experiment that contributed",
        "to the final paper results. Built from the 14 paper-canonical pull",
        "collections (`*_merged/` dirs only — smoke/pre-pilot runs excluded).",
        "",
        "## Files",
        "",
        "| File | Rows | Description |",
        "|---|---:|---|",
        f"| `paper_data.db` | — | SQLite, queryable. Tables: runs, prompts, measurements, diffuseness, cell_stats, selector_cells |",
        f"| `paper_measurements_full.csv` | {n_meas} | Long-form. One row per (run, prompt, ratio, allocation) measurement; includes the full per_step_kl array |",
        f"| `paper_grid_coverage.csv` | {len(set((m,t,l,pid) for m,t,l,pid,*_ in by_cell))} | (model, task, tier, prompt_id) × priority indicator. Cell = n ratios measured; blank = not run |",
        f"| `paper_grid_results.csv` | — | Same row keys; columns = priority×ratio mean_kl. Blank = not measured |",
        f"| `paper_cell_stats.csv` | {n_cs} | Cell-level aggregate from master_audit/stats.json |",
        f"| `paper_selector_cells.csv` | {n_sel} | 264-cell selector benchmark (paired SnapKV+random data) |",
        f"| `paper_diffuseness.csv` | {n_diff} | Per-prompt participation ratio / mass coverage / entropy |",
        "",
        "## Summary",
        "",
        f"- **Runs**: {n_runs} `*_merged/` producer outputs",
        f"- **Prompt-rows**: {n_prompts} (a prompt run under one priority counts once; a prompt run under 4 priorities counts 4 times)",
        f"- **Distinct prompt_ids**: {n_unique_prompts}",
        f"- **Measurements (trajectories)**: {n_meas}",
        "",
        "### Rows per model",
        "",
        "| model | prompt-rows |",
        "|---|---:|",
    ]
    for m, n in by_model:
        lines.append(f"| {m} | {n} |")

    lines += [
        "",
        "## Query examples",
        "",
        "```sql",
        "-- Every prompt that was run under both snapkv AND random at 131k",
        "SELECT DISTINCT p.prompt_id, p.task, r.model",
        "FROM prompts p JOIN runs r ON p.run_id=r.run_id",
        "WHERE p.tier_length=131072",
        "  AND p.prompt_id IN (SELECT prompt_id FROM prompts p2 JOIN runs r2 ON p2.run_id=r2.run_id WHERE r2.priority='snapkv' AND p2.tier_length=131072)",
        "  AND p.prompt_id IN (SELECT prompt_id FROM prompts p2 JOIN runs r2 ON p2.run_id=r2.run_id WHERE r2.priority='random' AND p2.tier_length=131072);",
        "",
        "-- KL by priority on Phi at 131k, r=0.04 (priority_inversion table)",
        "SELECT r.priority, m.ratio, AVG(m.mean_kl), COUNT(*)",
        "FROM measurements m JOIN prompts p ON m.prompt_row_id=p.prompt_row_id",
        "  JOIN runs r ON p.run_id=r.run_id",
        "WHERE r.model='phi_3p5_mini_instruct' AND p.tier_length=131072 AND m.ratio=0.04",
        "GROUP BY r.priority, m.ratio;",
        "",
        "-- Per-prompt PR joined to per-prompt KL gap (E3 within-Phi regression)",
        "SELECT d.prompt_id, d.participation_ratio,",
        "       AVG(CASE WHEN r.priority='snapkv' THEN m.mean_kl END) AS kl_snapkv,",
        "       AVG(CASE WHEN r.priority='random' THEN m.mean_kl END) AS kl_random",
        "FROM diffuseness d",
        "JOIN prompts p ON d.prompt_id=p.prompt_id AND d.tier_length=p.tier_length",
        "JOIN runs r ON p.run_id=r.run_id",
        "JOIN measurements m ON m.prompt_row_id=p.prompt_row_id",
        "WHERE d.model='phi_3p5' AND m.ratio=0.04",
        "GROUP BY d.prompt_id;",
        "```",
        "",
        "## Source pull collections",
        "",
    ]
    for pull in PAPER_PULL_ROOTS:
        lines.append(f"- `analyses/{pull}/`")

    (OUT_DIR / "paper_data_README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    merged_dirs = discover_merged_dirs()
    print(f"[build_central_db] discovered {len(merged_dirs)} paper-canonical merged dirs", flush=True)

    counts = {"prompts": 0, "measurements": 0, "runs": 0}
    for i, d in enumerate(merged_dirs):
        try:
            np_, nm = ingest_run(con, d)
        except Exception as e:
            print(f"[build_central_db] SKIP {d}: {e}", file=sys.stderr)
            continue
        counts["prompts"] += np_
        counts["measurements"] += nm
        counts["runs"] += 1
        if (i + 1) % 50 == 0:
            print(f"  …ingested {i+1}/{len(merged_dirs)} dirs", flush=True)
    con.commit()
    print(f"[build_central_db] runs={counts['runs']} prompt-rows={counts['prompts']} measurements={counts['measurements']}", flush=True)

    cs, diff, sel = ingest_master_audit(con)
    print(f"[build_central_db] master_audit: cell_stats={cs}  diffuseness={diff}  selector_cells={sel}", flush=True)
    con.commit()

    print("[build_central_db] writing CSV exports...", flush=True)
    export_csvs(con)
    write_readme(con, counts)
    con.close()

    print(f"[build_central_db] done. db={DB_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
