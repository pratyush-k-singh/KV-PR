#!/usr/bin/env python3
"""RULER length-sweep manifest generator (spec build-sequence step 5).

Consumes pre-prepared RULER outputs (one per length tier; produced by
`cli.py prepare-official-ruler --max-seq-length L --out-dir ...`) and
writes a single tiered manifest the long-context FlexAttention runner
will read. Mirrors `experiments/manifests/decode_probe_sub8k_2026-05-16.json`
in spirit, but the strata are *length tiers*, not regret strata: the
goal of the long-context sweep is the kernel/method behavior across
context length, not mechanism-of-failure replay.

Selection rule: within each tier, round-robin across RULER's tasks
(``metadata.official_ruler_task``), then sort by ``prompt_id`` within a
task. Round-robin gets task diversity ahead of per-task density when
``prompts-per-tier`` is small (a typical first sweep is 5–11 prompts per
tier; RULER ships 11 tasks). Pure deterministic — no randomness, no
seed.

The manifest does NOT generate RULER prompts (that is
`prepare-official-ruler`, which clones NVIDIA/RULER and tokenizes
context-length samples — slow at 131k, server-side, one prep per length).
This script only selects, so it is cheap and reproducible.

Spec: docs/superpowers/specs/2026-05-16-flexattention-longcontext.md §6, §8.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from pathlib import Path


def _read_ruler_dir(ruler_dir: Path) -> tuple[int, str, list[dict]]:
    """Return ``(max_seq_length, jsonl_path, records)`` for a prepared dir.

    ``official_ruler_manifest.json`` carries the length; the JSONL
    carries one record per prompt.
    """
    manifest_path = ruler_dir / "official_ruler_manifest.json"
    jsonl_path = ruler_dir / "official_ruler.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run `cli.py prepare-official-ruler "
            f"--out-dir {ruler_dir}` first."
        )
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"{jsonl_path} not found (expected alongside the manifest)."
        )
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_seq_length = int(meta["max_seq_length"])
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return max_seq_length, str(jsonl_path), records


def select_round_robin(
    records: Sequence[dict],
    n: int,
    *,
    offset: int = 0,
    stride: int = 1,
    task: str | None = None,
) -> list[dict]:
    """Pick ``n`` records, round-robin across RULER tasks.

    Records are grouped by ``metadata.official_ruler_task``; within a
    task they are sorted by ``prompt_id``. The base output is task-rotated:
    one prompt from each task in turn, looping until every task is exhausted.
    ``offset``/``stride`` then slice that deterministic order so prompt shards
    can run in parallel without overlapping. With the defaults the historical
    behavior is unchanged.

    With ``task`` set, selection is instead restricted to that single
    RULER task (records sorted by ``prompt_id``), and ``offset``/``stride``
    shard *within* the task — the within-task replication path (many
    prompts of one task, e.g. to characterise a tail task's distribution)
    rather than the default cross-task breadth.
    """
    if n <= 0:
        return []
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        t = r.get("metadata", {}).get("official_ruler_task", "unknown")
        by_task[t].append(r)
    for t in by_task:
        by_task[t].sort(key=lambda r: r["prompt_id"])
    if task is not None:
        ordered: list[dict] = list(by_task.get(task, []))
    else:
        tasks_sorted = sorted(by_task)
        ordered = []
        cursors: dict[str, int] = {t: 0 for t in tasks_sorted}
        while True:
            progressed = False
            for t in tasks_sorted:
                i = cursors[t]
                if i < len(by_task[t]):
                    ordered.append(by_task[t][i])
                    cursors[t] = i + 1
                    progressed = True
            if not progressed:
                break
    return ordered[offset::stride][:n]


def build_manifest(
    tiers_in: Sequence[tuple[Path, int]],
    prompts_per_tier: int,
    *,
    prompt_offset: int = 0,
    prompt_stride: int = 1,
    task: str | None = None,
) -> dict:
    """Assemble the length-sweep manifest from prepared RULER dirs.

    ``tiers_in`` is ``(ruler_dir, expected_length)`` pairs. The expected
    length must match the dir's ``official_ruler_manifest.json``
    ``max_seq_length`` (guards against directory typos). ``task`` (when
    set) restricts every tier to that single RULER task. Returns the
    manifest dict; the caller writes JSON.
    """
    tiers_out = []
    flat_ids: list[str] = []
    for ruler_dir, expected_length in tiers_in:
        max_seq_length, jsonl_path, records = _read_ruler_dir(ruler_dir)
        if max_seq_length != expected_length:
            raise ValueError(
                f"{ruler_dir}: prepared at max_seq_length={max_seq_length}, "
                f"but the tier was declared as {expected_length}. Fix the "
                f"--tier argument or re-prepare the dir."
            )
        picked = select_round_robin(
            records, prompts_per_tier,
            offset=prompt_offset,
            stride=prompt_stride,
            task=task,
        )
        picked_ids = [r["prompt_id"] for r in picked]
        tiers_out.append({
            "length": max_seq_length,
            "ruler_dir": str(ruler_dir),
            "ruler_jsonl": jsonl_path,
            "prompts_per_tier_requested": prompts_per_tier,
            "prompt_offset": prompt_offset,
            "prompt_stride": prompt_stride,
            "task_filter": task,
            "prompt_ids": picked_ids,
            "tasks": sorted({
                r.get("metadata", {}).get("official_ruler_task", "unknown")
                for r in picked
            }),
        })
        flat_ids.extend(picked_ids)
    tiers_out.sort(key=lambda t: t["length"])
    flat_ids = [pid for tier in tiers_out for pid in tier["prompt_ids"]]
    return OrderedDict([
        ("spec", "docs/superpowers/specs/2026-05-16-flexattention-longcontext.md"),
        ("stage", "Stage 1 long-context FlexAttention length sweep"),
        ("data_source", "ruler_official"),
        ("selection_rule",
         "round-robin across RULER tasks within each length tier; "
         "tasks sorted, prompts within a task sorted by prompt_id; "
         "deterministic, no seed; optional offset/stride prompt sharding"),
        ("tiers", tiers_out),
        ("prompt_ids", flat_ids),
    ])


def _parse_tier(spec: str) -> tuple[Path, int]:
    """Parse ``--tier <length>=<dir>`` (e.g. ``16384=data/ruler_l16k``)."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--tier expects '<length>=<dir>' (e.g. '16384=data/ruler_l16k'), "
            f"got: {spec!r}"
        )
    length_str, dir_str = spec.split("=", 1)
    return Path(dir_str), int(length_str)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ruler-length-sweep-manifest")
    p.add_argument(
        "--tier", action="append", type=_parse_tier, required=True,
        metavar="LENGTH=DIR",
        help="Repeated. e.g. --tier 16384=data/ruler_l16k --tier 32768=data/ruler_l32k",
    )
    p.add_argument(
        "--prompts-per-tier", type=int, default=5,
        help="Prompts to select per length tier (default 5; smoke usually 1-3).",
    )
    p.add_argument(
        "--prompt-offset", type=int, default=0,
        help="Shard offset into the deterministic round-robin order (default 0).",
    )
    p.add_argument(
        "--prompt-stride", type=int, default=1,
        help="Shard stride through the deterministic round-robin order (default 1).",
    )
    p.add_argument(
        "--task", default=None,
        help="Restrict selection to a single RULER task (e.g. 'vt') for "
             "within-task replication; default is round-robin across tasks.",
    )
    p.add_argument("--out", type=Path, required=True,
                   help="Output manifest JSON path.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    manifest = build_manifest(
        args.tier, args.prompts_per_tier,
        prompt_offset=args.prompt_offset,
        prompt_stride=args.prompt_stride,
        task=args.task,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[ruler-sweep] wrote {args.out} — "
          f"{len(manifest['tiers'])} tiers, "
          f"{len(manifest['prompt_ids'])} prompts total", flush=True)
    for tier in manifest["tiers"]:
        print(f"[ruler-sweep]   length={tier['length']:>7d}: "
              f"{len(tier['prompt_ids'])} prompts across "
              f"{len(tier['tasks'])} tasks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
