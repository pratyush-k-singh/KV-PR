#!/usr/bin/env python3
"""Merge split long-context decode-probe outputs.

Stage-A runs are expensive because every prompt expands into many
``(ratio, allocation, decode-step)`` masked forwards. The practical way to
make the sweep feasible on the H100 partition is to run independent
model/tier shards in parallel, then merge their ``trajectories.json`` files
back into the producer format the existing analyzer already understands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MERGE_MATCH_KEYS = (
    "model",
    "chat_template",
    "k_dec",
    "obs_window",
    "num_groups",
    "n_sink",
    "n_recent",
    "ratios",
    "support_estimator",
    "support_tau",
    # Priority sweep (spec §8.3): a snapkv shard and a recent shard share
    # everything else but are not interchangeable — they answer different
    # questions. Refuse to merge across priorities.
    "priority",
    # Published-allocator extras (status §17): a shard with extra_allocators
    # holds different per-prompt trajectory counts than one without —
    # refuse to merge across mismatched allocator sets.
    "extra_allocators",
)


def _read_payload(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "trajectories.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "prompts" not in data:
        raise ValueError(f"{path} does not look like a trajectories.json file")
    return data


def _merged_manifest(manifests: list[Any]) -> Any:
    flat: list[Any] = []
    for manifest in manifests:
        if isinstance(manifest, list):
            flat.extend(manifest)
        else:
            flat.append(manifest)
    seen: set[str] = set()
    out: list[Any] = []
    for item in flat:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[0] if len(out) == 1 else out


def merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Return one trajectories payload from compatible shard payloads."""
    if not payloads:
        raise ValueError("no inputs to merge")
    base = dict(payloads[0])
    for idx, data in enumerate(payloads[1:], start=2):
        for key in MERGE_MATCH_KEYS:
            if base.get(key) != data.get(key):
                raise ValueError(
                    f"input {idx} differs on {key}: "
                    f"{base.get(key)!r} != {data.get(key)!r}"
                )

    prompts: list[dict[str, Any]] = []
    seen_prompt_ids: set[str] = set()
    duplicates: list[str] = []
    for data in payloads:
        for prompt in data.get("prompts", []):
            prompt_id = str(prompt.get("prompt_id"))
            if prompt_id in seen_prompt_ids:
                duplicates.append(prompt_id)
                continue
            seen_prompt_ids.add(prompt_id)
            prompts.append(prompt)
    if duplicates:
        dup_preview = ", ".join(sorted(duplicates)[:8])
        raise ValueError(f"duplicate prompt_ids across shards: {dup_preview}")

    prompts.sort(key=lambda p: (
        int(p.get("tier_length") or 0),
        str(p.get("task") or ""),
        str(p.get("prompt_id") or ""),
    ))

    merged = {
        key: value
        for key, value in base.items()
        if key not in {"prompts", "prompts_attempted", "prompts_ok"}
    }
    merged["manifest"] = _merged_manifest([p.get("manifest") for p in payloads])
    merged["prompts_attempted"] = len(prompts)
    merged["prompts_ok"] = sum(1 for p in prompts if p.get("status") == "ok")
    merged["prompts"] = prompts
    return merged


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    eval_config = {
        key: value
        for key, value in payload.items()
        if key not in {"prompts", "prompts_attempted", "prompts_ok"}
    }
    (out_dir / "eval_config.json").write_text(
        json.dumps(eval_config, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="longcontext-decode-probe-merge")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Merged analysis directory to create/update.")
    p.add_argument("inputs", nargs="+", type=Path,
                   help="Shard directories or trajectories.json files.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    payload = merge_payloads([_read_payload(path) for path in args.inputs])
    write_outputs(args.out_dir, payload)
    print(
        f"[lc-merge] wrote {args.out_dir}/ "
        f"({payload['prompts_ok']}/{payload['prompts_attempted']} ok prompts)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
