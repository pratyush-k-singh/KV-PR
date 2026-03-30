"""Load a RULER length-sweep manifest into runner-ready records.

The tiered manifest (from the `ruler-length-sweep-manifest` CLI) lists,
per length tier, a RULER JSONL path and the selected prompt_ids. The
long-context decode-probe runner needs each prompt's text, task label
and tier length — this module performs that lookup and adaptation so
the GPU runner stays free of manifest/JSONL plumbing.

Spec: docs/superpowers/specs/2026-05-17-longcontext-decode-probe.md §4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RulerPromptRecord:
    """One RULER prompt, adapted for the long-context decode-probe runner."""

    prompt_id: str
    prompt: str
    task: str
    tier_length: int
    answers: list[str] = field(default_factory=list)


def _scan_jsonl_for_id(jsonl_path: Path, prompt_id: str) -> dict:
    """Return the JSONL record whose ``prompt_id`` matches; raise if absent.

    A linear scan — a length-sweep manifest names at most a few dozen
    prompts per tier, so an index would be premature.
    """
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("prompt_id") == prompt_id:
                return record
    raise KeyError(f"prompt_id {prompt_id!r} not found in {jsonl_path}")


def load_manifest_records(manifest_path: str | Path) -> list[RulerPromptRecord]:
    """Load every prompt named in the tiered manifest, ordered short→long.

    The manifest's ``tiers`` are length-sorted by the generator; records
    are emitted tier by tier, so the runner processes short prompts
    first — fail-fast at long context (parent spec §8.6 ordering).
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[RulerPromptRecord] = []
    for tier in manifest["tiers"]:
        jsonl_path = Path(tier["ruler_jsonl"])
        tier_length = int(tier["length"])
        for prompt_id in tier["prompt_ids"]:
            raw = _scan_jsonl_for_id(jsonl_path, prompt_id)
            records.append(RulerPromptRecord(
                prompt_id=raw["prompt_id"],
                prompt=raw["prompt"],
                task=raw.get("metadata", {}).get(
                    "official_ruler_task", "unknown"
                ),
                tier_length=tier_length,
                answers=list(raw.get("answers", [])),
            ))
    return records
