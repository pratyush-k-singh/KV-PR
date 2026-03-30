"""Tests for src/dart_pagedkv/ruler_manifest.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dart_pagedkv.ruler_manifest import RulerPromptRecord, load_manifest_records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestLoadManifestRecords(unittest.TestCase):
    def _build_manifest(self, root: Path) -> Path:
        short = root / "ruler_l2048" / "official_ruler.jsonl"
        short.parent.mkdir(parents=True)
        _write_jsonl(short, [
            {"prompt_id": "cwe_s_0", "prompt": "short cwe text",
             "answers": ["a"], "metadata": {"official_ruler_task": "cwe"}},
            {"prompt_id": "vt_s_0", "prompt": "short vt text",
             "answers": ["b"], "metadata": {"official_ruler_task": "vt"}},
        ])
        long = root / "ruler_l131072" / "official_ruler.jsonl"
        long.parent.mkdir(parents=True)
        _write_jsonl(long, [
            {"prompt_id": "cwe_l_0", "prompt": "long cwe text",
             "answers": ["c"], "metadata": {"official_ruler_task": "cwe"}},
        ])
        manifest = {
            "tiers": [
                {"length": 2048, "ruler_jsonl": str(short),
                 "prompt_ids": ["cwe_s_0", "vt_s_0"]},
                {"length": 131072, "ruler_jsonl": str(long),
                 "prompt_ids": ["cwe_l_0"]},
            ],
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_loads_all_tiers_short_to_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = load_manifest_records(self._build_manifest(Path(tmp)))
        self.assertEqual(
            [r.prompt_id for r in records], ["cwe_s_0", "vt_s_0", "cwe_l_0"]
        )
        self.assertEqual([r.tier_length for r in records], [2048, 2048, 131072])
        self.assertEqual([r.task for r in records], ["cwe", "vt", "cwe"])
        self.assertEqual(records[0].prompt, "short cwe text")
        self.assertEqual(records[2].answers, ["c"])
        self.assertIsInstance(records[0], RulerPromptRecord)

    def test_missing_prompt_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "r" / "official_ruler.jsonl"
            jsonl.parent.mkdir(parents=True)
            _write_jsonl(jsonl, [
                {"prompt_id": "exists", "prompt": "p", "metadata": {}},
            ])
            manifest = root / "m.json"
            manifest.write_text(json.dumps({
                "tiers": [{"length": 2048, "ruler_jsonl": str(jsonl),
                           "prompt_ids": ["does_not_exist"]}],
            }), encoding="utf-8")
            with self.assertRaises(KeyError):
                load_manifest_records(manifest)

    def test_missing_task_metadata_defaults_to_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "r" / "official_ruler.jsonl"
            jsonl.parent.mkdir(parents=True)
            _write_jsonl(jsonl, [{"prompt_id": "p0", "prompt": "p"}])
            manifest = root / "m.json"
            manifest.write_text(json.dumps({
                "tiers": [{"length": 4096, "ruler_jsonl": str(jsonl),
                           "prompt_ids": ["p0"]}],
            }), encoding="utf-8")
            records = load_manifest_records(manifest)
        self.assertEqual(records[0].task, "unknown")
        self.assertEqual(records[0].answers, [])


if __name__ == "__main__":
    unittest.main()
