"""Tests for experiments/scripts/ruler_length_sweep_manifest.py.

The pure ``select_round_robin`` selector and the ``build_manifest`` glue
are unit-tested here. The full prepare-RULER round-trip is server-side
(needs the NVIDIA RULER repo + a tokenizer) and is not in scope.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.ruler_length_sweep_manifest import (
    build_manifest,
    select_round_robin,
)


def _rec(prompt_id: str, task: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "metadata": {"official_ruler_task": task},
    }


class TestSelectRoundRobin(unittest.TestCase):
    def test_zero_returns_empty(self):
        self.assertEqual(select_round_robin([_rec("a", "t1")], 0), [])

    def test_round_robin_across_tasks(self):
        records = [
            _rec("niah_0001", "niah_single_1"),
            _rec("niah_0002", "niah_single_1"),
            _rec("vt_0001", "vt"),
            _rec("cwe_0001", "cwe"),
        ]
        picked = select_round_robin(records, 3)
        # tasks sorted: cwe, niah_single_1, vt → one from each in that order
        self.assertEqual(
            [r["prompt_id"] for r in picked],
            ["cwe_0001", "niah_0001", "vt_0001"],
        )

    def test_wraps_when_n_exceeds_task_count(self):
        records = [
            _rec("a1", "t_a"),
            _rec("a2", "t_a"),
            _rec("b1", "t_b"),
        ]
        # 3 tasks would be (a1, b1); 4th must come from second pass on t_a → a2.
        picked = select_round_robin(records, 3)
        self.assertEqual([r["prompt_id"] for r in picked], ["a1", "b1", "a2"])

    def test_stops_when_records_exhausted(self):
        records = [_rec("only_one", "t1")]
        picked = select_round_robin(records, 5)
        self.assertEqual([r["prompt_id"] for r in picked], ["only_one"])

    def test_deterministic_sort_within_task(self):
        records = [
            _rec("z_late", "t1"),
            _rec("a_early", "t1"),
            _rec("m_mid", "t1"),
        ]
        picked = select_round_robin(records, 3)
        self.assertEqual(
            [r["prompt_id"] for r in picked],
            ["a_early", "m_mid", "z_late"],
        )

    def test_offset_stride_shards_round_robin_order(self):
        records = [
            _rec("a1", "a"),
            _rec("a2", "a"),
            _rec("b1", "b"),
            _rec("b2", "b"),
            _rec("c1", "c"),
        ]
        picked = select_round_robin(records, 2, offset=1, stride=2)
        # Base order is a1, b1, c1, a2, b2; offset/stride gives b1, a2.
        self.assertEqual([r["prompt_id"] for r in picked], ["b1", "a2"])

    def test_task_filter_restricts_to_one_task_sorted(self):
        # Within-task replication: --task picks only that task, prompt_id-sorted.
        records = [
            _rec("vt_0003", "vt"),
            _rec("cwe_0001", "cwe"),
            _rec("vt_0001", "vt"),
            _rec("vt_0002", "vt"),
        ]
        picked = select_round_robin(records, 5, task="vt")
        self.assertEqual(
            [r["prompt_id"] for r in picked],
            ["vt_0001", "vt_0002", "vt_0003"],
        )

    def test_task_filter_shards_within_the_task(self):
        records = [_rec(f"vt_{i:04d}", "vt") for i in range(6)] + [_rec("cwe_0", "cwe")]
        picked = select_round_robin(records, 2, task="vt", offset=1, stride=2)
        # vt sorted vt_0000..vt_0005; offset 1 stride 2 -> vt_0001, vt_0003.
        self.assertEqual([r["prompt_id"] for r in picked], ["vt_0001", "vt_0003"])

    def test_task_filter_unknown_task_is_empty(self):
        self.assertEqual(select_round_robin([_rec("a", "t1")], 3, task="nope"), [])


class TestBuildManifest(unittest.TestCase):
    def _write_tier(self, root: Path, length: int, records: list[dict]) -> Path:
        d = root / f"ruler_l{length}"
        d.mkdir(parents=True)
        (d / "official_ruler_manifest.json").write_text(
            json.dumps({"max_seq_length": length}), encoding="utf-8"
        )
        with (d / "official_ruler.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return d

    def test_two_tiers_short_to_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d_short = self._write_tier(root, 16384, [
                _rec("s_niah_0001", "niah_single_1"),
                _rec("s_vt_0001", "vt"),
            ])
            d_long = self._write_tier(root, 131072, [
                _rec("l_niah_0001", "niah_single_1"),
                _rec("l_vt_0001", "vt"),
            ])
            manifest = build_manifest(
                [(d_long, 131072), (d_short, 16384)],  # out-of-order on purpose
                prompts_per_tier=1,
            )
        self.assertEqual([t["length"] for t in manifest["tiers"]], [16384, 131072])
        self.assertEqual(manifest["prompt_ids"], ["s_niah_0001", "l_niah_0001"])
        self.assertEqual(manifest["data_source"], "ruler_official")

    def test_build_manifest_records_prompt_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = self._write_tier(root, 2048, [
                _rec("a1", "a"),
                _rec("b1", "b"),
                _rec("a2", "a"),
            ])
            manifest = build_manifest(
                [(d, 2048)], prompts_per_tier=1,
                prompt_offset=1, prompt_stride=2,
            )
        self.assertEqual(manifest["tiers"][0]["prompt_offset"], 1)
        self.assertEqual(manifest["tiers"][0]["prompt_stride"], 2)
        self.assertEqual(manifest["prompt_ids"], ["b1"])

    def test_build_manifest_task_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = self._write_tier(root, 2048, [
                _rec("vt_2", "vt"), _rec("cwe_1", "cwe"), _rec("vt_1", "vt"),
            ])
            manifest = build_manifest([(d, 2048)], prompts_per_tier=5, task="vt")
        self.assertEqual(manifest["prompt_ids"], ["vt_1", "vt_2"])
        self.assertEqual(manifest["tiers"][0]["tasks"], ["vt"])
        self.assertEqual(manifest["tiers"][0]["task_filter"], "vt")

    def test_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = self._write_tier(root, 16384, [_rec("x", "t1")])
            with self.assertRaises(ValueError):
                build_manifest([(d, 32768)], prompts_per_tier=1)


if __name__ == "__main__":
    unittest.main()
