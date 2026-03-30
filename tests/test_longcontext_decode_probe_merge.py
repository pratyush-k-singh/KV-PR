from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.longcontext_decode_probe_merge import (
    merge_payloads,
    write_outputs,
)


def _payload(model: str, prompts: list[dict]) -> dict:
    return {
        "manifest": "m.json",
        "model": model,
        "chat_template": "never",
        "k_dec": 16,
        "obs_window": 32,
        "num_groups": 4,
        "n_sink": 4,
        "n_recent": 64,
        "ratios": [0.01, 0.02],
        "support_estimator": "mass_coverage",
        "support_tau": 0.95,
        "prompts_attempted": len(prompts),
        "prompts_ok": sum(1 for p in prompts if p.get("status") == "ok"),
        "prompts": prompts,
    }


class TestLongContextDecodeProbeMerge(unittest.TestCase):
    def test_merges_compatible_payloads_sorted_by_tier(self):
        a = _payload("model-a", [
            {"prompt_id": "p32", "tier_length": 32768, "task": "cwe", "status": "ok"},
        ])
        b = _payload("model-a", [
            {"prompt_id": "p2", "tier_length": 2048, "task": "cwe", "status": "ok"},
            {"prompt_id": "p4", "tier_length": 4096, "task": "vt", "status": "OOM"},
        ])

        merged = merge_payloads([a, b])

        self.assertEqual(merged["model"], "model-a")
        self.assertEqual(merged["prompts_attempted"], 3)
        self.assertEqual(merged["prompts_ok"], 2)
        self.assertEqual(
            [p["prompt_id"] for p in merged["prompts"]],
            ["p2", "p4", "p32"],
        )

    def test_rejects_incompatible_payloads(self):
        a = _payload("model-a", [])
        b = _payload("model-b", [])
        with self.assertRaisesRegex(ValueError, "differs on model"):
            merge_payloads([a, b])

    def test_rejects_duplicate_prompt_ids(self):
        a = _payload("model-a", [
            {"prompt_id": "same", "tier_length": 2048, "task": "cwe", "status": "ok"},
        ])
        b = _payload("model-a", [
            {"prompt_id": "same", "tier_length": 4096, "task": "cwe", "status": "ok"},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate prompt_ids"):
            merge_payloads([a, b])

    def test_write_outputs_writes_eval_config_and_trajectories(self):
        payload = merge_payloads([
            _payload("model-a", [
                {"prompt_id": "p", "tier_length": 2048, "task": "cwe", "status": "ok"},
            ])
        ])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            write_outputs(out, payload)
            self.assertTrue((out / "trajectories.json").exists())
            self.assertTrue((out / "eval_config.json").exists())
            eval_config = json.loads((out / "eval_config.json").read_text())
            self.assertNotIn("prompts", eval_config)


if __name__ == "__main__":
    unittest.main()
