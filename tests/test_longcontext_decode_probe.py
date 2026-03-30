"""Tests for experiments/scripts/longcontext_decode_probe.py.

Only the pure ``parse_ratios`` helper is unit-tested. The per-prompt
pipeline (`_run_prompt`, `group_windowed_priorities`) is GPU- and
HF-coupled — capture prefill, FlexAttention masked decode — and is
validated server-side by the Stage-A run; the windowed-priority math
it relies on is already gated in test_dart_pagedkv_windowed_priority.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.longcontext_decode_probe import (
    _trajectories_payload,
    _write_json_atomic,
    parse_ratios,
    protected_budget_floors,
    retained_coverage_metrics,
    select_group_priorities,
)


class TestParseRatios(unittest.TestCase):
    def test_parses_comma_separated(self):
        self.assertEqual(
            parse_ratios("0.005,0.01,0.02,0.04,0.08"),
            [0.005, 0.01, 0.02, 0.04, 0.08],
        )

    def test_tolerates_whitespace_and_blanks(self):
        self.assertEqual(parse_ratios(" 0.01, 0.02 ,"), [0.01, 0.02])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_ratios("")


class TestProtectedBudgetFloors(unittest.TestCase):
    def test_counts_sink_and_recent_for_every_group(self):
        self.assertEqual(
            protected_budget_floors(
                seq_len=2048, num_groups=4, n_sink=4, n_recent=64
            ),
            [68, 68, 68, 68],
        )

    def test_deduplicates_overlap_on_short_sequences(self):
        self.assertEqual(
            protected_budget_floors(
                seq_len=10, num_groups=3, n_sink=4, n_recent=8
            ),
            [10, 10, 10],
        )


class TestTrajectoriesPayload(unittest.TestCase):
    def test_merges_eval_config_and_counts_statuses(self):
        cfg = {"model": "m", "k_dec": 16}
        results = [
            {"prompt_id": "a", "status": "ok"},
            {"prompt_id": "b", "status": "OutOfMemoryError"},
            {"prompt_id": "c", "status": "ok"},
        ]
        payload = _trajectories_payload(cfg, results)
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["k_dec"], 16)
        self.assertEqual(payload["prompts_attempted"], 3)
        self.assertEqual(payload["prompts_ok"], 2)
        self.assertEqual(payload["prompts"], results)

    def test_empty_results_is_a_valid_payload(self):
        payload = _trajectories_payload({"model": "m"}, [])
        self.assertEqual(payload["prompts_attempted"], 0)
        self.assertEqual(payload["prompts_ok"], 0)
        self.assertEqual(payload["prompts"], [])


class TestWriteJsonAtomic(unittest.TestCase):
    def test_writes_readable_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trajectories.json"
            _write_json_atomic(path, {"a": 1, "b": [2, 3]})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"a": 1, "b": [2, 3]},
            )

    def test_overwrites_an_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trajectories.json"
            _write_json_atomic(path, {"v": 1})
            _write_json_atomic(path, {"v": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), {"v": 2}
            )

    def test_leaves_no_temp_file_behind(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trajectories.json"
            _write_json_atomic(path, {"v": 1})
            self.assertEqual(
                sorted(p.name for p in Path(d).iterdir()), ["trajectories.json"]
            )


class TestSelectGroupPriorities(unittest.TestCase):
    """select_group_priorities dispatches the within-group priority for
    the spec §8.3 control — recent/random need no Q/K trace and are pure
    CPU index math, so they are testable without a model."""

    def test_recent_is_replicated_across_groups_and_ascending(self):
        priorities = select_group_priorities(
            None, {}, num_groups=3, obs_window=32, seq_len=5,
            priority="recent", seed=0, device="cpu",
        )
        self.assertEqual(len(priorities), 3)
        for g in priorities:
            self.assertEqual(list(g), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_random_is_seeded_and_differs_per_group(self):
        priorities = select_group_priorities(
            None, {}, num_groups=2, obs_window=32, seq_len=8,
            priority="random", seed=11, device="cpu",
        )
        self.assertEqual(len(priorities), 2)
        self.assertEqual(len(priorities[0]), 8)
        self.assertNotEqual(priorities[0], priorities[1])

    def test_random_is_reproducible_across_calls(self):
        a = select_group_priorities(None, {}, num_groups=2, obs_window=32,
                                    seq_len=8, priority="random", seed=11,
                                    device="cpu")
        b = select_group_priorities(None, {}, num_groups=2, obs_window=32,
                                    seq_len=8, priority="random", seed=11,
                                    device="cpu")
        self.assertEqual(a, b)

    def test_shared_random_broadcasts_one_draw_to_all_groups(self):
        priorities = select_group_priorities(
            None, {}, num_groups=3, obs_window=32, seq_len=8,
            priority="shared_random", seed=11, device="cpu",
        )
        self.assertEqual(priorities[0], priorities[1])
        self.assertEqual(priorities[1], priorities[2])
        self.assertNotEqual(
            priorities[0],
            random_priority := select_group_priorities(
                None, {}, num_groups=1, obs_window=32, seq_len=8,
                priority="random", seed=12, device="cpu",
            )[0],
        )
        self.assertEqual(len(random_priority), 8)

    def test_unknown_priority_raises(self):
        with self.assertRaises(ValueError):
            select_group_priorities(None, {}, num_groups=2, obs_window=32,
                                    seq_len=5, priority="oracle", seed=0,
                                    device="cpu")


class TestRetainedCoverageMetrics(unittest.TestCase):
    def test_union_overlap_and_reference_mass(self):
        metrics = retained_coverage_metrics(
            hot_per_group=[[0, 1], [1, 2]],
            reference_priorities=[
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.25, 0.75, 0.0],
            ],
            seq_len=4,
        )
        self.assertEqual(metrics["union_size"], 3)
        self.assertEqual(metrics["intersection_size"], 1)
        self.assertAlmostEqual(metrics["mean_pairwise_jaccard"], 1 / 3)
        self.assertEqual(metrics["per_group_hot_counts"], [2, 2])
        self.assertEqual(metrics["per_group_retained_reference_mass"], [1.0, 1.0])
        self.assertAlmostEqual(metrics["union_fraction_of_context"], 0.75)
        self.assertAlmostEqual(metrics["union_fraction_of_group_budget_sum"], 0.75)


class _MockLayer:
    """Mock LayerQKCapture-input that `_layer_qkv_to_capture` accepts."""

    def __init__(self, query, key, scaling, num_key_value_groups):
        self.query = query
        self.key = key
        self.scaling = scaling
        self.num_key_value_groups = num_key_value_groups


class _MockTrace:
    """Mock trace.layers map for the k_norm / accumulated dispatch tests."""

    def __init__(self, layers):
        self.layers = layers


class TestSelectGroupPrioritiesKNormAccumulated(unittest.TestCase):
    """k_norm and accumulated dispatch from a real (small) trace through
    `_per_group_captures` → priority function. The math is gated in
    test_dart_pagedkv_windowed_priority; this just confirms the dispatcher."""

    @staticmethod
    def _trace_and_map():
        import torch
        torch.manual_seed(0)
        n_heads, n_kv_heads, seq_len, head_dim = 4, 2, 12, 8
        # 4 layers, 2 groups of 2 layers each.
        layers = {
            i: _MockLayer(
                query=torch.randn(n_heads, seq_len, head_dim),
                key=torch.randn(n_kv_heads, seq_len, head_dim),
                scaling=head_dim ** -0.5,
                num_key_value_groups=n_heads // n_kv_heads,
            )
            for i in range(4)
        }
        layer_to_group = {0: 0, 1: 0, 2: 1, 3: 1}
        return _MockTrace(layers), layer_to_group, seq_len

    def test_k_norm_returns_one_vector_per_group(self):
        trace, layer_to_group, seq_len = self._trace_and_map()
        priorities = select_group_priorities(
            trace, layer_to_group, num_groups=2, obs_window=4, seq_len=seq_len,
            priority="k_norm", seed=0, device="cpu",
        )
        self.assertEqual(len(priorities), 2)
        for vec in priorities:
            self.assertEqual(len(vec), seq_len)
        # Two groups have distinct layers → distinct k_norm priorities.
        self.assertNotEqual(priorities[0], priorities[1])

    def test_accumulated_returns_one_vector_per_group(self):
        trace, layer_to_group, seq_len = self._trace_and_map()
        priorities = select_group_priorities(
            trace, layer_to_group, num_groups=2, obs_window=4, seq_len=seq_len,
            priority="accumulated", seed=0, device="cpu",
            accumulated_chunk_size=4,
        )
        self.assertEqual(len(priorities), 2)
        for vec in priorities:
            self.assertEqual(len(vec), seq_len)
            # Every priority entry non-negative (sum of softmax weights).
            for v in vec:
                self.assertGreaterEqual(v, -1e-6)


if __name__ == "__main__":
    unittest.main()
