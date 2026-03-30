"""Tests for experiments/scripts/longcontext_decode_probe_analyze.py.

CPU-only — the analyzer is pure trajectory processing. Validates:
  * scenario picks (uniform / mincost_bad / oracle_bstar / worst_kl) on
    hand-constructed trajectory sets,
  * the per-token mean over decode steps 1..K-1 (step 0 = 0 skipped),
  * collapsed (tier, ratio) cells excluded from the aggregate.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.longcontext_decode_probe_analyze import (
    _pick_support,
    aggregate,
    analyze_prompt,
    bootstrap_median_ci,
    classify_allocation,
    heavy_tail_index,
    main,
    mean_per_step_kl,
    select_scenarios_for_ratio,
    tail_rate,
)


class TestMeanPerStepKL(unittest.TestCase):
    def test_skips_step_zero(self):
        # Step 0 is the shared prefill prediction (KL = 0 by construction);
        # the mean is over steps 1..K-1 only — avoids the §14.2 sum trap
        # diluting toward 0.
        self.assertEqual(mean_per_step_kl([0.0, 1.0, 3.0]), 2.0)

    def test_single_step_returns_zero(self):
        self.assertEqual(mean_per_step_kl([0.0]), 0.0)

    def test_handles_empty_list(self):
        self.assertEqual(mean_per_step_kl([]), 0.0)


class TestSelectScenarios(unittest.TestCase):
    def _t(self, allocation, cost, kls):
        return {"ratio": 0.05, "total": sum(allocation),
                "allocation": list(allocation), "cost": cost, "per_step_kl": kls}

    def test_identifies_all_four_scenarios(self):
        trajectories = [
            self._t([10, 10, 10, 10], cost=0.30, kls=[0.0, 0.5, 0.5, 0.5]),  # uniform
            self._t([40, 0, 0, 0],   cost=0.10, kls=[0.0, 2.0, 2.0, 2.0]),  # mincost
            self._t([0, 40, 0, 0],   cost=0.40, kls=[0.0, 0.1, 0.1, 0.1]),  # oracle
            self._t([0, 0, 0, 40],   cost=0.50, kls=[0.0, 5.0, 5.0, 5.0]),  # worst
        ]
        scenarios = select_scenarios_for_ratio(trajectories, [10, 10, 10, 10])
        self.assertEqual(scenarios["uniform"]["allocation"], [10, 10, 10, 10])
        self.assertEqual(scenarios["mincost_bad"]["allocation"], [40, 0, 0, 0])
        self.assertEqual(scenarios["oracle_bstar"]["allocation"], [0, 40, 0, 0])
        self.assertEqual(scenarios["worst_kl"]["allocation"], [0, 0, 0, 40])

    def test_collapsed_grid_all_scenarios_equal(self):
        single = self._t([68, 68, 68, 68], cost=0.20, kls=[0.0, 1.0, 1.0])
        scenarios = select_scenarios_for_ratio([single], [68, 68, 68, 68])
        for name in ("uniform", "mincost_bad", "oracle_bstar", "worst_kl"):
            self.assertEqual(scenarios[name]["allocation"], [68, 68, 68, 68])

    def test_uniform_not_found_returns_none(self):
        t = self._t([1, 2, 3, 4], cost=0.1, kls=[0.0, 1.0])
        scenarios = select_scenarios_for_ratio([t], [10, 10, 10, 10])
        self.assertIsNone(scenarios["uniform"])


class TestAnalyzePromptAndAggregate(unittest.TestCase):
    def _prompt_with_two_ratios(self):
        # 4 groups, n_sink=4, n_recent=4 -> floor_sum = 4*8 = 32 (over a long
        # enough prompt). At tokens=200, total = 0.5*200=100 -> discretionary
        # 68 (meaningful); at tokens=200, ratio 0.1 -> total 32 -> collapsed.
        return {
            "prompt_id": "p0", "task": "cwe", "tier_length": 200, "tokens": 200,
            "ratio_summaries": [
                {"ratio": 0.5, "total": 100, "discretionary": 68,
                 "n_distinct_allocations": 3, "collapsed": False},
                {"ratio": 0.1, "total": 32, "discretionary": 0,
                 "n_distinct_allocations": 1, "collapsed": True},
            ],
            "trajectories": [
                # ratio 0.5 — uniform_budget(4, 100, [8,8,8,8]) = [25,25,25,25]
                {"ratio": 0.5, "total": 100, "allocation": [25, 25, 25, 25],
                 "cost": 0.30, "per_step_kl": [0.0, 1.0, 1.0]},
                {"ratio": 0.5, "total": 100, "allocation": [76, 8, 8, 8],
                 "cost": 0.10, "per_step_kl": [0.0, 3.0, 3.0]},   # mincost & worst
                {"ratio": 0.5, "total": 100, "allocation": [8, 76, 8, 8],
                 "cost": 0.40, "per_step_kl": [0.0, 0.5, 0.5]},   # oracle
                # ratio 0.1 — single collapsed allocation
                {"ratio": 0.1, "total": 32, "allocation": [8, 8, 8, 8],
                 "cost": 0.20, "per_step_kl": [0.0, 2.0, 2.0]},
            ],
        }

    def test_analyze_prompt_deltas(self):
        analysed = analyze_prompt(
            self._prompt_with_two_ratios(),
            num_groups=4, n_sink=4, n_recent=4,
        )
        by_ratio = {c["ratio"]: c for c in analysed["per_ratio"]}
        meaningful = by_ratio[0.5]
        self.assertFalse(meaningful["collapsed"])
        # mincost - uniform per-step: 3.0 - 1.0 = 2.0
        self.assertAlmostEqual(
            meaningful["mincost_vs_uniform_per_step_delta"], 2.0, places=5
        )
        # worst - oracle per-step: 3.0 - 0.5 = 2.5
        self.assertAlmostEqual(
            meaningful["best_vs_worst_per_step_delta"], 2.5, places=5
        )
        # Collapsed ratio: every scenario lands on the single allocation.
        coll = by_ratio[0.1]
        self.assertTrue(coll["collapsed"])
        self.assertAlmostEqual(
            coll["mincost_vs_uniform_per_step_delta"], 0.0, places=5
        )

    def test_aggregate_excludes_collapsed_cells(self):
        analysed = analyze_prompt(
            self._prompt_with_two_ratios(),
            num_groups=4, n_sink=4, n_recent=4,
        )
        rows = aggregate([analysed])
        by_key = {(r["tier_length"], r["ratio"]): r for r in rows}
        meaningful_row = by_key[(200, 0.5)]
        collapsed_row = by_key[(200, 0.1)]
        self.assertEqual(meaningful_row["n_prompts_meaningful"], 1)
        self.assertEqual(meaningful_row["n_prompts_collapsed"], 0)
        self.assertAlmostEqual(
            meaningful_row["mean_mincost_vs_uniform_per_step"], 2.0, places=5
        )
        self.assertEqual(collapsed_row["n_prompts_meaningful"], 0)
        self.assertEqual(collapsed_row["n_prompts_collapsed"], 1)
        # Aggregate mean is None when no meaningful prompts contribute.
        self.assertIsNone(collapsed_row["mean_mincost_vs_uniform_per_step"])


def _saturation_prompt(*, with_support: bool = True, prompt_id: str = "p0") -> dict:
    """A 10000-token prompt, ratio 0.08 (total 800, uniform [200]*4).

    per-step KL means: uniform 0.10, A 0.04 (oracle), B 0.30 (worst).
      oracle_gap = uniform - oracle = 0.06
      worst_gap  = worst  - oracle = 0.26
    With effective support S_g=[50,250,60,55], S_g_max=[55,280,65,60] and
    B_g=[200]*4: min_margin_mean = -50, min_margin_max = -80.
    """
    prompt = {
        "prompt_id": prompt_id, "task": "cwe", "tier_length": 8192,
        "tokens": 10000, "status": "ok",
        "trajectories": [
            {"ratio": 0.08, "total": 800, "allocation": [200, 200, 200, 200],
             "cost": 0.30, "per_step_kl": [0.0, 0.10, 0.10, 0.10]},
            {"ratio": 0.08, "total": 800, "allocation": [596, 68, 68, 68],
             "cost": 0.50, "per_step_kl": [0.0, 0.04, 0.04, 0.04]},
            {"ratio": 0.08, "total": 800, "allocation": [68, 68, 68, 596],
             "cost": 0.45, "per_step_kl": [0.0, 0.30, 0.30, 0.30]},
        ],
        "ratio_summaries": [
            {"ratio": 0.08, "total": 800, "discretionary": 528,
             "n_distinct_allocations": 3, "collapsed": False},
        ],
    }
    if with_support:
        prompt["effective_support"] = {
            "per_group": [50.0, 250.0, 60.0, 55.0],
            "per_group_max": [55.0, 280.0, 65.0, 60.0],
            "estimator": "mass_coverage", "tau": 0.95, "source": "obs_window",
        }
    return prompt


class TestSaturationMetrics(unittest.TestCase):
    def test_oracle_gap_is_uniform_minus_oracle(self):
        cell = analyze_prompt(_saturation_prompt(), 4, 4, 64)["per_ratio"][0]
        self.assertAlmostEqual(cell["oracle_gap_per_step"], 0.06, places=6)

    def test_worst_gap_is_worst_minus_oracle(self):
        cell = analyze_prompt(_saturation_prompt(), 4, 4, 64)["per_ratio"][0]
        self.assertAlmostEqual(cell["worst_gap_per_step"], 0.26, places=6)

    def test_min_margin_mean_uses_per_group_support(self):
        # B_g=[200]*4, S_g=[50,250,60,55] -> margins [150,-50,140,145].
        cell = analyze_prompt(_saturation_prompt(), 4, 4, 64)["per_ratio"][0]
        self.assertAlmostEqual(cell["min_margin_mean"], -50.0, places=6)

    def test_min_margin_max_uses_bottleneck_support(self):
        # S_g_max=[55,280,65,60] -> margins [145,-80,135,140].
        cell = analyze_prompt(_saturation_prompt(), 4, 4, 64)["per_ratio"][0]
        self.assertAlmostEqual(cell["min_margin_max"], -80.0, places=6)

    def test_margins_none_without_effective_support(self):
        cell = analyze_prompt(
            _saturation_prompt(with_support=False), 4, 4, 64
        )["per_ratio"][0]
        self.assertIsNone(cell["min_margin_mean"])
        self.assertIsNone(cell["min_margin_max"])
        # oracle_gap needs no support data and is still computed.
        self.assertAlmostEqual(cell["oracle_gap_per_step"], 0.06, places=6)


class TestAggregateSaturation(unittest.TestCase):
    def test_means_oracle_gap_and_margins_across_prompts(self):
        rows = aggregate([
            analyze_prompt(_saturation_prompt(prompt_id="p0"), 4, 4, 64),
            analyze_prompt(_saturation_prompt(prompt_id="p1"), 4, 4, 64),
        ])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["n_prompts_meaningful"], 2)
        self.assertAlmostEqual(row["mean_oracle_gap_per_step"], 0.06, places=6)
        self.assertAlmostEqual(row["mean_min_margin_mean"], -50.0, places=6)
        self.assertAlmostEqual(row["mean_min_margin_max"], -80.0, places=6)

    def test_margins_none_in_aggregate_without_support(self):
        rows = aggregate([
            analyze_prompt(_saturation_prompt(with_support=False), 4, 4, 64),
        ])
        self.assertIsNone(rows[0]["mean_min_margin_mean"])
        # oracle_gap still aggregates.
        self.assertAlmostEqual(rows[0]["mean_oracle_gap_per_step"], 0.06, places=6)


class TestEndToEndOnSyntheticDir(unittest.TestCase):
    def test_writes_stats_and_report(self):
        prompt = TestAnalyzePromptAndAggregate()._prompt_with_two_ratios()
        prompt["status"] = "ok"
        trajectories = {
            "manifest": "synthetic", "model": "synthetic",
            "k_dec": 3, "obs_window": 8,
            "num_groups": 4, "n_sink": 4, "n_recent": 4,
            "ratios": [0.5, 0.1],
            "prompts_attempted": 1, "prompts_ok": 1,
            "prompts": [prompt],
        }
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp)
            (in_dir / "trajectories.json").write_text(
                json.dumps(trajectories), encoding="utf-8"
            )
            rc = main(["--in-dir", str(in_dir)])
        self.assertEqual(rc, 0)


class TestPickSupport(unittest.TestCase):
    """_pick_support selects the effective-support entry for the saturation
    predictor — the producer primary, or a grid entry on an override."""

    def _prompt(self):
        return {
            "effective_support": {
                "estimator": "mass_coverage", "tau": 0.95,
                "per_group": [100.0], "per_group_max": [200.0],
            },
            "effective_support_grid": [
                {"estimator": "mass_coverage", "tau": 0.7,
                 "per_group": [30.0], "per_group_max": [60.0]},
                {"estimator": "mass_coverage", "tau": 0.95,
                 "per_group": [100.0], "per_group_max": [200.0]},
                {"estimator": "participation_ratio", "tau": None,
                 "per_group": [12.0], "per_group_max": [25.0]},
                {"estimator": "entropy", "tau": None,
                 "per_group": [18.0], "per_group_max": [40.0]},
            ],
        }

    def test_none_estimator_returns_producer_primary(self):
        self.assertEqual(_pick_support(self._prompt(), None, None)["per_group"], [100.0])

    def test_none_estimator_on_run_without_support_is_none(self):
        self.assertIsNone(_pick_support({}, None, None))

    def test_picks_tau_free_estimator_from_grid(self):
        picked = _pick_support(self._prompt(), "participation_ratio", None)
        self.assertEqual(picked["per_group"], [12.0])

    def test_picks_mass_coverage_at_the_requested_tau(self):
        picked = _pick_support(self._prompt(), "mass_coverage", 0.7)
        self.assertEqual(picked["tau"], 0.7)
        self.assertEqual(picked["per_group"], [30.0])

    def test_missing_grid_entry_returns_none(self):
        # tau 0.99 was never pre-computed by the producer grid.
        self.assertIsNone(_pick_support(self._prompt(), "mass_coverage", 0.99))

    def test_estimator_override_on_pre_grid_run_returns_none(self):
        self.assertIsNone(
            _pick_support({"effective_support": {"per_group": [1.0]}}, "entropy", None)
        )


class TestAggregateQuantiles(unittest.TestCase):
    """aggregate() reports oracle_gap median and max, not just the mean —
    the within-task replication runs need the distribution shape (a heavy
    tail shows up as mean >> median)."""

    @staticmethod
    def _cell(ratio, oracle_gap):
        return {
            "ratio": ratio, "collapsed": False,
            "mincost_vs_uniform_per_step_delta": 0.0,
            "best_vs_worst_per_step_delta": 0.0,
            "oracle_gap_per_step": oracle_gap,
            "worst_gap_per_step": 0.0,
            "uniform": {"kl_per_step": 0.0},
            "mincost_bad": {"kl_per_step": 0.0},
            "min_margin_mean": None, "min_margin_max": None,
        }

    def test_reports_median_and_max_oracle_gap(self):
        gaps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.3]
        per_prompt = [
            {"tier_length": 32768, "per_ratio": [self._cell(0.08, g)]}
            for g in gaps
        ]
        rows = aggregate(per_prompt)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["mean_oracle_gap_per_step"], sum(gaps) / len(gaps))
        self.assertAlmostEqual(row["median_oracle_gap_per_step"], 0.0)
        self.assertAlmostEqual(row["max_oracle_gap_per_step"], 1.3)


class TestBootstrapMedianCI(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(bootstrap_median_ci([]))

    def test_constant_input_zero_width(self):
        # All draws are the same value → CI is degenerate at that value.
        lo, hi = bootstrap_median_ci([0.5] * 10, n_resamples=200, seed=1)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_ci_contains_true_median_for_normal_input(self):
        # n=50 draws from a roughly-normal-around-1 distribution.
        xs = [
            1.05, 0.95, 1.10, 0.90, 1.02, 0.98, 1.08, 0.92, 1.03, 0.97,
            1.06, 0.94, 1.07, 0.93, 1.04, 0.96, 1.09, 0.91, 1.01, 0.99,
        ] * 3
        lo, hi = bootstrap_median_ci(xs, level=0.95, n_resamples=2000, seed=0)
        self.assertLessEqual(lo, 1.0)
        self.assertGreaterEqual(hi, 1.0)
        self.assertLess(hi - lo, 0.3)

    def test_lo_le_hi(self):
        lo, hi = bootstrap_median_ci(
            [0.0, 0.1, 0.2, 0.3, 0.5, 1.5, 2.0], n_resamples=500, seed=7
        )
        self.assertLessEqual(lo, hi)

    def test_seed_determinism(self):
        a = bootstrap_median_ci([0.0, 1.0, 2.0, 3.0], n_resamples=200, seed=42)
        b = bootstrap_median_ci([0.0, 1.0, 2.0, 3.0], n_resamples=200, seed=42)
        self.assertEqual(a, b)


class TestTailRate(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(tail_rate([], threshold=0.5))

    def test_strictly_greater_than(self):
        self.assertEqual(tail_rate([0.5, 0.5, 0.5], threshold=0.5), 0.0)

    def test_all_above(self):
        self.assertEqual(tail_rate([1.0, 1.5, 2.0], threshold=0.5), 1.0)

    def test_partial(self):
        self.assertAlmostEqual(
            tail_rate([0.0, 0.0, 0.5, 1.0, 1.5], threshold=0.5), 0.4
        )


class TestHeavyTailIndex(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(heavy_tail_index([]))

    def test_zero_median_returns_none(self):
        # Median 0 (most prompts have no lever) — ratio is undefined.
        self.assertIsNone(heavy_tail_index([0.0, 0.0, 0.0, 0.7, 1.3]))

    def test_uniform_cell_index_one(self):
        # max == median → index = 1.
        self.assertAlmostEqual(heavy_tail_index([0.5, 0.5, 0.5]), 1.0)

    def test_heavy_tail(self):
        # median 0.5, max 5.0 → index 10.
        self.assertAlmostEqual(
            heavy_tail_index([0.5, 0.4, 0.5, 0.6, 0.5, 5.0]), 10.0
        )


class TestClassifyAllocation(unittest.TestCase):
    """classify_allocation buckets a budget vector into shape categories.

    The classifier underpins concentration / spread indices: dominant
    means one group has nearly all discretionary budget, starved means
    one group sits at the floor while others share."""

    def test_uniform(self):
        self.assertEqual(classify_allocation([80, 80, 80, 80], floor=64), "uniform")

    def test_uniform_with_rounding_wobble(self):
        # largest-remainder rounding can shift one entry by ±1
        self.assertEqual(classify_allocation([80, 80, 81, 80], floor=64), "uniform")

    def test_dominant_g0(self):
        self.assertEqual(classify_allocation([400, 68, 68, 68], floor=64), "dominant_g0")

    def test_dominant_g2(self):
        self.assertEqual(classify_allocation([68, 68, 400, 68], floor=64), "dominant_g2")

    def test_starved_g1(self):
        # group 1 at floor, others share the rest evenly
        self.assertEqual(classify_allocation([100, 68, 100, 100], floor=64), "starved_g1")

    def test_other(self):
        # No clear pattern
        self.assertEqual(classify_allocation([100, 150, 200, 50], floor=64), "other")

    def test_empty(self):
        self.assertEqual(classify_allocation([], floor=64), "other")


class TestConcSpreadIndicesInPipeline(unittest.TestCase):
    """analyze_prompt populates conc_idx_per_step and spread_idx_per_step."""

    @staticmethod
    def _prompt_with_alloc_set():
        # 4 groups, total=320, floor=68 (sink+recent=4+64)
        # uniform = [80, 80, 80, 80]; dominant_g0 = [116, 68, 68, 68];
        # starved_g0 = [68, 84, 84, 84]
        return {
            "prompt_id": "p0",
            "task": "vt",
            "tier_length": 32768,
            "tokens": 1000,
            "trajectories": [
                {"ratio": 0.08, "total": 320,
                 "allocation": [80, 80, 80, 80], "cost": 0.1,
                 "per_step_kl": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
                {"ratio": 0.08, "total": 320,
                 "allocation": [116, 68, 68, 68], "cost": 0.1,
                 "per_step_kl": [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
                {"ratio": 0.08, "total": 320,
                 "allocation": [68, 84, 84, 84], "cost": 0.1,
                 "per_step_kl": [0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]},
            ],
            "ratio_summaries": [
                {"ratio": 0.08, "total": 320, "discretionary": 48,
                 "collapsed": False, "n_distinct_allocations": 3}
            ],
        }

    def test_conc_idx_positive_when_dominant_beats_uniform(self):
        prompt = self._prompt_with_alloc_set()
        result = analyze_prompt(prompt, num_groups=4, n_sink=4, n_recent=64)
        entry = result["per_ratio"][0]
        # uniform kl 0.5; dominant_g0 kl 0.1 → conc_idx = (0.5-0.1)/0.5 = 0.8
        self.assertAlmostEqual(entry["conc_idx_per_step"], 0.8, places=3)
        # starved_g0 kl 0.4 → spread_idx = (0.5-0.4)/0.5 = 0.2
        self.assertAlmostEqual(entry["spread_idx_per_step"], 0.2, places=3)

    def test_aggregate_surfaces_median_conc_idx(self):
        prompts = [self._prompt_with_alloc_set() for _ in range(3)]
        for i, p in enumerate(prompts):
            p["prompt_id"] = f"p{i}"
        per_prompt = [
            analyze_prompt(p, num_groups=4, n_sink=4, n_recent=64) for p in prompts
        ]
        rows = aggregate(per_prompt)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["conc_idx_median"], 0.8, places=3)
        self.assertAlmostEqual(rows[0]["spread_idx_median"], 0.2, places=3)

    def test_zero_uniform_kl_returns_none(self):
        # Saturated cell: uniform kl is ~0; conc_idx undefined
        prompt = self._prompt_with_alloc_set()
        # zero out uniform's per_step_kl
        for t in prompt["trajectories"]:
            if t["allocation"] == [80, 80, 80, 80]:
                t["per_step_kl"] = [0.0] * 8
        result = analyze_prompt(prompt, num_groups=4, n_sink=4, n_recent=64)
        entry = result["per_ratio"][0]
        self.assertIsNone(entry["conc_idx_per_step"])
        self.assertIsNone(entry["spread_idx_per_step"])


class TestNamedAllocatorsInPipeline(unittest.TestCase):
    """End-to-end: an `allocator_tag` on a trajectory surfaces as a named
    entry in `analyze_prompt`'s per_ratio cell, and `aggregate` computes
    `<tag>_kl` and `<tag>_vs_uniform` cell-level means."""

    @staticmethod
    def _prompt_with_pyramid(uniform_kl: float, pyramid_kl: float):
        return {
            "prompt_id": "p0",
            "task": "vt",
            "tier_length": 32768,
            "tokens": 1000,
            "trajectories": [
                {
                    "ratio": 0.08,
                    "total": 320,
                    "allocation": [80, 80, 80, 80],
                    "cost": 0.1,
                    "per_step_kl": [0.0] + [uniform_kl] * 7,
                },
                {
                    "ratio": 0.08,
                    "total": 320,
                    "allocation": [96, 88, 72, 64],
                    "cost": 0.15,
                    "per_step_kl": [0.0] + [pyramid_kl] * 7,
                    "allocator_tag": "pyramid",
                },
            ],
            "ratio_summaries": [
                {"ratio": 0.08, "total": 320, "discretionary": 48,
                 "collapsed": False, "n_distinct_allocations": 2},
            ],
        }

    def test_named_allocator_in_per_ratio_cell(self):
        prompt = self._prompt_with_pyramid(uniform_kl=0.5, pyramid_kl=0.3)
        result = analyze_prompt(
            prompt, num_groups=4, n_sink=4, n_recent=64,
        )
        named = result["per_ratio"][0]["named_allocators"]
        self.assertIn("pyramid", named)
        self.assertAlmostEqual(named["pyramid"]["kl_per_step"], 0.3)

    def test_aggregate_surfaces_named_allocator_means(self):
        prompts = [
            self._prompt_with_pyramid(uniform_kl=0.5, pyramid_kl=0.3),
            self._prompt_with_pyramid(uniform_kl=0.6, pyramid_kl=0.45),
        ]
        for i, p in enumerate(prompts):
            p["prompt_id"] = f"p{i}"
        per_prompt = [
            analyze_prompt(p, num_groups=4, n_sink=4, n_recent=64)
            for p in prompts
        ]
        rows = aggregate(per_prompt)
        named = rows[0]["named_allocator_means"]
        self.assertIn("pyramid_kl", named)
        self.assertAlmostEqual(named["pyramid_kl"], 0.375)
        self.assertIn("pyramid_vs_uniform", named)
        # (0.3-0.5 + 0.45-0.6)/2 = (-0.2 + -0.15)/2 = -0.175
        self.assertAlmostEqual(named["pyramid_vs_uniform"], -0.175)


class TestAggregateExtendedColumns(unittest.TestCase):
    """aggregate() emits the new bootstrap-CI / tail-rate columns."""

    def test_new_columns_present(self):
        # Use the cell shape from TestAggregateQuantiles inline.
        cell = lambda r, g: {
            "ratio": r, "collapsed": False,
            "mincost_vs_uniform_per_step_delta": 0.0,
            "best_vs_worst_per_step_delta": 0.0,
            "oracle_gap_per_step": g,
            "worst_gap_per_step": 0.0,
            "uniform": {"kl_per_step": 0.0},
            "mincost_bad": {"kl_per_step": 0.0},
            "min_margin_mean": None, "min_margin_max": None,
        }
        gaps = [0.0, 0.0, 0.5, 0.5, 1.0, 1.5]
        per_prompt = [
            {"tier_length": 32768, "per_ratio": [cell(0.08, g)]} for g in gaps
        ]
        rows = aggregate(per_prompt)
        row = rows[0]
        for col in (
            "oracle_gap_median_ci95_lo",
            "oracle_gap_median_ci95_hi",
            "oracle_gap_heavy_tail_index",
            "oracle_gap_tail_rate_0p10",
            "oracle_gap_tail_rate_0p50",
            "oracle_gap_tail_rate_1p00",
        ):
            self.assertIn(col, row)
            self.assertIsNotNone(row[col])
        # Strictly-greater: gaps=[0,0,0.5,0.5,1.0,1.5] → tail@1.0 = 1/6 (1.5),
        # tail@0.5 = 2/6 (1.0 & 1.5), tail@0.1 = 4/6 (0.5,0.5,1.0,1.5).
        self.assertAlmostEqual(row["oracle_gap_tail_rate_1p00"], 1 / 6)
        self.assertAlmostEqual(row["oracle_gap_tail_rate_0p50"], 2 / 6)
        self.assertAlmostEqual(row["oracle_gap_tail_rate_0p10"], 4 / 6)


if __name__ == "__main__":
    unittest.main()
