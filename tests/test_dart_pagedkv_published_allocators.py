"""Tests for the published-method allocator vectors."""

from __future__ import annotations

import unittest

from src.dart_pagedkv.published_allocators import (
    PUBLISHED_ALLOCATORS,
    adakv_priority_weights,
    inverse_pyramid_alloc,
    proportional_alloc,
    pyramid_alloc,
)


class TestPyramidAlloc(unittest.TestCase):
    def test_alpha_zero_equals_uniform(self):
        # alpha=0 is the uniform baseline.
        budget = pyramid_alloc(4, 400, [0, 0, 0, 0], alpha=0.0)
        self.assertEqual(budget, [100, 100, 100, 100])

    def test_monotone_decreasing_default(self):
        # Default alpha=0.2 → 1.5:1 ratio across groups.
        budget = pyramid_alloc(4, 400, [0, 0, 0, 0], alpha=0.2)
        self.assertEqual(sum(budget), 400)
        # b[0] > b[3], and the gradient is non-positive.
        self.assertGreater(budget[0], budget[3])
        for i in range(len(budget) - 1):
            self.assertGreaterEqual(budget[i], budget[i + 1])

    def test_ratio_at_alpha_0_2(self):
        # Floor-free 4-group, total 400 → mean_disc 100.
        # b[0] ~ 100 * 1.2 = 120, b[3] ~ 100 * 0.8 = 80 → ratio 1.5:1.
        budget = pyramid_alloc(4, 400, [0, 0, 0, 0], alpha=0.2)
        self.assertEqual(budget[0], 120)
        self.assertEqual(budget[3], 80)

    def test_respects_floors(self):
        budget = pyramid_alloc(4, 200, [30, 30, 30, 30], alpha=0.2)
        self.assertEqual(sum(budget), 200)
        for b in budget:
            self.assertGreaterEqual(b, 30)

    def test_alpha_one_rejected(self):
        # alpha=1 would zero out the last group's discretionary — rejected.
        with self.assertRaises(ValueError):
            pyramid_alloc(4, 400, [0, 0, 0, 0], alpha=1.0)

    def test_alpha_negative_rejected(self):
        with self.assertRaises(ValueError):
            pyramid_alloc(4, 400, [0, 0, 0, 0], alpha=-0.1)

    def test_single_group(self):
        self.assertEqual(pyramid_alloc(1, 100, [10]), [100])

    def test_total_below_floors_rejected(self):
        with self.assertRaises(ValueError):
            pyramid_alloc(4, 100, [30, 30, 30, 30])

    def test_deterministic(self):
        a = pyramid_alloc(4, 1000, [50, 50, 50, 50], alpha=0.3)
        b = pyramid_alloc(4, 1000, [50, 50, 50, 50], alpha=0.3)
        self.assertEqual(a, b)


class TestInversePyramidAlloc(unittest.TestCase):
    def test_alpha_zero_equals_uniform(self):
        self.assertEqual(
            inverse_pyramid_alloc(4, 400, [0, 0, 0, 0], alpha=0.0),
            [100, 100, 100, 100],
        )

    def test_monotone_increasing_default(self):
        budget = inverse_pyramid_alloc(4, 400, [0, 0, 0, 0], alpha=0.2)
        self.assertEqual(sum(budget), 400)
        self.assertGreater(budget[3], budget[0])
        for i in range(len(budget) - 1):
            self.assertLessEqual(budget[i], budget[i + 1])

    def test_mirror_of_pyramid(self):
        # inverse_pyramid is pyramid reversed (same total/floors/alpha).
        floors = [10, 10, 10, 10]
        p = pyramid_alloc(4, 200, floors, alpha=0.4)
        ip = inverse_pyramid_alloc(4, 200, floors, alpha=0.4)
        self.assertEqual(list(reversed(p)), ip)

    def test_respects_floors(self):
        budget = inverse_pyramid_alloc(4, 200, [30, 30, 30, 30], alpha=0.2)
        for b in budget:
            self.assertGreaterEqual(b, 30)
        self.assertEqual(sum(budget), 200)


class TestProportionalAlloc(unittest.TestCase):
    def test_equal_weights_equals_uniform(self):
        self.assertEqual(
            proportional_alloc(4, 400, [0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0]),
            [100, 100, 100, 100],
        )

    def test_proportional_to_weights(self):
        # discretionary 400, weights [4,3,2,1] → shares 160/120/80/40.
        budget = proportional_alloc(
            4, 400, [0, 0, 0, 0], [4.0, 3.0, 2.0, 1.0]
        )
        self.assertEqual(budget, [160, 120, 80, 40])

    def test_zero_weight_group_keeps_floor(self):
        # group 2 has weight 0: receives only its floor. Total 400, floor_sum
        # 40, discretionary 360, split 1:1:1 among groups 0/1/3 → 120 each.
        budget = proportional_alloc(
            4, 400, [10, 10, 10, 10], [1.0, 1.0, 0.0, 1.0]
        )
        self.assertEqual(sum(budget), 400)
        self.assertEqual(budget[2], 10)
        self.assertEqual(budget[0], 130)
        self.assertEqual(budget[1], 130)
        self.assertEqual(budget[3], 130)

    def test_all_zero_weights_rejected(self):
        with self.assertRaises(ValueError):
            proportional_alloc(4, 400, [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0])

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            proportional_alloc(4, 400, [0, 0, 0, 0], [1.0, -0.5, 1.0, 1.0])

    def test_weights_length_mismatch(self):
        with self.assertRaises(ValueError):
            proportional_alloc(4, 400, [0, 0, 0, 0], [1.0, 1.0, 1.0])

    def test_respects_floors(self):
        budget = proportional_alloc(
            4, 200, [30, 30, 30, 30], [3.0, 1.0, 1.0, 1.0]
        )
        for b in budget:
            self.assertGreaterEqual(b, 30)
        self.assertEqual(sum(budget), 200)


class TestAdakvPriorityWeights(unittest.TestCase):
    def test_uniform_priorities_yield_equal_weights(self):
        # All groups have flat priority → equal top-K mass → equal weights.
        priorities = [[1.0] * 100 for _ in range(4)]
        weights = adakv_priority_weights(priorities, 400, [0, 0, 0, 0])
        self.assertEqual(weights, [100.0, 100.0, 100.0, 100.0])

    def test_spikier_group_gets_more_weight(self):
        # group 0: one big value; others: flat.
        priorities = [
            [1000.0] + [0.0] * 99,
            [1.0] * 100,
            [1.0] * 100,
            [1.0] * 100,
        ]
        weights = adakv_priority_weights(priorities, 400, [0, 0, 0, 0])
        self.assertGreater(weights[0], weights[1])

    def test_floors_count_against_budget(self):
        # Per-group budget = floor + discretionary//G.
        # priorities length 50, per_group_budget = 10 + 200//4 = 60 → topk=50.
        priorities = [[1.0] * 50 for _ in range(4)]
        weights = adakv_priority_weights(priorities, 200, [10, 10, 10, 10])
        self.assertEqual(weights, [50.0, 50.0, 50.0, 50.0])

    def test_pipes_into_proportional_alloc(self):
        # End-to-end AdaKV: priority weights → proportional_alloc.
        priorities = [
            [10.0] * 50, [5.0] * 50, [2.5] * 50, [1.25] * 50,
        ]
        weights = adakv_priority_weights(priorities, 400, [0, 0, 0, 0])
        budget = proportional_alloc(4, 400, [0, 0, 0, 0], weights)
        # group 0 should dominate.
        self.assertEqual(sum(budget), 400)
        self.assertGreater(budget[0], budget[1])
        self.assertGreater(budget[1], budget[2])
        self.assertGreater(budget[2], budget[3])

    def test_empty_groups_rejected(self):
        with self.assertRaises(ValueError):
            adakv_priority_weights([], 400, [])


class TestRegistry(unittest.TestCase):
    def test_registry_contents(self):
        self.assertIn("pyramid", PUBLISHED_ALLOCATORS)
        self.assertIn("inverse_pyramid", PUBLISHED_ALLOCATORS)

    def test_registry_callables(self):
        # Sanity: registered functions match by reference.
        self.assertIs(PUBLISHED_ALLOCATORS["pyramid"], pyramid_alloc)
        self.assertIs(
            PUBLISHED_ALLOCATORS["inverse_pyramid"], inverse_pyramid_alloc
        )


if __name__ == "__main__":
    unittest.main()
