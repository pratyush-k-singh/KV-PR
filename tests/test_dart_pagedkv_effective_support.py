"""Tests for effective-context support (spec 2026-05-17-effective-context-saturation §6).

Support is measured on full-KV attention distributions; the estimators have
closed-form values on one-hot and uniform distributions, which is what the
spec's "known-distribution check" calls for.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.dart_pagedkv.effective_support import (
    ESTIMATORS,
    GroupSupport,
    block_support,
    distribution_support,
    effective_support,
    group_support_from_layers,
)


def _uniform_over(k: int, t: int) -> np.ndarray:
    """A length-``t`` distribution: mass 1/k on the first ``k`` keys, 0 elsewhere."""
    p = np.zeros(t, dtype=np.float32)
    p[:k] = 1.0 / k
    return p


def _one_hot(t: int, idx: int = 0) -> np.ndarray:
    p = np.zeros(t, dtype=np.float32)
    p[idx] = 1.0
    return p


class TestDistributionSupportKnownValues(unittest.TestCase):
    def test_one_hot_support_is_one_for_every_estimator(self):
        p = _one_hot(50, idx=7)
        for estimator in ESTIMATORS:
            with self.subTest(estimator=estimator):
                s = distribution_support(p, estimator=estimator, tau=0.95)
                self.assertAlmostEqual(float(s), 1.0, places=5)

    def test_uniform_over_k_participation_ratio_equals_k(self):
        s = distribution_support(
            _uniform_over(10, 64), estimator="participation_ratio"
        )
        self.assertAlmostEqual(float(s), 10.0, places=4)

    def test_uniform_over_k_entropy_support_equals_k(self):
        s = distribution_support(_uniform_over(10, 64), estimator="entropy")
        self.assertAlmostEqual(float(s), 10.0, places=4)

    def test_uniform_over_k_mass_coverage_is_ceil_tau_k(self):
        # top-m mass = m/k; smallest m with m/k >= tau is ceil(tau*k).
        s = distribution_support(
            _uniform_over(10, 64), estimator="mass_coverage", tau=0.8
        )
        self.assertEqual(int(s), 8)  # ceil(0.8 * 10)

    def test_mass_coverage_on_skewed_distribution(self):
        # sorted desc cumsum = [0.5, 0.8, 1.0]
        p = np.array([0.2, 0.5, 0.3], dtype=np.float32)
        self.assertEqual(
            int(distribution_support(p, estimator="mass_coverage", tau=0.5)), 1
        )
        self.assertEqual(
            int(distribution_support(p, estimator="mass_coverage", tau=0.7)), 2
        )
        self.assertEqual(
            int(distribution_support(p, estimator="mass_coverage", tau=0.9)), 3
        )

    def test_participation_ratio_on_skewed_distribution(self):
        p = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        # 1 / sum(p^2) = 1 / 0.38
        s = distribution_support(p, estimator="participation_ratio")
        self.assertAlmostEqual(float(s), 1.0 / 0.38, places=4)

    def test_entropy_support_on_skewed_distribution(self):
        p = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        h = -(0.5 * math.log(0.5) + 0.3 * math.log(0.3) + 0.2 * math.log(0.2))
        s = distribution_support(p, estimator="entropy")
        self.assertAlmostEqual(float(s), math.exp(h), places=4)

    def test_unnormalised_input_is_normalised_first(self):
        # [5, 3, 2] must give the same support as [0.5, 0.3, 0.2].
        raw = np.array([5.0, 3.0, 2.0], dtype=np.float32)
        norm = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        for estimator in ESTIMATORS:
            with self.subTest(estimator=estimator):
                self.assertAlmostEqual(
                    float(distribution_support(raw, estimator=estimator, tau=0.7)),
                    float(distribution_support(norm, estimator=estimator, tau=0.7)),
                    places=4,
                )

    def test_vectorises_over_leading_axes(self):
        # A [H, w, T] block returns a [H, w] support array.
        block = np.stack(
            [
                np.stack([_one_hot(40), _uniform_over(4, 40)]),
                np.stack([_uniform_over(4, 40), _uniform_over(4, 40)]),
            ]
        )
        s = distribution_support(block, estimator="participation_ratio")
        self.assertEqual(s.shape, (2, 2))
        self.assertAlmostEqual(float(s[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(s[0, 1]), 4.0, places=4)


class TestDistributionSupportValidation(unittest.TestCase):
    def test_unknown_estimator_raises(self):
        with self.assertRaises(ValueError):
            distribution_support(_one_hot(10), estimator="median")

    def test_negative_entry_raises(self):
        p = np.array([0.6, -0.1, 0.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            distribution_support(p, estimator="entropy")

    def test_non_finite_entry_raises(self):
        p = np.array([0.6, np.nan, 0.4], dtype=np.float32)
        with self.assertRaises(ValueError):
            distribution_support(p, estimator="entropy")

    def test_zero_sum_row_raises(self):
        with self.assertRaises(ValueError):
            distribution_support(np.zeros(10, dtype=np.float32), estimator="entropy")


class TestBlockSupport(unittest.TestCase):
    def test_mean_and_max_over_rows(self):
        # row 0 support 1, row 1 support 4 (participation ratio).
        block = np.stack([_one_hot(40), _uniform_over(4, 40)])
        mean, mx = block_support(block, estimator="participation_ratio")
        self.assertAlmostEqual(mean, 2.5, places=4)
        self.assertAlmostEqual(mx, 4.0, places=4)

    def test_flattens_head_and_row_axes(self):
        # [H=2, w=2, T] — all four distributions averaged / maxed together.
        block = np.stack(
            [
                np.stack([_one_hot(40), _one_hot(40)]),
                np.stack([_uniform_over(4, 40), _uniform_over(4, 40)]),
            ]
        )
        mean, mx = block_support(block, estimator="participation_ratio")
        self.assertAlmostEqual(mean, 2.5, places=4)
        self.assertAlmostEqual(mx, 4.0, places=4)


class TestEffectiveSupport(unittest.TestCase):
    def _layers(self):
        # 4 layers, each a [2, T] block; participation-ratio supports:
        #   layer 0: rows 1, 1   -> mean 1.0,  max 1
        #   layer 1: rows 4, 4   -> mean 4.0,  max 4
        #   layer 2: rows 1, 4   -> mean 2.5,  max 4
        #   layer 3: rows 1, 1   -> mean 1.0,  max 1
        return {
            0: np.stack([_one_hot(40), _one_hot(40)]),
            1: np.stack([_uniform_over(4, 40), _uniform_over(4, 40)]),
            2: np.stack([_one_hot(40), _uniform_over(4, 40)]),
            3: np.stack([_one_hot(40), _one_hot(40)]),
        }

    def test_per_group_mean_averages_rows_then_layers(self):
        result = effective_support(
            self._layers(), [[0, 1], [2, 3]], estimator="participation_ratio"
        )
        self.assertIsInstance(result, GroupSupport)
        # group 0: mean(layer means 1.0, 4.0) = 2.5
        # group 1: mean(layer means 2.5, 1.0) = 1.75
        self.assertAlmostEqual(result.per_group_mean[0], 2.5, places=4)
        self.assertAlmostEqual(result.per_group_mean[1], 1.75, places=4)

    def test_per_group_max_is_bottleneck_over_rows_and_layers(self):
        result = effective_support(
            self._layers(), [[0, 1], [2, 3]], estimator="participation_ratio"
        )
        # group 0: max(layer maxes 1, 4) = 4 ; group 1: max(4, 1) = 4
        self.assertAlmostEqual(result.per_group_max[0], 4.0, places=4)
        self.assertAlmostEqual(result.per_group_max[1], 4.0, places=4)

    def test_records_estimator_and_tau(self):
        result = effective_support(
            self._layers(), [[0, 1], [2, 3]],
            estimator="mass_coverage", tau=0.9,
        )
        self.assertEqual(result.estimator, "mass_coverage")
        self.assertAlmostEqual(result.tau, 0.9, places=6)

    def test_single_group_covers_all_layers(self):
        result = effective_support(
            self._layers(), [[0, 1, 2, 3]], estimator="participation_ratio"
        )
        # mean of layer means 1.0, 4.0, 2.5, 1.0
        self.assertAlmostEqual(result.per_group_mean[0], 2.125, places=4)

    def test_empty_groups_raises(self):
        with self.assertRaises(ValueError):
            effective_support(self._layers(), [], estimator="entropy")

    def test_group_referencing_missing_layer_raises(self):
        with self.assertRaises(ValueError):
            effective_support(self._layers(), [[0, 1], [2, 9]], estimator="entropy")


class TestGroupSupportFromLayers(unittest.TestCase):
    """Streaming aggregation — for callers that compute support per layer."""

    def test_aggregates_layer_means_and_maxes(self):
        layer_mean = {0: 1.0, 1: 4.0, 2: 2.5, 3: 1.0}
        layer_max = {0: 1.0, 1: 4.0, 2: 4.0, 3: 1.0}
        gs = group_support_from_layers(
            layer_mean, layer_max, [[0, 1], [2, 3]],
            estimator="participation_ratio", tau=0.9,
        )
        self.assertIsInstance(gs, GroupSupport)
        # group 0: mean(1.0, 4.0)=2.5, max(1,4)=4 ; group 1: mean(2.5,1.0)=1.75
        self.assertAlmostEqual(gs.per_group_mean[0], 2.5, places=6)
        self.assertAlmostEqual(gs.per_group_mean[1], 1.75, places=6)
        self.assertAlmostEqual(gs.per_group_max[0], 4.0, places=6)
        self.assertAlmostEqual(gs.per_group_max[1], 4.0, places=6)
        self.assertEqual(gs.estimator, "participation_ratio")
        self.assertAlmostEqual(gs.tau, 0.9, places=6)

    def test_streaming_equals_all_in_one_effective_support(self):
        layers = {
            0: np.stack([_one_hot(40), _one_hot(40)]),
            1: np.stack([_uniform_over(4, 40), _uniform_over(4, 40)]),
            2: np.stack([_one_hot(40), _uniform_over(4, 40)]),
            3: np.stack([_one_hot(40), _one_hot(40)]),
        }
        groups = [[0, 1], [2, 3]]
        all_in_one = effective_support(layers, groups, estimator="entropy")
        layer_mean, layer_max = {}, {}
        for layer, block in layers.items():
            layer_mean[layer], layer_max[layer] = block_support(
                block, estimator="entropy"
            )
        streamed = group_support_from_layers(
            layer_mean, layer_max, groups, estimator="entropy"
        )
        self.assertEqual(streamed.per_group_mean, all_in_one.per_group_mean)
        self.assertEqual(streamed.per_group_max, all_in_one.per_group_max)

    def test_empty_groups_raises(self):
        with self.assertRaises(ValueError):
            group_support_from_layers({0: 1.0}, {0: 1.0}, [])

    def test_missing_layer_raises(self):
        with self.assertRaises(ValueError):
            group_support_from_layers({0: 1.0}, {0: 1.0}, [[0, 5]])


if __name__ == "__main__":
    unittest.main()
