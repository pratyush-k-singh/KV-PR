"""Tests for src/dart_pagedkv/windowed_priority.py.

The decision gate (spec 2026-05-17-longcontext-decode-probe §8.1): the
windowed pooled priority — the vector that feeds hot-set top-k
selection — must equal the full-matrix `snapkv_priority` within fp
tolerance. The full O(T²) `recompute_attention_matrix` is used here as
the reference at a small T where it still fits; that is exactly the
sub-8k cross-check the spec calls for. All CPU, synthetic Q/K.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.dart_pagedkv.priorities import snapkv_priority
from src.dart_pagedkv.windowed_priority import (
    accumulated_attention_priority,
    k_norm_priority,
    random_priority,
    recent_priority,
    windowed_attention_per_head,
    windowed_recompute_attention,
    windowed_snapkv_priority,
)
from src.trace.collect import LayerQKCapture, recompute_attention_matrix


def _synthetic_captures(
    n_layers: int, n_heads: int, n_kv_heads: int,
    seq_len: int, head_dim: int, *, seed: int = 0,
) -> dict[int, LayerQKCapture]:
    """Random post-RoPE-shaped Q/K captures, GQA-consistent."""
    gen = torch.Generator().manual_seed(seed)
    kv_groups = n_heads // n_kv_heads
    captures: dict[int, LayerQKCapture] = {}
    for layer in range(n_layers):
        captures[layer] = LayerQKCapture(
            query=torch.randn(1, n_heads, seq_len, head_dim, generator=gen),
            key=torch.randn(1, n_kv_heads, seq_len, head_dim, generator=gen),
            scaling=head_dim ** -0.5,
            num_key_value_groups=kv_groups,
        )
    return captures


class TestWindowedMatchesFullMatrix(unittest.TestCase):
    """The §8.1 decision gate."""

    def test_windowed_priority_equals_snapkv_priority(self):
        # GQA case (8 q-heads, 2 kv-heads) — full [T,T] still fits at T=64.
        caps = _synthetic_captures(
            n_layers=3, n_heads=8, n_kv_heads=2, seq_len=64, head_dim=16
        )
        obs_window = 16
        full = recompute_attention_matrix(caps, device="cpu", heads_per_batch=4)
        full_priority = snapkv_priority(full, obs_window=obs_window)

        windowed = windowed_recompute_attention(
            caps, obs_window=obs_window, device="cpu", heads_per_batch=4
        )
        windowed_priority = windowed_snapkv_priority(windowed)

        self.assertEqual(windowed.shape, (obs_window, 64))
        np.testing.assert_allclose(
            windowed_priority, full_priority, rtol=1e-5, atol=1e-5
        )

    def test_windowed_attention_equals_full_last_rows(self):
        # MHA case (no GQA): n_kv_heads == n_heads.
        caps = _synthetic_captures(
            n_layers=2, n_heads=4, n_kv_heads=4, seq_len=48, head_dim=8, seed=1
        )
        obs_window = 12
        full = recompute_attention_matrix(caps, device="cpu", heads_per_batch=8)
        windowed = windowed_recompute_attention(
            caps, obs_window=obs_window, device="cpu", heads_per_batch=8
        )
        np.testing.assert_allclose(
            windowed, full[-obs_window:, :], rtol=1e-5, atol=1e-5
        )

    def test_obs_window_at_or_above_seq_len_clamps_to_full(self):
        caps = _synthetic_captures(
            n_layers=1, n_heads=2, n_kv_heads=2, seq_len=10, head_dim=4, seed=2
        )
        windowed = windowed_recompute_attention(caps, obs_window=999, device="cpu")
        full = recompute_attention_matrix(caps, device="cpu")
        self.assertEqual(windowed.shape, (10, 10))
        np.testing.assert_allclose(windowed, full, rtol=1e-5, atol=1e-5)

    def test_rejects_empty_captures(self):
        with self.assertRaises(ValueError):
            windowed_recompute_attention({}, obs_window=8)

    def test_rejects_nonpositive_obs_window(self):
        caps = _synthetic_captures(1, 2, 2, 8, 4)
        with self.assertRaises(ValueError):
            windowed_recompute_attention(caps, obs_window=0)


class TestWindowedSnapkvPriority(unittest.TestCase):
    def test_sums_all_rows_by_default(self):
        block = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            windowed_snapkv_priority(block), np.array([4.0, 6.0], np.float32)
        )

    def test_obs_window_restricts_to_last_rows(self):
        block = np.array(
            [[1.0, 1.0], [10.0, 10.0], [100.0, 100.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(
            windowed_snapkv_priority(block, obs_window=2),
            np.array([110.0, 110.0], dtype=np.float32),
        )

    def test_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            windowed_snapkv_priority(np.zeros((2, 3, 4), dtype=np.float32))


class TestWindowedAttentionPerHead(unittest.TestCase):
    """Per-head windowed attention for the effective-support estimator."""

    def test_head_mean_equals_windowed_recompute(self):
        # GQA layer; the head-mean of the per-head block must equal the
        # head-meaned `windowed_recompute_attention` for that single layer.
        caps = _synthetic_captures(
            n_layers=1, n_heads=8, n_kv_heads=2, seq_len=40, head_dim=16, seed=3
        )
        obs_window = 12
        per_head = windowed_attention_per_head(
            caps[0], obs_window=obs_window, device="cpu", heads_per_batch=4
        )
        self.assertEqual(per_head.shape, (8, obs_window, 40))
        reference = windowed_recompute_attention(
            {0: caps[0]}, obs_window=obs_window, device="cpu", heads_per_batch=4
        )
        np.testing.assert_allclose(
            per_head.mean(axis=0), reference, rtol=1e-5, atol=1e-5
        )

    def test_rows_are_softmax_distributions(self):
        # Every [head, query-row] distribution sums to 1 over the keys.
        caps = _synthetic_captures(1, 4, 4, 32, 8, seed=4)
        per_head = windowed_attention_per_head(caps[0], obs_window=8, device="cpu")
        sums = per_head.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=0, atol=1e-5)

    def test_obs_window_clamps_to_seq_len(self):
        caps = _synthetic_captures(1, 2, 2, 10, 4, seed=5)
        per_head = windowed_attention_per_head(caps[0], obs_window=999, device="cpu")
        self.assertEqual(per_head.shape, (2, 10, 10))

    def test_rejects_nonpositive_obs_window(self):
        caps = _synthetic_captures(1, 2, 2, 8, 4)
        with self.assertRaises(ValueError):
            windowed_attention_per_head(caps[0], obs_window=0)


class TestRecentPriority(unittest.TestCase):
    """recent_priority is the priority-sweep control: a pure positional
    priority (no attention), top-k = the most-recent prompt keys."""

    def test_priority_is_ascending_by_position(self):
        self.assertEqual(recent_priority(5).tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_top_k_selects_the_most_recent_keys(self):
        p = recent_priority(10)
        top3 = sorted(range(10), key=lambda i: p[i])[-3:]
        self.assertEqual(sorted(top3), [7, 8, 9])


class TestRandomPriority(unittest.TestCase):
    """random_priority is the priority-sweep floor: a seeded random
    priority — deterministic per seed so the sweep is reproducible."""

    def test_deterministic_for_a_given_seed(self):
        self.assertEqual(
            random_priority(20, seed=7).tolist(),
            random_priority(20, seed=7).tolist(),
        )

    def test_different_seeds_give_different_priorities(self):
        self.assertNotEqual(
            random_priority(20, seed=7).tolist(),
            random_priority(20, seed=8).tolist(),
        )

    def test_length_matches_sequence(self):
        self.assertEqual(len(random_priority(33, seed=1)), 33)


class TestKNormPriority(unittest.TestCase):
    """k_norm_priority — per-token mean ||K[i]||_2 across (layers, heads).

    The cheapest Q/K-derived priority: K-only, no attention compute.
    """

    def test_matches_manual_norm_on_synthetic_caps(self):
        caps = _synthetic_captures(
            n_layers=2, n_heads=4, n_kv_heads=2, seq_len=16, head_dim=8, seed=3
        )
        got = k_norm_priority(caps)

        # Brute-force reference: mean over layers of (mean over kv-heads of ||K_i||_2).
        expected = np.zeros(16, dtype=np.float32)
        for capture in caps.values():
            norms = torch.linalg.vector_norm(capture.key, dim=-1).squeeze(0)
            expected += norms.mean(dim=0).numpy()
        expected /= len(caps)

        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_length_equals_seq_len(self):
        caps = _synthetic_captures(
            n_layers=1, n_heads=2, n_kv_heads=2, seq_len=11, head_dim=4
        )
        self.assertEqual(k_norm_priority(caps).shape, (11,))

    def test_empty_captures_rejected(self):
        with self.assertRaises(ValueError):
            k_norm_priority({})

    def test_no_q_dependency(self):
        # k_norm_priority is K-only: zeroing Q must NOT change the result.
        caps1 = _synthetic_captures(
            n_layers=2, n_heads=4, n_kv_heads=2, seq_len=12, head_dim=4, seed=11
        )
        caps2 = {
            i: LayerQKCapture(
                query=torch.zeros_like(c.query),
                key=c.key.clone(),
                scaling=c.scaling,
                num_key_value_groups=c.num_key_value_groups,
            )
            for i, c in caps1.items()
        }
        np.testing.assert_allclose(
            k_norm_priority(caps1), k_norm_priority(caps2), rtol=1e-6, atol=1e-7
        )


class TestAccumulatedAttentionPriority(unittest.TestCase):
    """accumulated_attention_priority — chunked H2O over the full prefill.

    Sums column attention across **all** Q rows (not just the last w).
    The chunked computation must agree with the brute-force [T,T]
    reference (windowed_recompute_attention with obs_window=T).
    """

    def test_equals_windowed_with_window_eq_seq_len(self):
        caps = _synthetic_captures(
            n_layers=2, n_heads=4, n_kv_heads=2, seq_len=24, head_dim=8, seed=5
        )
        # Reference: full attention block, then column sum.
        full = windowed_recompute_attention(caps, obs_window=24, device="cpu", heads_per_batch=2)
        reference = full.sum(axis=0)

        # Chunked accumulation should match.
        got = accumulated_attention_priority(
            caps, chunk_size=8, device="cpu", heads_per_batch=2
        )
        np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-5)

    def test_chunk_size_invariance(self):
        # Different chunk_size values must give the same result (modulo fp).
        caps = _synthetic_captures(
            n_layers=1, n_heads=4, n_kv_heads=2, seq_len=20, head_dim=8, seed=9
        )
        a = accumulated_attention_priority(caps, chunk_size=4)
        b = accumulated_attention_priority(caps, chunk_size=10)
        c = accumulated_attention_priority(caps, chunk_size=20)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(a, c, rtol=1e-5, atol=1e-5)

    def test_priority_length(self):
        caps = _synthetic_captures(
            n_layers=1, n_heads=2, n_kv_heads=2, seq_len=15, head_dim=4
        )
        self.assertEqual(
            accumulated_attention_priority(caps, chunk_size=5).shape, (15,)
        )

    def test_empty_captures_rejected(self):
        with self.assertRaises(ValueError):
            accumulated_attention_priority({})

    def test_invalid_chunk_size_rejected(self):
        caps = _synthetic_captures(
            n_layers=1, n_heads=2, n_kv_heads=2, seq_len=8, head_dim=4
        )
        with self.assertRaises(ValueError):
            accumulated_attention_priority(caps, chunk_size=0)
        with self.assertRaises(ValueError):
            accumulated_attention_priority(caps, chunk_size=-2)

    def test_causal_mask_correctness(self):
        # Token 0 receives attention from every query row 0..T-1 (it's in
        # every row's causal window). Token T-1 receives attention only
        # from row T-1. So the accumulated priority should be monotone-
        # decreasing for a small, well-conditioned synthetic input where
        # softmax over the lengthening window is roughly uniform — token
        # 0's column sum ≈ sum_r(1/(r+1)) (harmonic), token T-1 ≈ 1/T.
        # For random Q/K the monotone direction holds as a weaker
        # tendency; the strong invariant is just non-negativity.
        caps = _synthetic_captures(
            n_layers=1, n_heads=1, n_kv_heads=1, seq_len=12, head_dim=8, seed=1
        )
        p = accumulated_attention_priority(caps, chunk_size=4)
        # all attention weights are positive → column sums are non-negative
        self.assertTrue((p >= -1e-6).all())
        # column sum total over T rows = T (each softmax row sums to 1)
        self.assertAlmostEqual(float(p.sum()), 12.0, places=4)


if __name__ == "__main__":
    unittest.main()
