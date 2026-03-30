"""Tests for src/dart_pagedkv/flex_decode.py.

Only the pure ``decode_step_params`` arithmetic is unit-tested here. The
``FlexCachedDecodeSession`` is HF-cache- and FlexAttention-coupled (HF
flex needs Triton + a modern GPU) and is validated server-side by
`experiments/scripts/flex_decode_check.py` — all-hot masked decode must
match the full-KV reference (KL ≈ 0), severe masked decode must not.
"""

from __future__ import annotations

import unittest

import torch._dynamo

from src.dart_pagedkv.flex_decode import (
    current_attn_implementation,
    decode_step_params,
    temporary_attn_implementation,
)


class _DummyConfig:
    def __init__(self, attn: str = "sdpa"):
        self._attn_implementation = attn


class _DummyModel:
    def __init__(self, attn: str = "sdpa"):
        self.config = _DummyConfig(attn)
        self.calls = []

    def set_attn_implementation(self, implementation):
        self.calls.append(implementation)
        self.config._attn_implementation = implementation


class TestDynamoRecompileLimit(unittest.TestCase):
    """Importing flex_decode must raise Dynamo's recompile cache.

    Cached incremental decode feeds flex_attention a key tensor that
    grows one position per step, so its compiled kernel is recompiled per
    shape. The default cache_size_limit of 8 is exhausted within one
    decode, after which Dynamo abandons the frame for slow eager
    execution. The module raises the limit at import so every shape stays
    compiled and is reused across budget trajectories.
    """

    def test_cache_size_limit_raised_on_import(self):
        # flex_decode is imported at module load (top of this file).
        self.assertGreaterEqual(torch._dynamo.config.cache_size_limit, 1024)
        self.assertGreaterEqual(
            torch._dynamo.config.accumulated_cache_size_limit, 1024
        )

    def test_compiles_static_shapes_not_symbolic(self):
        # Dynamic-shape flex kernels hit a Triton autotuning
        # "misaligned address" crash; the module pins static
        # (per-shape) compilation instead.
        self.assertFalse(torch._dynamo.config.automatic_dynamic_shapes)
        self.assertTrue(torch._dynamo.config.assume_static_by_default)


class TestDecodeStepParams(unittest.TestCase):
    def test_step_1_is_first_decoded_position(self):
        # prefill 100 tokens; step 1 forwards the token at absolute pos 100,
        # the cache then holds 101 keys.
        self.assertEqual(decode_step_params(100, 1), (101, 100))

    def test_step_grows_kv_and_q_start_by_one(self):
        self.assertEqual(decode_step_params(100, 5), (105, 104))
        self.assertEqual(decode_step_params(2048, 64), (2112, 2111))

    def test_kv_len_is_one_past_q_start(self):
        for L in (1, 50, 8192):
            for step in (1, 10, 63):
                kv_len, q_start = decode_step_params(L, step)
                self.assertEqual(kv_len, q_start + 1)


class TestTemporaryAttnImplementation(unittest.TestCase):
    def test_switches_and_restores(self):
        model = _DummyModel("sdpa")
        with temporary_attn_implementation(model, "flex_attention"):
            self.assertEqual(current_attn_implementation(model), "flex_attention")
        self.assertEqual(current_attn_implementation(model), "sdpa")
        self.assertEqual(model.calls, ["flex_attention", "sdpa"])

    def test_noop_when_already_selected(self):
        model = _DummyModel("flex_attention")
        with temporary_attn_implementation(model, "flex_attention"):
            self.assertEqual(current_attn_implementation(model), "flex_attention")
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
