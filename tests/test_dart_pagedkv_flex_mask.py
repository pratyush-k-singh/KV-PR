"""Tests for src/dart_pagedkv/flex_mask.py — FlexAttention hot/cold masking."""

from __future__ import annotations

import unittest

import torch

from torch.nn.attention.flex_attention import BlockMask, flex_attention

from src.dart_pagedkv.flex_mask import (
    build_group_block_masks,
    build_is_hot_vector,
    install_per_layer_block_mask_hooks,
    make_group_mask_mod,
)
from src.dart_pagedkv.logit_kl import build_group_attention_masks


class _FakeAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_mask = None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        self.received_mask = attention_mask
        return hidden_states


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttn()


class _FakeModel(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(_FakeLayer() for _ in range(n_layers))


class TestBuildIsHotVector(unittest.TestCase):
    def test_hot_prompt_keys_and_decoded_positions_are_true(self):
        # hot prompt keys {0,1,2}; key_boundary 4; position 4 is a decoded
        # token (>= key_boundary) and is always visible.
        is_hot = build_is_hot_vector([0, 1, 2], total_seq_len=5, key_boundary=4)
        self.assertEqual(is_hot.tolist(), [True, True, True, False, True])

    def test_decoded_tail_always_hot_even_if_not_listed(self):
        is_hot = build_is_hot_vector([0], total_seq_len=6, key_boundary=3)
        # prompt keys: 0 hot, 1/2 cold; decoded 3,4,5 always hot.
        self.assertEqual(is_hot.tolist(), [True, False, False, True, True, True])


class TestMaskMod(unittest.TestCase):
    def _allowed_grid(self, is_hot, total_seq_len):
        mask_mod = make_group_mask_mod(is_hot)
        q = torch.arange(total_seq_len).view(total_seq_len, 1)
        kv = torch.arange(total_seq_len).view(1, total_seq_len)
        return mask_mod(0, 0, q, kv)  # [T, T] bool

    def test_blocks_future_keys(self):
        is_hot = build_is_hot_vector([0, 1, 2, 3], total_seq_len=5, key_boundary=4)
        allowed = self._allowed_grid(is_hot, 5)
        self.assertFalse(bool(allowed[1, 2]))  # query 1 cannot see key 2
        self.assertTrue(bool(allowed[3, 1]))   # query 3 can see key 1

    def test_blocks_cold_prompt_keys(self):
        is_hot = build_is_hot_vector([0, 1], total_seq_len=5, key_boundary=4)
        allowed = self._allowed_grid(is_hot, 5)
        self.assertFalse(bool(allowed[4, 2]))  # key 2 cold
        self.assertFalse(bool(allowed[4, 3]))  # key 3 cold
        self.assertTrue(bool(allowed[4, 0]))   # key 0 hot

    def test_q_start_offsets_causal_boundary_for_incremental_decode(self):
        # Incremental-decode shape: a length-1 query. FlexAttention passes
        # q_idx=0, but the token's absolute position is q_start. Without
        # q_start the causal check kv<=q_idx would allow only key 0.
        is_hot = build_is_hot_vector([0, 1, 2, 3, 4, 5], total_seq_len=6, key_boundary=6)
        mask_mod = make_group_mask_mod(is_hot, q_start=4)
        q = torch.tensor([[0]])           # [1,1] — single query, q_idx 0
        kv = torch.arange(6).view(1, 6)   # [1,6]
        allowed = mask_mod(0, 0, q, kv)
        # absolute query pos 4 -> sees keys 0..4, blocks future key 5.
        self.assertEqual(allowed.tolist(), [[True, True, True, True, True, False]])

    def test_equivalence_with_eager_build_group_attention_masks(self):
        # The flex mask_mod must mark exactly the same (q, kv) pairs allowed
        # as the eager additive mask (0.0 = allowed, -inf = blocked).
        for hot, T, kb in [
            ([0, 1], 5, 4),
            ([0, 2], 6, 4),
            ([0, 1, 2, 3], 8, 6),
            ([0], 5, 3),
        ]:
            eager = build_group_attention_masks([hot], total_seq_len=T, key_boundary=kb)[0]
            eager_allowed = (eager[0, 0] == 0.0)  # [T, T] bool
            is_hot = build_is_hot_vector(hot, total_seq_len=T, key_boundary=kb)
            flex_allowed = self._allowed_grid(is_hot, T)
            self.assertTrue(
                torch.equal(eager_allowed, flex_allowed),
                f"mismatch for hot={hot} T={T} kb={kb}",
            )


class TestBuildGroupBlockMasks(unittest.TestCase):
    def test_returns_a_block_mask_per_group(self):
        masks = build_group_block_masks(
            [[0, 1], [0, 2]], kv_len=8, key_boundary=6
        )
        self.assertEqual(set(masks), {0, 1})
        # Each value is a BlockMask; its shape is padded up to the block
        # multiple, so it covers at least the requested sequence length.
        for bm in masks.values():
            self.assertIsInstance(bm, BlockMask)
            self.assertGreaterEqual(bm.shape[-1], 8)
            self.assertGreaterEqual(bm.shape[-2], 8)

    def test_rejects_a_hot_set_missing_the_sink(self):
        # group hot set excludes key 0 -> position-0 query has no visible
        # key -> degenerate attention row. Reject loudly, as the eager path.
        with self.assertRaises(ValueError):
            build_group_block_masks([[1, 2]], kv_len=5, key_boundary=4)

    def test_incremental_shape_q_len_1(self):
        # Decode-step shape: Q_LEN=1, KV_LEN=cache length, q_start=abs pos.
        masks = build_group_block_masks(
            [[0, 1, 2]], kv_len=8, key_boundary=8, q_len=1, q_start=7
        )
        bm = masks[0]
        self.assertIsInstance(bm, BlockMask)
        self.assertGreaterEqual(bm.shape[-1], 8)   # KV dim
        self.assertGreaterEqual(bm.shape[-2], 1)   # Q dim


class TestFlexVsEagerNumerical(unittest.TestCase):
    """Step 2 correctness gate: flex_attention + our BlockMask must produce
    the same attention output as eager softmax-dot-product + our additive
    mask. Square (prefill) and incremental (decode-step) shapes."""

    @staticmethod
    def _eager_attn(q, k, v, additive_mask):
        d = q.shape[-1]
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5) + additive_mask
        return torch.softmax(scores, dim=-1) @ v

    def test_2a_square_prefill_flex_matches_eager(self):
        torch.manual_seed(0)
        T, d = 24, 16
        hot, kb = [0, 2, 3, 5, 7], 20  # group hot prompt keys; boundary 20
        q, k, v = (torch.randn(1, 1, T, d) for _ in range(3))
        eager_mask = build_group_attention_masks([hot], T, kb)[0]
        out_eager = self._eager_attn(q, k, v, eager_mask)
        bm = build_group_block_masks([hot], kv_len=T, key_boundary=kb)[0]
        out_flex = flex_attention(q, k, v, block_mask=bm)
        torch.testing.assert_close(out_flex, out_eager, atol=2e-4, rtol=2e-4)

    def test_2b_incremental_decode_flex_matches_eager(self):
        torch.manual_seed(1)
        T, d = 24, 16
        hot, kb = [0, 1, 4, 6], 20
        P = T - 1  # the decode query's absolute position
        q = torch.randn(1, 1, 1, d)
        k, v = (torch.randn(1, 1, T, d) for _ in range(2))
        # eager reference: row P of the full mask is the mask for a query
        # at absolute position P attending over all T keys.
        full_mask = build_group_attention_masks([hot], T, kb)[0]
        row_mask = full_mask[:, :, P:P + 1, :]  # [1,1,1,T]
        out_eager = self._eager_attn(q, k, v, row_mask)
        bm = build_group_block_masks(
            [hot], kv_len=T, key_boundary=kb, q_len=1, q_start=P
        )[0]
        out_flex = flex_attention(q, k, v, block_mask=bm)
        torch.testing.assert_close(out_flex, out_eager, atol=2e-4, rtol=2e-4)

    def test_2c_batched_continuation_flex_matches_eager(self):
        # Single-forward masked-decode shape: N continuation queries at
        # absolute positions q_start..q_start+N-1 attend the prompt cache
        # plus each other. The mask_mod's causal term must offset every
        # query by q_start — not only a length-1 query (test_2b).
        torch.manual_seed(2)
        T, d = 28, 16
        hot, kb = [0, 1, 4, 6], 22  # prefill_len = key_boundary = 22
        N = T - kb                  # 6 continuation queries, abs pos 22..27
        q = torch.randn(1, 1, N, d)
        k, v = (torch.randn(1, 1, T, d) for _ in range(2))
        # eager reference: rows kb..kb+N-1 of the full additive mask are
        # the per-query masks for the N continuation positions.
        full_mask = build_group_attention_masks([hot], T, kb)[0]
        rows_mask = full_mask[:, :, kb:kb + N, :]  # [1,1,N,T]
        out_eager = self._eager_attn(q, k, v, rows_mask)
        bm = build_group_block_masks(
            [hot], kv_len=T, key_boundary=kb, q_len=N, q_start=kb
        )[0]
        out_flex = flex_attention(q, k, v, block_mask=bm)
        torch.testing.assert_close(out_flex, out_eager, atol=2e-4, rtol=2e-4)


class TestInstallBlockMaskHooks(unittest.TestCase):
    def test_swaps_attention_mask_with_per_group_block_mask(self):
        model = _FakeModel(2)
        masks = build_group_block_masks(
            [[0, 1], [0, 2]], kv_len=8, key_boundary=6
        )
        handles = install_per_layer_block_mask_hooks(
            model, masks, layer_to_group={0: 0, 1: 1}
        )
        try:
            for layer in model.layers:
                layer.self_attn(torch.zeros(1, 8, 4), attention_mask="ORIGINAL")
        finally:
            for h in handles:
                h.remove()
        self.assertIs(model.layers[0].self_attn.received_mask, masks[0])
        self.assertIs(model.layers[1].self_attn.received_mask, masks[1])

    def test_untracked_layers_keep_their_original_mask(self):
        model = _FakeModel(2)
        masks = build_group_block_masks([[0, 1]], kv_len=8, key_boundary=6)
        # only layer 0 is mapped to a group
        handles = install_per_layer_block_mask_hooks(
            model, masks, layer_to_group={0: 0}
        )
        try:
            for layer in model.layers:
                layer.self_attn(torch.zeros(1, 8, 4), attention_mask="ORIGINAL")
        finally:
            for h in handles:
                h.remove()
        self.assertIs(model.layers[0].self_attn.received_mask, masks[0])
        self.assertEqual(model.layers[1].self_attn.received_mask, "ORIGINAL")


if __name__ == "__main__":
    unittest.main()
