import math
import unittest

import torch

from src.dart_pagedkv.logit_kl import (
    build_group_attention_masks,
    install_per_layer_mask_hooks,
    token_kl_divergence,
)


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


class TokenKLDivergenceTests(unittest.TestCase):
    def test_identical_logits_give_zero(self):
        logits = torch.randn(3, 50)
        self.assertAlmostEqual(token_kl_divergence(logits, logits.clone()), 0.0, places=6)

    def test_peaked_full_vs_uniform_hot_equals_log_vocab(self):
        # full puts ~all mass on token 0; hot is uniform over 4 tokens.
        full = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        hot = torch.zeros(1, 4)
        self.assertAlmostEqual(token_kl_divergence(full, hot), math.log(4), places=4)

    def test_averages_over_positions(self):
        # position 0: KL = log 4; position 1: identical -> KL = 0. Mean = log(4)/2.
        full = torch.tensor([[100.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
        hot = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
        self.assertAlmostEqual(token_kl_divergence(full, hot), math.log(4) / 2, places=4)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            token_kl_divergence(torch.randn(3, 50), torch.randn(2, 50))

    def test_empty_positions_returns_zero(self):
        self.assertEqual(
            token_kl_divergence(torch.zeros(0, 10), torch.zeros(0, 10)), 0.0
        )


class BuildGroupAttentionMasksTests(unittest.TestCase):
    def test_mask_shape_is_one_one_t_t(self):
        masks = build_group_attention_masks([[0, 1]], total_seq_len=5, key_boundary=4)
        self.assertEqual(masks[0].shape, (1, 1, 5, 5))

    def test_causal_blocks_future_keys(self):
        masks = build_group_attention_masks(
            [[0, 1, 2, 3]], total_seq_len=5, key_boundary=4
        )
        m = masks[0]
        self.assertEqual(m[0, 0, 1, 2].item(), float("-inf"))  # query 1 cannot see key 2
        self.assertEqual(m[0, 0, 3, 1].item(), 0.0)  # query 3 can see key 1

    def test_cold_prompt_keys_are_blocked(self):
        # group hot = {0, 1}; key_boundary 4 -> cold prompt keys = {2, 3}.
        masks = build_group_attention_masks([[0, 1]], total_seq_len=5, key_boundary=4)
        m = masks[0]
        self.assertEqual(m[0, 0, 4, 2].item(), float("-inf"))
        self.assertEqual(m[0, 0, 4, 3].item(), float("-inf"))
        self.assertEqual(m[0, 0, 4, 0].item(), 0.0)
        self.assertEqual(m[0, 0, 4, 1].item(), 0.0)

    def test_tail_keys_past_key_boundary_stay_hot(self):
        # key 4 is the answer tail (>= key_boundary 4) -> allowed (subject to causal).
        masks = build_group_attention_masks([[0]], total_seq_len=5, key_boundary=4)
        self.assertEqual(masks[0][0, 0, 4, 4].item(), 0.0)

    def test_distinct_masks_per_group(self):
        # Both groups keep key 0 (the sink); they differ on which others are cold.
        masks = build_group_attention_masks(
            [[0, 1], [0, 2]], total_seq_len=5, key_boundary=4
        )
        self.assertEqual(masks[0][0, 0, 4, 2].item(), float("-inf"))  # group 0 blocks 2
        self.assertEqual(masks[1][0, 0, 4, 2].item(), 0.0)  # group 1 keeps 2
        self.assertEqual(masks[1][0, 0, 4, 1].item(), float("-inf"))  # group 1 blocks 1

    def test_rejects_a_hot_set_missing_the_sink(self):
        # A group whose hot set excludes key 0 would give the position-0 query
        # an all-masked attention row -> softmax NaN. Reject it loudly.
        with self.assertRaises(ValueError):
            build_group_attention_masks([[1, 2, 3]], total_seq_len=5, key_boundary=4)


class InstallPerLayerMaskHooksTests(unittest.TestCase):
    def test_each_layer_receives_its_groups_mask(self):
        model = _FakeModel(3)
        group_masks = {
            0: torch.full((1, 1, 2, 2), 1.0),
            1: torch.full((1, 1, 2, 2), 2.0),
        }
        layer_to_group = {0: 0, 1: 1, 2: 0}
        handles = install_per_layer_mask_hooks(model, group_masks, layer_to_group)
        try:
            hidden = torch.zeros(1, 2, 4)
            for layer in model.layers:
                layer.self_attn(hidden, attention_mask=torch.zeros(1, 1, 2, 2))
            # The default-arg closure capture must give each layer its own mask.
            self.assertTrue(torch.equal(model.layers[0].self_attn.received_mask, group_masks[0]))
            self.assertTrue(torch.equal(model.layers[1].self_attn.received_mask, group_masks[1]))
            self.assertTrue(torch.equal(model.layers[2].self_attn.received_mask, group_masks[0]))
        finally:
            for handle in handles:
                handle.remove()

    def test_layers_absent_from_layer_to_group_are_untouched(self):
        model = _FakeModel(2)
        handles = install_per_layer_mask_hooks(model, {0: torch.ones(1, 1, 2, 2)}, {0: 0})
        try:
            placeholder = torch.zeros(1, 1, 2, 2)
            model.layers[1].self_attn(torch.zeros(1, 2, 4), attention_mask=placeholder)
            self.assertTrue(torch.equal(model.layers[1].self_attn.received_mask, placeholder))
        finally:
            for handle in handles:
                handle.remove()

    def test_removed_hooks_stop_swapping(self):
        model = _FakeModel(1)
        handles = install_per_layer_mask_hooks(model, {0: torch.ones(1, 1, 2, 2)}, {0: 0})
        for handle in handles:
            handle.remove()
        placeholder = torch.zeros(1, 1, 2, 2)
        model.layers[0].self_attn(torch.zeros(1, 2, 4), attention_mask=placeholder)
        self.assertTrue(torch.equal(model.layers[0].self_attn.received_mask, placeholder))


if __name__ == "__main__":
    unittest.main()
