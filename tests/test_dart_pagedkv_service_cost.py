import unittest

import torch

from src.dart_pagedkv.service_cost import cold_attention_demand


class ColdAttentionDemandTests(unittest.TestCase):
    def test_all_keys_hot_gives_zero_demand(self):
        query = torch.randn(2, 3, 4)
        key = torch.randn(2, 5, 4)
        demand = cold_attention_demand(
            query, key, hot_key_indices=list(range(5)), scaling=0.5
        )
        self.assertAlmostEqual(demand, 0.0, places=6)

    def test_all_keys_cold_gives_full_demand(self):
        query = torch.randn(2, 3, 4)
        key = torch.randn(2, 5, 4)
        demand = cold_attention_demand(query, key, hot_key_indices=[], scaling=0.5)
        self.assertAlmostEqual(demand, 1.0, places=6)

    def test_uniform_attention_yields_cold_fraction(self):
        # Zero queries -> all scores 0 -> uniform softmax over attended keys.
        query = torch.zeros(1, 1, 4)
        key = torch.randn(1, 4, 4)
        # Query 0 at offset 3 attends causally to keys 0..3 (all four).
        demand = cold_attention_demand(
            query, key, hot_key_indices=[3], scaling=1.0, query_offset=3
        )
        self.assertAlmostEqual(demand, 0.75, places=6)

    def test_causal_mask_excludes_future_keys(self):
        # Query 0 at offset 1 attends to keys 0,1 only -> uniform [0.5, 0.5].
        query = torch.zeros(1, 1, 4)
        key = torch.randn(1, 4, 4)
        demand = cold_attention_demand(
            query, key, hot_key_indices=[1], scaling=1.0, query_offset=1
        )
        self.assertAlmostEqual(demand, 0.5, places=6)

    def test_gqa_maps_query_heads_to_kv_heads(self):
        # 4 query heads, 2 kv heads, group factor 2; zero queries -> uniform.
        query = torch.zeros(4, 1, 4)
        key = torch.randn(2, 2, 4)
        demand = cold_attention_demand(
            query,
            key,
            hot_key_indices=[],
            scaling=1.0,
            num_key_value_groups=2,
            query_offset=1,
        )
        self.assertAlmostEqual(demand, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
