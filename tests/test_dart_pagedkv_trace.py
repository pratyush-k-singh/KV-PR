import unittest

import torch

from src.dart_pagedkv.trace import LayerQKV, _assemble_qkv_layers
from src.trace.collect import LayerQKCapture


def _make_capture(scaling: float = 0.5, ngroups: int = 1) -> LayerQKCapture:
    return LayerQKCapture(
        query=torch.tensor([[[[1.0, 2.0]]]]),   # [1, 1, 1, 2]
        key=torch.tensor([[[[99.0, 99.0]]]]),
        scaling=scaling,
        num_key_value_groups=ngroups,
    )


class AssembleQKVLayersTests(unittest.TestCase):
    def test_extracts_query_from_capture_and_kv_from_cache(self):
        cached_k = torch.tensor([[[[10.0, 20.0]]]])
        cached_v = torch.tensor([[[[30.0, 40.0]]]])
        capture = _make_capture(scaling=0.25, ngroups=2)

        result = _assemble_qkv_layers({0: capture}, [(cached_k, cached_v)])

        self.assertIn(0, result)
        layer = result[0]
        self.assertIsInstance(layer, LayerQKV)
        self.assertTrue(torch.equal(layer.query, torch.tensor([[[1.0, 2.0]]])))
        self.assertTrue(torch.equal(layer.key, torch.tensor([[[10.0, 20.0]]])))
        self.assertTrue(torch.equal(layer.value, torch.tensor([[[30.0, 40.0]]])))
        self.assertEqual(layer.scaling, 0.25)
        self.assertEqual(layer.num_key_value_groups, 2)

    def test_strips_the_batch_dimension(self):
        cached_k = torch.zeros(1, 2, 3, 4)
        cached_v = torch.zeros(1, 2, 3, 4)
        result = _assemble_qkv_layers(
            {0: LayerQKCapture(query=torch.zeros(1, 4, 3, 4), key=torch.zeros(1, 2, 3, 4), scaling=1.0, num_key_value_groups=2)},
            [(cached_k, cached_v)],
        )
        self.assertEqual(result[0].query.shape, (4, 3, 4))
        self.assertEqual(result[0].key.shape, (2, 3, 4))
        self.assertEqual(result[0].value.shape, (2, 3, 4))

    def test_skips_layers_missing_from_cache(self):
        capture = _make_capture()
        result = _assemble_qkv_layers({5: capture}, [(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))])
        self.assertNotIn(5, result)

    def test_handles_multiple_layers(self):
        captures = {
            0: _make_capture(scaling=0.1, ngroups=1),
            1: _make_capture(scaling=0.2, ngroups=1),
        }
        legacy = [
            (torch.tensor([[[[1.0, 1.0]]]]), torch.tensor([[[[2.0, 2.0]]]])),
            (torch.tensor([[[[3.0, 3.0]]]]), torch.tensor([[[[4.0, 4.0]]]])),
        ]
        result = _assemble_qkv_layers(captures, legacy)
        self.assertEqual(sorted(result.keys()), [0, 1])
        self.assertEqual(result[0].scaling, 0.1)
        self.assertEqual(result[1].scaling, 0.2)


if __name__ == "__main__":
    unittest.main()
