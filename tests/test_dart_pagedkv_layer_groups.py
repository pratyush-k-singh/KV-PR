import unittest

from src.dart_pagedkv.layer_groups import layer_to_group, make_layer_groups


class MakeLayerGroupsTests(unittest.TestCase):
    def test_even_split_produces_equal_contiguous_groups(self):
        self.assertEqual(
            make_layer_groups(12, 4),
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        )

    def test_uneven_split_gives_remainder_to_earlier_groups(self):
        self.assertEqual(
            make_layer_groups(10, 3),
            [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )

    def test_single_group_holds_all_layers(self):
        self.assertEqual(make_layer_groups(8, 1), [[0, 1, 2, 3, 4, 5, 6, 7]])

    def test_group_per_layer_produces_singletons(self):
        self.assertEqual(make_layer_groups(5, 5), [[0], [1], [2], [3], [4]])

    def test_groups_partition_every_layer_exactly_once(self):
        groups = make_layer_groups(17, 4)
        flat = [layer for group in groups for layer in group]
        self.assertEqual(flat, list(range(17)))
        self.assertEqual(len(groups), 4)

    def test_group_sizes_are_non_increasing(self):
        sizes = [len(group) for group in make_layer_groups(17, 4)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_zero_groups_raises(self):
        with self.assertRaises(ValueError):
            make_layer_groups(8, 0)

    def test_more_groups_than_layers_raises(self):
        with self.assertRaises(ValueError):
            make_layer_groups(4, 5)

    def test_negative_layers_raises(self):
        with self.assertRaises(ValueError):
            make_layer_groups(-1, 1)


class LayerToGroupTests(unittest.TestCase):
    def test_inverts_a_partition_to_layer_group_map(self):
        groups = make_layer_groups(10, 3)
        self.assertEqual(
            layer_to_group(groups),
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2},
        )

    def test_maps_every_layer_exactly_once(self):
        groups = make_layer_groups(17, 4)
        mapping = layer_to_group(groups)
        self.assertEqual(sorted(mapping), list(range(17)))

    def test_round_trips_with_make_layer_groups(self):
        groups = make_layer_groups(13, 5)
        mapping = layer_to_group(groups)
        for group_idx, layers in enumerate(groups):
            for layer in layers:
                self.assertEqual(mapping[layer], group_idx)


if __name__ == "__main__":
    unittest.main()
