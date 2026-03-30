import unittest

from src.dart_pagedkv.two_tier import cold_indices, group_hot_sets


class GroupHotSetsTests(unittest.TestCase):
    def test_one_hot_list_per_group(self):
        priorities = [[0.1, 0.9, 0.5, 0.2], [0.8, 0.1, 0.2, 0.7]]
        hot = group_hot_sets([2, 2], priorities, seq_len=4, n_sink=0, n_recent=0)
        self.assertEqual(len(hot), 2)
        self.assertEqual(hot[0], [1, 2])
        self.assertEqual(hot[1], [0, 3])

    def test_per_group_budgets_size_the_hot_sets(self):
        priorities = [[0.1, 0.9, 0.5, 0.2], [0.8, 0.1, 0.2, 0.7]]
        hot = group_hot_sets([1, 3], priorities, seq_len=4, n_sink=0, n_recent=0)
        self.assertEqual(len(hot[0]), 1)
        self.assertEqual(len(hot[1]), 3)

    def test_protected_positions_are_hot_in_every_group(self):
        priorities = [[0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.9]]
        hot = group_hot_sets([2, 2], priorities, seq_len=4, n_sink=1, n_recent=1)
        for group_hot in hot:
            self.assertIn(0, group_hot)
            self.assertIn(3, group_hot)

    def test_rejects_budget_priority_length_mismatch(self):
        with self.assertRaises(ValueError):
            group_hot_sets([2], [[0.1, 0.2], [0.3, 0.4]], seq_len=2, n_sink=0, n_recent=0)

    def test_rejects_priority_length_not_seq_len(self):
        with self.assertRaises(ValueError):
            group_hot_sets([1], [[0.1, 0.2, 0.3]], seq_len=2, n_sink=0, n_recent=0)


class ColdIndicesTests(unittest.TestCase):
    def test_returns_the_sorted_complement(self):
        self.assertEqual(cold_indices([1, 2], seq_len=5), [0, 3, 4])

    def test_empty_hot_gives_all_indices(self):
        self.assertEqual(cold_indices([], seq_len=4), [0, 1, 2, 3])

    def test_full_hot_gives_empty(self):
        self.assertEqual(cold_indices([0, 1, 2], seq_len=3), [])

    def test_out_of_range_hot_values_are_ignored(self):
        self.assertEqual(cold_indices([1, 9], seq_len=3), [0, 2])


if __name__ == "__main__":
    unittest.main()
