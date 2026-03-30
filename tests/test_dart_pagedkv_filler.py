import unittest

from src.dart_pagedkv.filler import protected_positions, select_hot


class ProtectedPositionsTests(unittest.TestCase):
    def test_combines_sink_and_recent_positions(self):
        self.assertEqual(
            protected_positions(10, n_sink=2, n_recent=3),
            [0, 1, 7, 8, 9],
        )

    def test_overlap_is_deduplicated(self):
        self.assertEqual(
            protected_positions(5, n_sink=3, n_recent=3),
            [0, 1, 2, 3, 4],
        )

    def test_zero_sink_and_recent_gives_empty(self):
        self.assertEqual(protected_positions(10, n_sink=0, n_recent=0), [])

    def test_clamps_counts_to_sequence_length(self):
        self.assertEqual(
            protected_positions(4, n_sink=10, n_recent=10),
            [0, 1, 2, 3],
        )


class SelectHotTests(unittest.TestCase):
    def test_selects_the_highest_priority_positions(self):
        self.assertEqual(select_hot([0.1, 0.9, 0.5, 0.2], budget=2), [1, 2])

    def test_ties_break_to_the_lowest_index(self):
        self.assertEqual(select_hot([0.5, 0.5, 0.5], budget=2), [0, 1])

    def test_budget_at_or_above_length_returns_all_positions(self):
        self.assertEqual(select_hot([0.1, 0.2, 0.3], budget=5), [0, 1, 2])

    def test_protected_positions_are_forced_in(self):
        self.assertEqual(
            select_hot([0.9, 0.8, 0.7, 0.0], budget=2, protected=[3]),
            [0, 3],
        )

    def test_result_is_sorted_and_has_budget_length(self):
        hot = select_hot([0.3, 0.1, 0.7, 0.2, 0.9], budget=3)
        self.assertEqual(hot, sorted(hot))
        self.assertEqual(len(hot), 3)

    def test_hot_sets_nest_as_budget_grows(self):
        priority = [0.1, 0.5, 0.9, 0.3, 0.7]
        smaller = set(select_hot(priority, budget=2))
        larger = set(select_hot(priority, budget=3))
        largest = set(select_hot(priority, budget=4))
        self.assertTrue(smaller.issubset(larger))
        self.assertTrue(larger.issubset(largest))

    def test_hot_sets_nest_with_protected_positions(self):
        priority = [0.1, 0.5, 0.9, 0.3, 0.7]
        smaller = set(select_hot(priority, budget=2, protected=[0]))
        larger = set(select_hot(priority, budget=3, protected=[0]))
        self.assertTrue(smaller.issubset(larger))


if __name__ == "__main__":
    unittest.main()
