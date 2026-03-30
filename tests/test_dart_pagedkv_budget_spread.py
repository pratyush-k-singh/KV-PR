import unittest

from src.dart_pagedkv.budget import uniform_budget, validate_budget
from src.dart_pagedkv.budget_spread import predeclared_spread, ratio_allocation_grid


class PredeclaredSpreadTests(unittest.TestCase):
    def test_every_vector_is_a_valid_budget(self):
        spread = predeclared_spread(num_groups=4, total=64, floors=[0, 0, 0, 0])
        for b in spread:
            validate_budget(b, total=64, floors=[0, 0, 0, 0])

    def test_includes_the_uniform_vector(self):
        spread = predeclared_spread(num_groups=4, total=64, floors=[0, 0, 0, 0])
        uniform = uniform_budget(num_groups=4, total=64, floors=[0, 0, 0, 0])
        self.assertIn(uniform, spread)

    def test_includes_single_dominant_per_group(self):
        spread = predeclared_spread(num_groups=3, total=60, floors=[2, 2, 2])
        expected_max = 60 - 2 * 2
        for g in range(3):
            self.assertTrue(
                any(
                    b[g] == expected_max
                    and all(b[h] == 2 for h in range(3) if h != g)
                    for b in spread
                ),
                msg=f"no single-dominant vector for group {g}",
            )

    def test_includes_single_starved_per_group(self):
        spread = predeclared_spread(num_groups=3, total=60, floors=[2, 2, 2])
        for g in range(3):
            self.assertTrue(
                any(b[g] == 2 and sum(b[h] for h in range(3) if h != g) == 58 for b in spread),
                msg=f"no single-starved vector for group {g}",
            )

    def test_size_is_at_least_2g_plus_1_for_g4(self):
        spread = predeclared_spread(num_groups=4, total=64, floors=[0, 0, 0, 0])
        self.assertGreaterEqual(len(spread), 2 * 4 + 1)

    def test_no_duplicate_vectors(self):
        spread = predeclared_spread(num_groups=4, total=64, floors=[0, 0, 0, 0])
        unique = {tuple(b) for b in spread}
        self.assertEqual(len(unique), len(spread))

    def test_g_equals_one_returns_just_uniform(self):
        spread = predeclared_spread(num_groups=1, total=64, floors=[0])
        self.assertEqual(spread, [[64]])


class RatioAllocationGridTests(unittest.TestCase):
    def test_total_is_ratio_times_prompt_len(self):
        grid = ratio_allocation_grid(
            prompt_len=10000, ratios=[0.01, 0.04], num_groups=4,
            floors=[0, 0, 0, 0],
        )
        self.assertEqual([e["ratio"] for e in grid], [0.01, 0.04])
        self.assertEqual([e["total"] for e in grid], [100, 400])

    def test_each_entry_carries_a_valid_allocation_grid(self):
        grid = ratio_allocation_grid(
            prompt_len=8192, ratios=[0.02], num_groups=4, floors=[0, 0, 0, 0],
        )
        entry = grid[0]
        for allocation in entry["allocations"]:
            validate_budget(allocation, total=entry["total"],
                            floors=[0, 0, 0, 0])

    def test_tiny_ratio_clamps_total_up_to_floor_sum(self):
        # 0.5% of 2048 = 10, but floors sum to 16 — total clamps to 16.
        grid = ratio_allocation_grid(
            prompt_len=2048, ratios=[0.005], num_groups=4, floors=[4, 4, 4, 4],
        )
        self.assertEqual(grid[0]["total"], 16)

    def test_rejects_nonpositive_ratio(self):
        with self.assertRaises(ValueError):
            ratio_allocation_grid(2048, ratios=[0.0], num_groups=4,
                                  floors=[0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
