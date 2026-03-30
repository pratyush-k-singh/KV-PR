import unittest

from src.dart_pagedkv.budget import (
    interpolate_budget,
    largest_remainder_round,
    uniform_budget,
    validate_budget,
)


class ValidateBudgetTests(unittest.TestCase):
    def test_accepts_a_valid_budget(self):
        validate_budget([3, 3, 2], total=8, floors=[0, 0, 0])

    def test_accepts_a_budget_meeting_floors_exactly(self):
        validate_budget([2, 2, 2], total=6, floors=[2, 2, 2])

    def test_rejects_wrong_sum(self):
        with self.assertRaises(ValueError):
            validate_budget([3, 3, 3], total=8, floors=[0, 0, 0])

    def test_rejects_entry_below_floor(self):
        with self.assertRaises(ValueError):
            validate_budget([1, 4, 3], total=8, floors=[2, 0, 0])

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            validate_budget([4, 4], total=8, floors=[0, 0, 0])

    def test_rejects_non_integer_entry(self):
        with self.assertRaises(ValueError):
            validate_budget([3.5, 3.0, 1.5], total=8, floors=[0, 0, 0])


class LargestRemainderRoundTests(unittest.TestCase):
    def test_rounds_to_integers_summing_to_total(self):
        result = largest_remainder_round([2.5, 2.5, 3.0], total=8, floors=[0, 0, 0])
        self.assertEqual(result, [3, 2, 3])
        self.assertEqual(sum(result), 8)

    def test_respects_floors(self):
        result = largest_remainder_round([2.4, 2.6, 3.0], total=8, floors=[2, 2, 2])
        self.assertEqual(result, [2, 3, 3])
        self.assertEqual(sum(result), 8)
        for value, floor in zip(result, [2, 2, 2]):
            self.assertGreaterEqual(value, floor)

    def test_tie_break_goes_to_lowest_index(self):
        result = largest_remainder_round([0.5, 0.5, 0.5, 0.5], total=2, floors=[0, 0, 0, 0])
        self.assertEqual(result, [1, 1, 0, 0])

    def test_is_deterministic(self):
        first = largest_remainder_round([2.5, 2.5, 3.0], total=8, floors=[0, 0, 0])
        second = largest_remainder_round([2.5, 2.5, 3.0], total=8, floors=[0, 0, 0])
        self.assertEqual(first, second)

    def test_integral_input_is_returned_unchanged(self):
        self.assertEqual(
            largest_remainder_round([3, 3, 2], total=8, floors=[0, 0, 0]),
            [3, 3, 2],
        )

    def test_rejects_fractional_entry_below_floor(self):
        with self.assertRaises(ValueError):
            largest_remainder_round([1.0, 4.0, 3.0], total=8, floors=[2, 0, 0])

    def test_rejects_floors_exceeding_total(self):
        with self.assertRaises(ValueError):
            largest_remainder_round([4.0, 4.0], total=8, floors=[5, 5])


class InterpolateBudgetTests(unittest.TestCase):
    def test_lam_one_returns_the_advice_budget(self):
        result = interpolate_budget([5, 2, 1], [1, 3, 4], lam=1.0, total=8, floors=[0, 0, 0])
        self.assertEqual(result, [5, 2, 1])

    def test_lam_zero_returns_the_robust_budget(self):
        result = interpolate_budget([5, 2, 1], [1, 3, 4], lam=0.0, total=8, floors=[0, 0, 0])
        self.assertEqual(result, [1, 3, 4])

    def test_blend_sums_to_total_and_respects_floors(self):
        result = interpolate_budget([5, 2, 1], [1, 3, 4], lam=0.5, total=8, floors=[0, 0, 0])
        self.assertEqual(result, [3, 3, 2])
        validate_budget(result, total=8, floors=[0, 0, 0])

    def test_lam_is_clamped_to_unit_interval(self):
        high = interpolate_budget([5, 2, 1], [1, 3, 4], lam=2.0, total=8, floors=[0, 0, 0])
        low = interpolate_budget([5, 2, 1], [1, 3, 4], lam=-1.0, total=8, floors=[0, 0, 0])
        self.assertEqual(high, [5, 2, 1])
        self.assertEqual(low, [1, 3, 4])


class UniformBudgetTests(unittest.TestCase):
    def test_even_total_splits_equally(self):
        self.assertEqual(uniform_budget(4, total=12, floors=[0, 0, 0, 0]), [3, 3, 3, 3])

    def test_uneven_total_uses_largest_remainder(self):
        result = uniform_budget(3, total=8, floors=[0, 0, 0])
        self.assertEqual(result, [3, 3, 2])
        self.assertEqual(sum(result), 8)

    def test_respects_floors(self):
        result = uniform_budget(3, total=11, floors=[2, 2, 2])
        self.assertEqual(sum(result), 11)
        for value in result:
            self.assertGreaterEqual(value, 2)

    def test_rejects_floors_length_mismatch(self):
        with self.assertRaises(ValueError):
            uniform_budget(4, total=12, floors=[0, 0, 0])


if __name__ == "__main__":
    unittest.main()
