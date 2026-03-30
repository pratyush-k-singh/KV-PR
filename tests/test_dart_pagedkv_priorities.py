import unittest

import numpy as np

from src.dart_pagedkv.priorities import snapkv_priority


class SnapKVPriorityTests(unittest.TestCase):
    def test_priority_uses_last_obs_window_rows(self):
        matrix = np.zeros((10, 10), dtype=np.float32)
        matrix[5:] = 1.0
        priority = snapkv_priority(matrix, obs_window=5)
        np.testing.assert_array_equal(priority, np.full(10, 5.0, dtype=np.float32))

    def test_priority_assigns_more_to_attended_tokens(self):
        matrix = np.zeros((5, 5), dtype=np.float32)
        matrix[-1, 0] = 5.0
        matrix[-1, 2] = 1.0
        priority = snapkv_priority(matrix, obs_window=1)
        self.assertEqual(priority[0], 5.0)
        self.assertEqual(priority[2], 1.0)
        self.assertEqual(priority[1], 0.0)

    def test_obs_window_larger_than_seq_len_uses_all_rows(self):
        matrix = np.ones((5, 5), dtype=np.float32)
        priority = snapkv_priority(matrix, obs_window=100)
        np.testing.assert_array_equal(priority, np.full(5, 5.0, dtype=np.float32))

    def test_output_shape_matches_seq_len(self):
        matrix = np.random.rand(8, 8).astype(np.float32)
        priority = snapkv_priority(matrix, obs_window=3)
        self.assertEqual(priority.shape, (8,))

    def test_rejects_non_square_matrix(self):
        with self.assertRaises(ValueError):
            snapkv_priority(np.zeros((3, 5), dtype=np.float32), obs_window=2)

    def test_rejects_non_positive_obs_window(self):
        with self.assertRaises(ValueError):
            snapkv_priority(np.zeros((5, 5), dtype=np.float32), obs_window=0)


if __name__ == "__main__":
    unittest.main()
