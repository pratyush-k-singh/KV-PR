import unittest

import numpy as np
import torch

from src.trace.collect import (
    LayerQKCapture,
    extract_training_samples,
    format_prompt_for_tokenizer,
    middle_truncate_inputs,
    recompute_attention_matrix,
    stream_training_samples_from_qk_layers,
)


def _make_capture(query, key, *, scaling=1.0, num_key_value_groups=1):
    return LayerQKCapture(
        query=torch.tensor(query, dtype=torch.float32),
        key=torch.tensor(key, dtype=torch.float32),
        scaling=scaling,
        num_key_value_groups=num_key_value_groups,
    )


class StreamTrainingSampleTests(unittest.TestCase):
    def test_middle_truncate_keeps_prompt_head_and_tail(self):
        encoded = {
            "input_ids": torch.arange(10).view(1, 10),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
        }

        truncated = middle_truncate_inputs(encoded, max_length=6)

        self.assertEqual(truncated["input_ids"].tolist(), [[0, 1, 2, 7, 8, 9]])
        self.assertEqual(truncated["attention_mask"].shape, (1, 6))

    def test_chat_template_auto_is_noop_without_template(self):
        class Tokenizer:
            chat_template = None

        self.assertEqual(format_prompt_for_tokenizer(Tokenizer(), "hello", chat_template="auto"), "hello")

    def test_chat_template_always_requires_template(self):
        class Tokenizer:
            chat_template = None

        with self.assertRaises(ValueError):
            format_prompt_for_tokenizer(Tokenizer(), "hello", chat_template="always")

    def test_streamed_single_layer_matches_recomputed_attention(self):
        capture = _make_capture(
            [[[[1.0, 0.0], [0.2, 0.8], [0.5, 0.5], [0.9, 0.1]],
              [[0.0, 1.0], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]]],
            [[[[1.0, 0.0], [0.0, 1.0], [0.7, 0.3], [0.3, 0.7]],
              [[0.0, 1.0], [1.0, 0.0], [0.4, 0.6], [0.6, 0.4]]]],
        )

        streamed_x, streamed_y = stream_training_samples_from_qk_layers(
            [capture],
            window_size=3,
            stride=1,
            position_stride=2,
            row_batch_size=2,
            heads_per_batch=1,
            device="cpu",
        )
        matrix = recompute_attention_matrix({0: capture}, device="cpu", heads_per_batch=1)
        dense_x, dense_y = extract_training_samples(
            matrix,
            window_size=3,
            stride=1,
            position_stride=2,
        )

        self.assertTrue(np.allclose(streamed_x, dense_x, atol=1e-6))
        self.assertTrue(np.allclose(streamed_y, dense_y, atol=1e-6))

    def test_streamed_multi_layer_matches_recomputed_attention(self):
        capture_a = _make_capture(
            [[[[1.0, 0.0], [0.2, 0.8], [0.5, 0.5], [0.9, 0.1]],
              [[0.0, 1.0], [0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]]],
            [[[[1.0, 0.0], [0.0, 1.0], [0.7, 0.3], [0.3, 0.7]],
              [[0.0, 1.0], [1.0, 0.0], [0.4, 0.6], [0.6, 0.4]]]],
        )
        capture_b = _make_capture(
            [[[[0.9, 0.1], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]],
              [[0.1, 0.9], [0.6, 0.4], [0.7, 0.3], [0.4, 0.6]]]],
            [[[[0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]],
              [[0.2, 0.8], [0.8, 0.2], [0.5, 0.5], [0.9, 0.1]]]],
        )
        captures = {0: capture_a, 1: capture_b}

        streamed_x, streamed_y = stream_training_samples_from_qk_layers(
            captures,
            window_size=2,
            stride=1,
            position_stride=1,
            row_batch_size=2,
            heads_per_batch=2,
            device="cpu",
        )
        matrix = recompute_attention_matrix(captures, device="cpu", heads_per_batch=2)
        dense_x, dense_y = extract_training_samples(
            matrix,
            window_size=2,
            stride=1,
            position_stride=1,
        )

        self.assertTrue(np.allclose(streamed_x, dense_x, atol=1e-6))
        self.assertTrue(np.allclose(streamed_y, dense_y, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
