import unittest

from src.benchmarks.infinitebench import _answers_from_row, build_prompt_from_row


class InfiniteBenchLoaderTests(unittest.TestCase):
    def test_choice_prompt_includes_options(self):
        prompt = build_prompt_from_row(
            {
                "context": "Book text",
                "input": "Who is missing?",
                "options": ["Alice", "Bob"],
            }
        )

        self.assertIn("Options:", prompt)
        self.assertIn("- Alice", prompt)
        self.assertIn("exact option text", prompt)

    def test_math_calc_answer_sequence_stays_one_reference(self):
        refs = _answers_from_row({"answer": [1, 2, 3]}, "math_calc")

        self.assertEqual(refs, ["1 2 3"])


if __name__ == "__main__":
    unittest.main()
