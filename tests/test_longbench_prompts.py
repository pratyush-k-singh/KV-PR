import unittest

from src.benchmarks.longbench import PromptRecord, build_prompt_from_row, generate_stratified_split, select_split_records
from experiments.scripts.evaluate_live import _select_balanced_eval_subset


class LongBenchPromptTests(unittest.TestCase):
    def test_passage_count_prompt_adds_missing_task_instruction(self):
        prompt = build_prompt_from_row(
            {
                "dataset": "passage_count",
                "context": "Paragraph 1: A\n\nParagraph 2: A",
                "input": "",
            }
        )

        self.assertIn("how many unique paragraphs", prompt)
        self.assertIn("The final answer is:", prompt)
        self.assertNotIn("Question:\n\n", prompt)

    def test_multidoc_qa_prompt_requests_answer_only(self):
        prompt = build_prompt_from_row(
            {
                "dataset": "hotpotqa",
                "context": "Passage 1:\nA fact.",
                "input": "What is the fact?",
            }
        )

        self.assertIn("given passages", prompt)
        self.assertIn("Only give me the answer", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_three_way_split_has_disjoint_named_fields(self):
        records = [
            PromptRecord(f"a_{idx:04d}", "a", "cat", "prompt", ["answer"])
            for idx in range(10)
        ] + [
            PromptRecord(f"b_{idx:04d}", "b", "cat", "prompt", ["answer"])
            for idx in range(10)
        ]

        split = generate_stratified_split(records, n_train=6, n_dev=4, n_test=6, seed=1)

        self.assertEqual(len(split["train"]), 6)
        self.assertEqual(len(split["dev"]), 4)
        self.assertEqual(len(split["test"]), 6)
        self.assertEqual(split["eval"], split["test"])
        assigned = split["train"] + split["dev"] + split["test"] + split["unused"]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual([r.prompt_id for r in select_split_records(records, split, "dev")], split["dev"])

    def test_seeded_balanced_eval_subset_samples_within_each_subset(self):
        records = [
            PromptRecord(f"a_{idx:04d}", "a", "cat", "prompt", ["answer"])
            for idx in range(20)
        ] + [
            PromptRecord(f"b_{idx:04d}", "b", "cat", "prompt", ["answer"])
            for idx in range(20)
        ]

        ordered = _select_balanced_eval_subset(records, 4)
        seeded = _select_balanced_eval_subset(records, 4, seed=7)

        self.assertEqual([record.subset for record in seeded], ["a", "b", "a", "b"])
        self.assertNotEqual([record.prompt_id for record in ordered], [record.prompt_id for record in seeded])


if __name__ == "__main__":
    unittest.main()
