"""LongBench loading and split helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


LONG_BENCH_ENGLISH_SUBSETS: Dict[str, str] = {
    "narrativeqa": "1doc_qa",
    "qasper": "1doc_qa",
    "multifieldqa_en": "1doc_qa",
    "hotpotqa": "mdoc_qa",
    "2wikimqa": "mdoc_qa",
    "musique": "mdoc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "trec": "few_shot",
    "triviaqa": "few_shot",
    "samsum": "few_shot",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}

LONG_BENCH_PROMPT_TEMPLATES: Dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. Answer the "
        "query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any "
        "other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please "
        "carefully read these paragraphs and determine how many unique paragraphs there are after removing "
        "duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph "
        "the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below.\n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below.\n{context}{input}Next line of code:\n",
}

LONG_BENCH_MAX_NEW_TOKENS: Dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}


@dataclass(slots=True)
class PromptRecord:
    """One evaluation or training prompt."""

    prompt_id: str
    subset: str
    category: str
    prompt: str
    answers: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_answer_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def build_prompt_from_row(row: Dict[str, Any], subset: str | None = None) -> str:
    """Create a prompt string from a LongBench row."""
    dataset = str(subset or row.get("dataset") or "").strip()
    instruction = str(row.get("instruction", "")).strip()
    context = str(row.get("context", "")).strip()
    question = str(row.get("input", "")).strip()

    template = LONG_BENCH_PROMPT_TEMPLATES.get(dataset)
    if template is not None:
        return template.format(context=context, input=question).strip()

    fields = []
    if instruction:
        fields.append(instruction)
    if context:
        fields.append(context)
    if question:
        fields.append(f"Question:\n{question}")

    prompt = "\n\n".join(part for part in fields if part).strip()
    if prompt:
        return prompt
    return json.dumps(row, ensure_ascii=False)


def generate_prompt_id(subset: str, index: int) -> str:
    return f"{subset}_{index:04d}"


def load_longbench_records(
    data_dir: str | Path,
    subsets: Iterable[str] | None = None,
) -> List[PromptRecord]:
    """Load LongBench prompt records from local JSONL files."""
    root = Path(data_dir)
    subset_names = list(subsets or LONG_BENCH_ENGLISH_SUBSETS.keys())
    records: List[PromptRecord] = []

    for subset in subset_names:
        path = root / f"{subset}.jsonl"
        if not path.exists():
            continue

        category = LONG_BENCH_ENGLISH_SUBSETS.get(subset, "other")
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                row = json.loads(line)
                prompt_id = generate_prompt_id(subset, idx)
                metadata = {
                    "language": row.get("language"),
                    "length": row.get("length"),
                    "all_classes": row.get("all_classes"),
                }
                records.append(
                    PromptRecord(
                        prompt_id=prompt_id,
                        subset=subset,
                        category=category,
                        prompt=build_prompt_from_row(row, subset=subset),
                        answers=_as_answer_list(
                            row.get("answers")
                            or row.get("answer")
                            or row.get("output")
                        ),
                        metadata={k: v for k, v in metadata.items() if v is not None},
                    )
                )
    return records


def _take_evenly_by_subset(
    pools: Dict[str, List[PromptRecord]],
    *,
    total: int,
    subset_names: Sequence[str],
) -> List[str]:
    """Take up to total prompt IDs from subset pools as evenly as possible."""
    if total <= 0:
        return []

    taken: List[str] = []
    remaining = int(total)
    active = [name for name in subset_names if pools.get(name)]

    while remaining > 0 and active:
        base = max(1, remaining // len(active))
        progressed = False
        for subset in list(active):
            pool = pools[subset]
            n_take = min(base, remaining, len(pool))
            if n_take <= 0:
                active.remove(subset)
                continue
            taken.extend(record.prompt_id for record in pool[:n_take])
            del pool[:n_take]
            remaining -= n_take
            progressed = True
            if not pool:
                active.remove(subset)
            if remaining <= 0:
                break
        if not progressed:
            break

    return taken


def _count_by_subset(records: Sequence[PromptRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.subset] = counts.get(record.subset, 0) + 1
    return dict(sorted(counts.items()))


def generate_stratified_split(
    records: Sequence[PromptRecord],
    n_train: int = 300,
    seed: int = 42,
    n_dev: int = 0,
    n_test: int = 0,
) -> Dict[str, Any]:
    """Generate a subset-stratified split file payload.

    Backwards-compatible two-way behavior is preserved when n_dev and n_test
    are both zero: the payload contains ``train`` and ``eval`` where ``eval``
    is every non-training prompt. When either three-way count is requested,
    the payload contains ``train``, ``dev``, ``test``, and ``eval``. In that
    case, ``eval`` aliases ``test`` so older commands evaluate the locked test
    split by default.
    """
    if not records:
        raise ValueError("No LongBench records found.")

    by_subset: Dict[str, List[PromptRecord]] = {}
    for record in records:
        by_subset.setdefault(record.subset, []).append(record)

    rng = random.Random(seed)
    for subset_records in by_subset.values():
        rng.shuffle(subset_records)

    subset_names = sorted(by_subset)
    pools = {subset: list(by_subset[subset]) for subset in subset_names}

    train_ids = _take_evenly_by_subset(pools, total=n_train, subset_names=subset_names)
    dev_ids = _take_evenly_by_subset(pools, total=n_dev, subset_names=subset_names)
    test_ids = _take_evenly_by_subset(pools, total=n_test, subset_names=subset_names)
    remaining_ids = [record.prompt_id for subset in subset_names for record in pools[subset]]

    train_set = set(train_ids)
    if len(train_set) < len(train_ids):
        raise RuntimeError("Duplicate prompt_ids detected while building split.")
    all_assigned = train_ids + dev_ids + test_ids + remaining_ids
    if len(set(all_assigned)) != len(all_assigned):
        raise RuntimeError("Duplicate prompt_ids detected across split fields.")

    by_id = {record.prompt_id: record for record in records}
    payload: Dict[str, Any] = {"train": sorted(train_ids)}
    if n_dev > 0 or n_test > 0:
        payload["dev"] = sorted(dev_ids)
        payload["test"] = sorted(test_ids)
        payload["eval"] = sorted(test_ids)
        payload["unused"] = sorted(remaining_ids)
    else:
        payload["eval"] = sorted(dev_ids + test_ids + remaining_ids)

    payload["metadata"] = {
        "n_train": len(train_ids),
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
        "n_eval": len(payload["eval"]),
        "n_unused": len(remaining_ids),
        "seed": seed,
        "subsets": subset_names,
        "split_schema": "longbench_stratified_v2" if (n_dev > 0 or n_test > 0) else "longbench_stratified_v1",
        "requested": {
            "n_train": int(n_train),
            "n_dev": int(n_dev),
            "n_test": int(n_test),
        },
        "distributions": {
            field: _count_by_subset([by_id[prompt_id] for prompt_id in payload.get(field, [])])
            for field in ("train", "dev", "test", "eval", "unused")
            if field in payload
        },
    }
    return payload


def select_split_records(
    records: Sequence[PromptRecord],
    split_payload: Dict[str, Any],
    field: str = "eval",
) -> List[PromptRecord]:
    """Return records listed in a split field, preserving split order."""
    if field not in split_payload:
        available = ", ".join(sorted(k for k, v in split_payload.items() if isinstance(v, list)))
        raise KeyError(f"Split field '{field}' not found. Available list fields: {available}")

    prompt_ids = [str(item) for item in split_payload[field]]
    by_id = {record.prompt_id: record for record in records}
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Split field '{field}' references missing prompt IDs: {preview}")
    return [by_id[prompt_id] for prompt_id in prompt_ids]


def split_prompt_ids(split_payload: Dict[str, Any], field: str) -> List[str]:
    if field not in split_payload:
        available = ", ".join(sorted(k for k, v in split_payload.items() if isinstance(v, list)))
        raise KeyError(f"Split field '{field}' not found. Available list fields: {available}")
    values = split_payload[field]
    if not isinstance(values, list):
        raise ValueError(f"Split field '{field}' is not a list")
    return [str(item) for item in values]


def load_split_file(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
