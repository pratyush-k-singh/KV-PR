"""InfiniteBench loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .longbench import PromptRecord, _as_answer_list


INFINITEBENCH_ENGLISH_SUBSETS: Dict[str, str] = {
    "code_debug": "code",
    "code_run": "code",
    "kv_retrieval": "synthetic",
    "longbook_choice_eng": "1doc_qa",
    "longbook_qa_eng": "1doc_qa",
    "longbook_sum_eng": "summarization",
    "longdialogue_qa_eng": "few_shot",
    "math_calc": "synthetic",
    "math_find": "synthetic",
    "number_string": "synthetic",
    "passkey": "synthetic",
}

INFINITEBENCH_MAX_NEW_TOKENS: Dict[str, int] = {
    "code_debug": 64,
    "code_run": 64,
    "kv_retrieval": 64,
    "longbook_choice_eng": 64,
    "longbook_qa_eng": 128,
    "longbook_sum_eng": 512,
    "longdialogue_qa_eng": 64,
    "math_calc": 2048,
    "math_find": 32,
    "number_string": 32,
    "passkey": 32,
}


def build_prompt_from_row(row: Dict[str, Any]) -> str:
    context = str(row.get("context", "")).strip()
    question = str(row.get("input", "")).strip()
    parts = []
    if context:
        parts.append(context)
    options = row.get("options") or []
    if options:
        option_text = "\n".join(f"- {option}" for option in options)
        parts.append(f"Options:\n{option_text}")
    if question:
        parts.append(f"Question:\n{question}")
    if options:
        parts.append("Answer with the exact option text.")
    return "\n\n".join(parts) if parts else json.dumps(row, ensure_ascii=False)


def _answers_from_row(row: Dict[str, Any], subset: str) -> List[str]:
    answer = row.get("answer")
    if subset == "math_calc" and isinstance(answer, Sequence) and not isinstance(answer, str):
        return [" ".join(str(item) for item in answer)]
    return _as_answer_list(answer)


def load_infinitebench_records(
    data_dir: str | Path,
    subsets: Iterable[str] | None = None,
) -> List[PromptRecord]:
    root = Path(data_dir)
    subset_names = list(subsets or INFINITEBENCH_ENGLISH_SUBSETS.keys())
    records: List[PromptRecord] = []

    for subset in subset_names:
        path = root / f"{subset}.jsonl"
        if not path.exists():
            continue
        category = INFINITEBENCH_ENGLISH_SUBSETS.get(subset, "other")
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                row = json.loads(line)
                records.append(
                    PromptRecord(
                        prompt_id=f"ib_{subset}_{idx:04d}",
                        subset=subset,
                        category=category,
                        prompt=build_prompt_from_row(row),
                        answers=_answers_from_row(row, subset),
                    )
                )
    return records
