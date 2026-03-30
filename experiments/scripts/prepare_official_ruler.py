"""Prepare official NVIDIA RULER tasks for the local custom evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml


REQUIRED_PACKAGES = {
    "wonderwords": "wonderwords",
    "nltk": "nltk",
    "tenacity": "tenacity",
    "yaml": "pyyaml",
}

TASK_TO_SUBSET = {
    "niah_single_1": "ruler_official_niah_single_1",
    "niah_single_2": "ruler_official_niah_single_2",
    "niah_single_3": "ruler_official_niah_single_3",
    "niah_multikey_1": "ruler_official_niah_multikey_1",
    "niah_multikey_2": "ruler_official_niah_multikey_2",
    "niah_multikey_3": "ruler_official_niah_multikey_3",
    "niah_multivalue": "ruler_official_niah_multivalue",
    "niah_multiquery": "ruler_official_niah_multiquery",
    "vt": "ruler_official_vt",
    "cwe": "ruler_official_cwe",
    "fwe": "ruler_official_fwe",
}

RULER_REQUIRED_FILES = (
    "scripts/synthetic.yaml",
    "scripts/data/template.py",
    "scripts/data/tokenizer.py",
    "scripts/data/manifest_utils.py",
    "scripts/data/synthetic/constants.py",
    "scripts/data/synthetic/niah.py",
    "scripts/data/synthetic/variable_tracking.py",
    "scripts/data/synthetic/common_words_extraction.py",
    "scripts/data/synthetic/freq_words_extraction.py",
    "scripts/data/synthetic/json/english_words.json",
)

# Data files that the RULER clone does NOT ship — must be downloaded
# post-clone (one-time, login-node, needs internet). The
# ``download_paulgraham_essay.py`` helper lives next to the target;
# without ``PaulGrahamEssays.json`` the niah essay-haystack tasks die
# mid-job with a FileNotFoundError. The QA datasets aren't required for
# the niah/vt/cwe/fwe smoke task set so they are not preflight-checked.
RULER_DATA_FILES = (
    "scripts/data/synthetic/json/PaulGrahamEssays.json",
)


def _parse_name_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _ensure_packages(*, install: bool) -> None:
    missing = [package for module, package in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing and not install:
        raise RuntimeError(
            "Missing RULER dependencies: "
            + ", ".join(sorted(missing))
            + ". Re-run with --install-deps or install them in the active environment."
        )
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *sorted(missing)], check=True)


def _ensure_ruler_repo(root: Path, *, repo: str, branch: str) -> None:
    if all((root / rel_path).exists() for rel_path in RULER_REQUIRED_FILES):
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, repo, str(root)], check=True)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task_config(ruler_root: Path, task: str, *, model_template_type: str) -> dict:
    data_dir = ruler_root / "scripts" / "data"
    constants = _load_module(data_dir / "synthetic" / "constants.py", "ruler_synthetic_constants")
    template_module = _load_module(data_dir / "template.py", "ruler_templates")
    with (ruler_root / "scripts" / "synthetic.yaml").open("r", encoding="utf-8") as handle:
        customized = yaml.safe_load(handle)
    if task not in customized:
        raise ValueError(f"{task} is not an official RULER synthetic task in scripts/synthetic.yaml")
    config = dict(customized[task])
    base = dict(constants.TASKS[config["task"]])
    merged = {**base, **config}
    task_template = merged["template"]
    answer_prefix = merged.get("answer_prefix", "")
    if model_template_type not in template_module.Templates:
        raise ValueError(
            f"Unknown RULER model template {model_template_type!r}. "
            f"Available templates: {sorted(template_module.Templates)}"
        )
    model_template = template_module.Templates[model_template_type]
    merged["template"] = model_template.format(task_template=task_template) + answer_prefix
    return merged


def _run_prepare(args, *, task: str, raw_dir: Path) -> Path:
    config = _task_config(args.ruler_root, task, model_template_type=args.model_template_type)
    task_script = args.ruler_root / "scripts" / "data" / "synthetic" / f"{config['task']}.py"
    cmd = [
        sys.executable,
        str(task_script),
        "--save_dir",
        str(raw_dir),
        "--save_name",
        task,
        "--subset",
        args.subset,
        "--tokenizer_path",
        args.tokenizer_path,
        "--tokenizer_type",
        args.tokenizer_type,
        "--max_seq_length",
        str(args.max_seq_length),
        "--tokens_to_generate",
        str(config["tokens_to_generate"]),
        "--num_samples",
        str(args.samples_per_task),
        "--random_seed",
        str(args.seed),
        "--template",
        config["template"],
    ]
    for key, value in config["args"].items():
        cmd.extend([f"--{key}", str(value)])
    if args.remove_newline_tab:
        cmd.append("--remove_newline_tab")
    subprocess.run(cmd, check=True)
    return raw_dir / task / f"{args.subset}.jsonl"


def _convert_task(source: Path, *, task: str, max_seq_length: int, seed: int) -> list[dict]:
    subset = TASK_TO_SUBSET.get(task, f"ruler_official_{task}")
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for local_idx, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            index = int(payload.get("index", local_idx))
            rows.append(
                {
                    "prompt_id": f"{subset}_l{max_seq_length}_s{seed}_{index:04d}",
                    "subset": subset,
                    "category": "official_ruler",
                    "prompt": payload["input"],
                    "answers": payload.get("outputs", []),
                    "metadata": {
                        "official_ruler_task": task,
                        "source_index": index,
                        "length": payload.get("length"),
                        "length_w_model_temp": payload.get("length_w_model_temp"),
                        "answer_prefix": payload.get("answer_prefix", ""),
                    },
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, default=Path("third_party/RULER"))
    parser.add_argument("--repo", default="https://github.com/NVIDIA/RULER.git")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--out-dir", type=Path, default=Path("data/ruler_official_swapclf_v2"))
    parser.add_argument(
        "--tasks",
        default=(
            "niah_single_1,niah_single_2,niah_single_3,"
            "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
            "niah_multivalue,niah_multiquery,vt,cwe,fwe"
        ),
    )
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--tokenizer-path", default="models/llama3.1-8b-instruct")
    parser.add_argument("--tokenizer-type", default="hf")
    parser.add_argument("--model-template-type", default="meta-llama3")
    parser.add_argument("--remove-newline-tab", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.ruler_root = args.ruler_root.resolve()
    for task in _parse_name_list(args.tasks):
        if task not in TASK_TO_SUBSET:
            raise ValueError(
                f"{task} is not in the supported official RULER subset for this smoke. "
                f"Supported: {sorted(TASK_TO_SUBSET)}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _ensure_packages(install=args.install_deps)
    _ensure_ruler_repo(args.ruler_root, repo=args.repo, branch=args.branch)

    missing_data = [
        rel for rel in RULER_DATA_FILES
        if not (args.ruler_root / rel).exists()
    ]
    if missing_data:
        joined = "\n  ".join(missing_data)
        raise FileNotFoundError(
            f"RULER haystack data not present:\n  {joined}\n\n"
            f"NVIDIA RULER does not ship these — download once on a "
            f"LOGIN node (compute nodes lack internet):\n\n"
            f"  pip install beautifulsoup4 requests html2text tqdm && "
            f"cd {args.ruler_root}/scripts/data/synthetic/json/ && "
            f"python download_paulgraham_essay.py\n\n"
            f"Then re-submit this sbatch."
        )

    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab/english/")
    except (LookupError, ImportError):
        raise FileNotFoundError(
            "NLTK punkt_tab tokenizer not present. RULER's NIAH "
            "generation uses sent_tokenize; download once on a LOGIN "
            "node (compute nodes lack internet):\n\n"
            "  python -m nltk.downloader punkt_tab\n\n"
            "Then re-submit this sbatch."
        )

    tasks = _parse_name_list(args.tasks)
    rows = []
    for task in tasks:
        source = _run_prepare(args, task=task, raw_dir=raw_dir)
        rows.extend(_convert_task(source, task=task, max_seq_length=args.max_seq_length, seed=args.seed))

    out_file = args.out_dir / "official_ruler.jsonl"
    with out_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source": "NVIDIA/RULER",
        "repo": args.repo,
        "branch": args.branch,
        "tasks": tasks,
        "samples_per_task": args.samples_per_task,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "tokenizer_path": args.tokenizer_path,
        "tokenizer_type": args.tokenizer_type,
        "model_template_type": args.model_template_type,
        "rows": len(rows),
        "output": str(out_file),
    }
    (args.out_dir / "official_ruler_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} official RULER records to {out_file}")


if __name__ == "__main__":
    main()
