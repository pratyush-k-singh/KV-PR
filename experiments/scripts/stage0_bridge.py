#!/usr/bin/env python3
"""Stage 0 bridge experiment: formal cost vs teacher-forced fidelity.

For each prompt in a LongBench split:
  1. Encode the prompt (smoke mode: prompt only) or prompt + reference answer
     (teacher_forced mode: the answer positions are the decode-shaped queries).
  2. Run prefill with Q/K capture + past_key_values; extract per-layer (Q, K, V).
  3. Compute the SnapKV-style frozen token priority over candidate prompt keys.
  4. For each budget vector in the predeclared spread:
       - per-group hot sets from the shared priority and the budget;
       - per-layer cold_attention_demand (the formal cost c_t);
       - per-layer attention_output_deviation (the per-layer fidelity rung);
       - optionally (--include-logit-kl) a hot-only forward and logit-KL;
       - aggregate over layers.
  5. Report Spearman rho(cost, fidelity) -- and rho(cost, logit_kl) when
     enabled -- plus per-category bucket signs (Gate 0).

Outputs to ``--out``:
  ``eval_config.json``       run config snapshot
  ``records.jsonl``          one record per (prompt, budget)
  ``gate0_report.json``      overall + per-bucket rho, pass/fail vs --gate0-threshold
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch

from src.benchmarks.longbench import (
    LONG_BENCH_ENGLISH_SUBSETS,
    PromptRecord,
    load_longbench_records,
    load_split_file,
    select_split_records,
)
from src.dart_pagedkv.budget_spread import predeclared_spread
from src.dart_pagedkv.fidelity import attention_output_deviation
from src.dart_pagedkv.gate0 import gate0_report, spearman_rho
from src.dart_pagedkv.layer_groups import make_layer_groups
from src.dart_pagedkv.logit_kl import logit_kl_for_budget
from src.dart_pagedkv.priorities import snapkv_priority
from src.dart_pagedkv.service_cost import cold_attention_demand
from src.dart_pagedkv.trace import capture_prompt_trace
from src.dart_pagedkv.two_tier import group_hot_sets
from src.trace.collect import (
    LayerQKCapture,
    load_model_and_tokenizer,
    recompute_attention_matrix,
    resolve_model_input_device,
    tokenize_prompt,
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stage0-bridge",
        description="Stage 0 bridge: cost vs teacher-forced fidelity over a budget spread.",
    )
    parser.add_argument("--model", required=True, help="HF model path or id.")
    parser.add_argument("--data-dir", required=True, help="Directory of LongBench JSONL files.")
    parser.add_argument("--split-file", default=None, help="Optional split JSON.")
    parser.add_argument("--split-field", default="eval", help="Split field name.")
    parser.add_argument("--subsets", nargs="*", default=None, help="LongBench subsets to load.")
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--chat-template", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--num-groups", type=int, default=4, help="G layer-groups.")
    parser.add_argument("--total-budget", type=int, default=256,
                        help="Total hot-tier budget B (entries of each budget vector).")
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--n-recent", type=int, default=32)
    parser.add_argument("--budget-floor", type=int, default=None,
                        help="Minimum budget every layer-group keeps (its protected "
                             "sink+recent positions). Default = n_sink + n_recent. "
                             "Keeps every group's sink hot, so no group is fully "
                             "starved and the logit-KL forward has no all-masked "
                             "attention rows (which would soft-max to NaN).")
    parser.add_argument("--obs-window", type=int, default=16)
    parser.add_argument("--query-mode", choices=["smoke", "teacher_forced"],
                        default="teacher_forced",
                        help="smoke: last-N prompt positions as decode queries; "
                             "teacher_forced: prompt + reference answer, the answer positions "
                             "are decode queries (the default, required for any Gate-0 claim).")
    parser.add_argument("--n-decode-queries", type=int, default=32,
                        help="(smoke mode) trailing positions used as decode queries.")
    parser.add_argument("--n-layers-to-track", type=int, default=None,
                        help="Capture Q/K for only the last N layers (memory-tight envs). "
                             "Default = all model layers.")
    parser.add_argument("--gate0-threshold", type=float, default=0.45)
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads-per-batch", type=int, default=8,
                        help="Head batch size for the attention recomputation.")
    parser.add_argument("--include-logit-kl", action="store_true",
                        help="Also compute the logit-KL fidelity rung: one extra "
                             "forward per budget vector with per-layer cold-key "
                             "masking. Forces an eager/sdpa attention model.")
    parser.add_argument("--logit-kl-attn-impl", default="eager",
                        choices=["eager", "sdpa"],
                        help="Attention implementation for the logit-KL forwards.")
    return parser.parse_args(argv)


def _select_prompts(args: argparse.Namespace) -> list[PromptRecord]:
    subsets = args.subsets or list(LONG_BENCH_ENGLISH_SUBSETS.keys())
    records = load_longbench_records(args.data_dir, subsets)
    if not records:
        raise FileNotFoundError(
            f"No LongBench records found under {args.data_dir} for subsets {subsets}."
        )
    if args.split_file:
        split = load_split_file(args.split_file)
        records = select_split_records(records, split, field=args.split_field)
    if len(records) > args.max_prompts:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(records), size=args.max_prompts, replace=False).tolist())
        records = [records[i] for i in idx]
    return records


def _encode_with_optional_reference(
    tokenizer,
    prompt_text: str,
    *,
    max_length: int,
    chat_template: str,
    device: str,
    reference: str | None,
) -> Tuple[dict, int, int]:
    """Return (inputs, prompt_token_count, total_token_count).

    The prompt itself is tokenized with chat template / middle truncation. When
    a reference is supplied (teacher_forced mode), its tokens are appended raw
    -- this is faithful to what the model would observe under teacher forcing
    during the answer span.
    """
    prompt_inputs = tokenize_prompt(
        tokenizer,
        prompt_text,
        device=device,
        max_length=max_length,
        chat_template=chat_template,
    )
    prompt_n = int(prompt_inputs["input_ids"].shape[1])
    if not reference:
        return prompt_inputs, prompt_n, prompt_n

    ref_encoded = tokenizer(reference, return_tensors="pt", add_special_tokens=False)
    ref_ids = ref_encoded["input_ids"].to(prompt_inputs["input_ids"].device)
    if ref_ids.shape[1] == 0:
        return prompt_inputs, prompt_n, prompt_n
    combined_ids = torch.cat([prompt_inputs["input_ids"], ref_ids], dim=1)
    base_mask = prompt_inputs.get("attention_mask")
    if base_mask is None:
        base_mask = torch.ones_like(prompt_inputs["input_ids"])
    combined_mask = torch.cat([base_mask, torch.ones_like(ref_ids)], dim=1)
    return (
        {"input_ids": combined_ids, "attention_mask": combined_mask},
        prompt_n,
        int(combined_ids.shape[1]),
    )


def _layer_qkv_to_capture(layer) -> LayerQKCapture:
    """Re-add a batch dim so recompute_attention_matrix accepts our trace tensors."""
    return LayerQKCapture(
        query=layer.query.unsqueeze(0),
        key=layer.key.unsqueeze(0),
        scaling=layer.scaling,
        num_key_value_groups=layer.num_key_value_groups,
    )


def _layer_to_group_map(num_layers: int, num_groups: int, n_track: int) -> Tuple[dict[int, int], list[int]]:
    """Map each tracked model-layer-index to its group, contiguous and near-even."""
    tracked = list(range(num_layers - n_track, num_layers))
    groups = make_layer_groups(n_track, num_groups)
    mapping = {
        tracked[local_idx]: group_idx
        for group_idx, local_idxs in enumerate(groups)
        for local_idx in local_idxs
    }
    return mapping, tracked


def _decode_window(
    args: argparse.Namespace,
    prompt_n: int,
    total_n: int,
    has_reference: bool,
) -> Tuple[int, int, int]:
    """Return (query_start, query_end, key_split_boundary).

    The key-split boundary is the last index of "candidate" keys (the prompt
    keys subject to the hot/cold split). Anything at or after it is always
    hot (the decode-side queries / answer continuation themselves can't be
    dropped without breaking the teacher-forced replay).
    """
    if args.query_mode == "teacher_forced" and has_reference and total_n > prompt_n:
        return prompt_n, total_n, prompt_n
    n_decode = min(int(args.n_decode_queries), total_n - 1)
    if n_decode <= 0:
        raise ValueError(f"Prompt too short for smoke window (total_n={total_n}).")
    return total_n - n_decode, total_n, total_n - n_decode


def _compute_attention_matrix(trace_layers, device: str, heads_per_batch: int) -> np.ndarray:
    captures = {idx: _layer_qkv_to_capture(layer) for idx, layer in trace_layers.items()}
    return recompute_attention_matrix(captures, device=device, heads_per_batch=heads_per_batch)


def _within_prompt_mean_rho(records: list[dict], cost_key: str, fid_key: str) -> float:
    """Mean per-prompt Spearman rho between two record fields across budgets.

    This is the *within-prompt* rank correlation -- the signal an online cost
    proxy must actually carry: given a single prompt, does cost order the 9
    budget vectors the same way fidelity does? Cross-prompt correlation (the
    overall Spearman) can be high while within-prompt is at chance.
    """
    from collections import defaultdict

    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        if cost_key in r and fid_key in r:
            groups[r["prompt_id"]].append((float(r[cost_key]), float(r[fid_key])))
    rhos = [
        spearman_rho([p[0] for p in pairs], [p[1] for p in pairs])
        for pairs in groups.values()
        if len(pairs) >= 2
    ]
    return float(np.mean(rhos)) if rhos else 0.0


def _compute_layer_metrics(
    trace_layers,
    layer_to_group: dict[int, int],
    hot_per_group: list[list[int]],
    always_hot_tail: Iterable[int],
    query_start: int,
    query_end: int,
) -> Tuple[list[float], list[float]]:
    tail = list(always_hot_tail)
    costs: list[float] = []
    deviations: list[float] = []
    for layer_idx in sorted(trace_layers.keys()):
        group_idx = layer_to_group.get(layer_idx)
        if group_idx is None:
            continue
        layer = trace_layers[layer_idx]
        all_hot = list(hot_per_group[group_idx]) + tail
        q = layer.query[:, query_start:query_end, :].contiguous()
        cost = cold_attention_demand(
            q,
            layer.key,
            all_hot,
            scaling=layer.scaling,
            num_key_value_groups=layer.num_key_value_groups,
            query_offset=query_start,
        )
        deviation = attention_output_deviation(
            q,
            layer.key,
            layer.value,
            all_hot,
            scaling=layer.scaling,
            num_key_value_groups=layer.num_key_value_groups,
            query_offset=query_start,
        )
        costs.append(cost)
        deviations.append(deviation)
    return costs, deviations


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    print(f"[stage0-bridge] loading model: {args.model}", flush=True)
    attn_impl = args.logit_kl_attn_impl if args.include_logit_kl else None
    model, tokenizer = load_model_and_tokenizer(
        args.model, device=args.device, attn_implementation=attn_impl
    )
    input_device = resolve_model_input_device(model, args.device)
    model_dtype = next(model.parameters()).dtype

    print(f"[stage0-bridge] loading prompts from {args.data_dir}", flush=True)
    prompts = _select_prompts(args)
    print(f"[stage0-bridge] {len(prompts)} prompts", flush=True)

    num_layers = int(model.config.num_hidden_layers)
    n_track = min(args.n_layers_to_track or num_layers, num_layers)
    if n_track < args.num_groups:
        raise ValueError(
            f"--n-layers-to-track ({n_track}) must be >= --num-groups ({args.num_groups})."
        )
    layer_to_group, tracked = _layer_to_group_map(num_layers, args.num_groups, n_track)

    budget_floor = (
        args.budget_floor
        if args.budget_floor is not None
        else args.n_sink + args.n_recent
    )
    if args.num_groups * budget_floor > args.total_budget:
        raise ValueError(
            f"--total-budget ({args.total_budget}) cannot give all "
            f"{args.num_groups} groups their budget floor ({budget_floor}); "
            f"raise --total-budget to >= {args.num_groups * budget_floor} or "
            f"lower --n-sink/--n-recent/--budget-floor."
        )
    floors = [budget_floor] * args.num_groups
    spread = predeclared_spread(args.num_groups, args.total_budget, floors)
    print(
        f"[stage0-bridge] G={args.num_groups} total={args.total_budget} "
        f"floor={budget_floor} |spread|={len(spread)} "
        f"tracked_layers={n_track}/{num_layers}",
        flush=True,
    )

    records_out: list[dict] = []
    for prompt_idx, record in enumerate(prompts):
        prompt_start = time.perf_counter()
        print(
            f"[stage0-bridge] prompt {prompt_idx + 1}/{len(prompts)}  "
            f"id={record.prompt_id}  subset={record.subset}",
            flush=True,
        )
        reference = (
            record.answers[0]
            if (args.query_mode == "teacher_forced" and record.answers)
            else None
        )
        encoded, prompt_n, total_n = _encode_with_optional_reference(
            tokenizer,
            record.prompt,
            max_length=args.max_seq_len,
            chat_template=args.chat_template,
            device=str(input_device),
            reference=reference,
        )
        try:
            query_start, query_end, key_boundary = _decode_window(
                args, prompt_n, total_n, has_reference=reference is not None
            )
        except ValueError as exc:
            print(f"[stage0-bridge] skip {record.prompt_id}: {exc}", flush=True)
            continue
        print(
            f"[stage0-bridge]   seq_len={total_n}  prompt_n={prompt_n}  "
            f"query_window={query_end - query_start}",
            flush=True,
        )

        capture_start = time.perf_counter()
        with torch.inference_mode():
            trace = capture_prompt_trace(model, encoded, n_layers_to_track=n_track)
        print(
            f"[stage0-bridge]   trace captured in {time.perf_counter() - capture_start:.1f}s",
            flush=True,
        )

        attn_matrix = _compute_attention_matrix(
            trace.layers, device=args.device, heads_per_batch=args.heads_per_batch
        )
        full_priority = snapkv_priority(attn_matrix, obs_window=args.obs_window)
        prompt_priority = full_priority[:key_boundary].tolist()
        group_priorities = [prompt_priority for _ in range(args.num_groups)]

        always_hot_tail = list(range(key_boundary, total_n))
        before_count = len(records_out)

        for budget_idx, budget in enumerate(spread):
            hot_per_group = group_hot_sets(
                budget=budget,
                group_priorities=group_priorities,
                seq_len=key_boundary,
                n_sink=args.n_sink,
                n_recent=args.n_recent,
            )
            layer_costs, layer_devs = _compute_layer_metrics(
                trace.layers,
                layer_to_group,
                hot_per_group,
                always_hot_tail,
                query_start,
                query_end,
            )
            if not layer_costs:
                continue
            entry = {
                "prompt_id": record.prompt_id,
                "subset": record.subset,
                "category": record.category,
                "prompt_idx": prompt_idx,
                "budget_idx": budget_idx,
                "budget": list(int(x) for x in budget),
                "total_budget": int(args.total_budget),
                "cost": float(np.mean(layer_costs)),
                "cost_max": float(np.max(layer_costs)),
                "fidelity": float(np.mean(layer_devs)),
                "fidelity_max": float(np.max(layer_devs)),
                "layer_costs": [float(x) for x in layer_costs],
                "layer_deviations": [float(x) for x in layer_devs],
                "n_layers": int(len(layer_costs)),
                "seq_len": int(total_n),
                "prompt_n": int(prompt_n),
                "key_boundary": int(key_boundary),
                "query_window": [int(query_start), int(query_end)],
            }
            if args.include_logit_kl:
                entry["logit_kl"] = logit_kl_for_budget(
                    model,
                    encoded,
                    trace.prefill_logits,
                    hot_per_group=hot_per_group,
                    layer_to_group=layer_to_group,
                    total_seq_len=total_n,
                    key_boundary=key_boundary,
                    query_start=query_start,
                    query_end=query_end,
                    dtype=model_dtype,
                    device=input_device,
                )
            records_out.append(entry)

        new_prompt_records = records_out[before_count:]
        if new_prompt_records:
            mean_cost = float(np.mean([r["cost"] for r in new_prompt_records]))
            mean_fid = float(np.mean([r["fidelity"] for r in new_prompt_records]))
            elapsed = time.perf_counter() - prompt_start
            extra = ""
            if args.include_logit_kl:
                kls = [r["logit_kl"] for r in new_prompt_records if "logit_kl" in r]
                if kls:
                    extra = f"  mean_logit_kl={float(np.mean(kls)):.4f}"
            print(
                f"[stage0-bridge]   done in {elapsed:.1f}s  "
                f"mean_cost={mean_cost:.4f}  mean_fidelity={mean_fid:.4f}{extra}",
                flush=True,
            )

        # Free per-prompt trace before the next prompt to keep peak memory bounded.
        del trace, attn_matrix
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in records_out:
            fh.write(json.dumps(rec) + "\n")

    costs = [r["cost"] for r in records_out]
    fidelity = [r["fidelity"] for r in records_out]
    costs_max = [r["cost_max"] for r in records_out]
    fidelity_max = [r["fidelity_max"] for r in records_out]
    categories = [r["category"] for r in records_out]

    def _build(x, y, cost_key, fid_key):
        sub = gate0_report(x, y, buckets=categories, threshold=args.gate0_threshold)
        sub["within_prompt_mean_rho"] = _within_prompt_mean_rho(
            records_out, cost_key, fid_key
        )
        return sub

    # Primary: mean-aggregated cost vs mean-aggregated output-deviation.
    report = _build(costs, fidelity, "cost", "fidelity")
    # Max-aggregation variant -- bottleneck-sensitive cost/fidelity, motivated
    # by the v2-floored finding that mean-aggregation smooths the bottleneck
    # while logit-KL is bottleneck-sensitive.
    report["max_aggregation"] = _build(costs_max, fidelity_max, "cost_max", "fidelity_max")

    if args.include_logit_kl:
        logit_kls = [r["logit_kl"] for r in records_out if "logit_kl" in r]
        if logit_kls and len(logit_kls) == len(costs):
            report["logit_kl"] = _build(costs, logit_kls, "cost", "logit_kl")
            report["max_aggregation"]["logit_kl"] = _build(
                costs_max, logit_kls, "cost_max", "logit_kl"
            )

    with (out_dir / "gate0_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("", flush=True)
    print(
        f"[stage0-bridge] Gate-0 summary  (n={report['n']}, "
        f"threshold={args.gate0_threshold})",
        flush=True,
    )
    header = f"  {'pairing':<28}  {'overall_rho':>11}  {'within_prompt':>13}  {'+/-/0':>9}  pass"
    print(header, flush=True)

    pairings: list[tuple[str, dict]] = [
        ("mean-cost vs mean-fidelity ", report),
        ("max-cost  vs max-fidelity  ", report["max_aggregation"]),
    ]
    if "logit_kl" in report:
        pairings.append(("mean-cost vs logit-KL      ", report["logit_kl"]))
    if "logit_kl" in report.get("max_aggregation", {}):
        pairings.append(("max-cost  vs logit-KL      ", report["max_aggregation"]["logit_kl"]))

    for label, sub in pairings:
        wp = sub.get("within_prompt_mean_rho", float("nan"))
        bs = f"{sub['positive_buckets']}/{sub['negative_buckets']}/{sub['zero_buckets']}"
        passed = "yes" if sub["overall_pass"] else "no "
        print(
            f"  {label}  {sub['overall_rho']:>+11.4f}  {wp:>+13.4f}  {bs:>9}  {passed}",
            flush=True,
        )

    print(f"[stage0-bridge] records: {records_path}", flush=True)
    print(f"[stage0-bridge] report:  {out_dir / 'gate0_report.json'}", flush=True)


if __name__ == "__main__":
    main()
