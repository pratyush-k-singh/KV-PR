"""E -- end-to-end reference-fidelity accuracy under masked KV.

The audit's `longcontext_decode_probe` measures per-step KL under teacher-
forced masked decode: it always feeds the *gold-continuation* token at
each step and records how the masked logit differs from the full-KV
logit. That captures the divergence between the two distributions but
not whether the divergence flips the next-token argmax.

This producer answers the user-facing version: under each priority's
masked KV cache, what fraction of autoregressively-generated tokens
match the full-KV reference's autoregressive decode? Argmax of each
masked step's logits feeds back as the next input; we compare token
sequences against the reference's argmax decode token-by-token.

For each prompt we:
  1. Prefill once.
  2. Run `full_kv_reference(k_gen)` to obtain the gold token sequence
     (greedy autoregressive decode under full KV).
  3. For each requested priority:
     a. Compute the within-group priority vector via the same
        `select_group_priorities` the KL audit uses.
     b. Build hot_per_group at the requested uniform allocation (r and
        ratio match the audit's canonical cell).
     c. Run `masked_decode_autoregressive(hot_per_group, k_gen)` to get
        the priority's token sequence.
     d. Score: first-divergence step + prefix-match rate.

Output (per prompt, per priority): generated tokens, divergence step,
prefix-match-rate-vs-reference at each k in {2, 4, 8, 16}, mean
matching-prefix length.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.dart_pagedkv.flex_decode import FlexCachedDecodeSession
from src.dart_pagedkv.layernorm_intervention import install_post_norm_hooks
from src.dart_pagedkv.ruler_manifest import load_manifest_records
from src.dart_pagedkv.trace import capture_prompt_trace
from src.dart_pagedkv.two_tier import group_hot_sets

from experiments.scripts.longcontext_decode_probe import (
    PRIORITY_MODES,
    attention_backends_for_model,
    protected_budget_floors,
    select_group_priorities,
)
from experiments.scripts.stage0_bridge import _layer_to_group_map
from src.trace.collect import (
    load_model_and_tokenizer,
    resolve_model_input_device,
    tokenize_prompt,
)


def _prefix_match_length(a: list[int], b: list[int]) -> int:
    """First k such that a[k] != b[k]; len(a) if identical up to min length."""
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return n


def _scores(reference: list[int], generated: list[int]) -> dict[str, Any]:
    n = min(len(reference), len(generated))
    div = _prefix_match_length(reference, generated)
    out = {
        "first_divergence_step": int(div),
        "prefix_match_rate_full": float(div) / float(n) if n > 0 else 0.0,
        "n_steps_compared": int(n),
    }
    for k in (2, 4, 8, 16):
        if n >= k:
            matches = sum(1 for i in range(k) if reference[i] == generated[i])
            out[f"prefix_match_rate_k{k}"] = matches / float(k)
    return out


def _run_prompt(
    record, *, model, tokenizer, device, num_layers: int, num_groups: int,
    obs_window: int, k_gen: int, n_sink: int, n_recent: int,
    ratio: float, priorities: list[str], priority_seed: int,
    masked_attn_implementation: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prompt_id": record.prompt_id,
        "task": record.task,
        "tier_length": record.tier_length,
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inputs = tokenize_prompt(
        tokenizer, record.prompt, device=str(device),
        max_length=None, chat_template="never",
    )
    prompt_ids = inputs["input_ids"]
    seq_len = int(prompt_ids.shape[1])
    out["tokens"] = seq_len

    layer_to_group, _ = _layer_to_group_map(num_layers, num_groups, num_layers)

    t0 = time.perf_counter()
    trace = capture_prompt_trace(
        model, inputs,
        n_layers_to_track=num_layers,
        return_logits=False,
        include_values=False,
        query_slice_start=None,  # accumulated needs full Q; others slice internally
    )
    out["capture_prefill_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    session = FlexCachedDecodeSession(
        model, prompt_ids, layer_to_group, device=device,
        masked_attn_implementation=masked_attn_implementation,
    )
    out["prefill_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref_tokens, _ref_logits = session.full_kv_reference(k_gen)
    out["ref_decode_s"] = time.perf_counter() - t0
    out["full_kv_tokens"] = list(map(int, ref_tokens))

    # Allocation matching the audit's canonical uniform cell at the
    # requested ratio. Floors carve out the protected sink + recent
    # positions; discretionary tokens are split uniformly across groups.
    floors = protected_budget_floors(seq_len, num_groups, n_sink, n_recent)
    floor_sum = int(sum(floors))
    total = max(int(round(ratio * seq_len)), floor_sum)
    discretionary = max(total - floor_sum, 0)
    per_group = discretionary // num_groups
    allocation = [int(floors[g] + per_group) for g in range(num_groups)]
    # Distribute the discretionary remainder so |max - min| <= 1.
    rem = discretionary - per_group * num_groups
    for g in range(rem):
        allocation[g] += 1

    out["ratio"] = float(ratio)
    out["total_budget"] = int(sum(allocation))
    out["allocation"] = list(allocation)

    per_priority: dict[str, Any] = {}
    for pri in priorities:
        t0 = time.perf_counter()
        group_priorities = select_group_priorities(
            trace, layer_to_group, num_groups, obs_window, seq_len,
            priority=pri, seed=priority_seed, device=str(device),
        )
        hot_per_group = group_hot_sets(
            allocation, group_priorities, seq_len, n_sink, n_recent,
        )
        pri_compute_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        gen_tokens = session.masked_decode_autoregressive(hot_per_group, k_gen)
        gen_s = time.perf_counter() - t0

        scores = _scores(out["full_kv_tokens"], gen_tokens)
        per_priority[pri] = {
            "tokens": list(map(int, gen_tokens)),
            "priority_compute_s": pri_compute_s,
            "autoregressive_decode_s": gen_s,
            **scores,
        }
    out["per_priority"] = per_priority
    out["peak_mem_gb"] = (
        float(torch.cuda.max_memory_allocated() / (1024**3))
        if torch.cuda.is_available() else float("nan")
    )
    out["status"] = "ok"
    return out


def _trajectories_payload(eval_config: dict, results: list[dict]) -> dict:
    payload = dict(eval_config)
    payload["prompts_attempted"] = len(results)
    payload["prompts_ok"] = sum(1 for r in results if r.get("status") == "ok")
    payload["prompts"] = results
    return payload


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="E -- end-to-end reference-fidelity under masked KV.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k-gen", type=int, default=16,
                   help="Number of tokens to autoregressively generate.")
    p.add_argument("--obs-window", type=int, default=32)
    p.add_argument("--num-groups", type=int, default=4)
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--n-recent", type=int, default=64)
    p.add_argument("--ratio", type=float, default=0.04,
                   help="Uniform-allocation ratio to test (canonical "
                        "audit tightest budget is 0.04).")
    p.add_argument("--priorities", type=str, default="snapkv,random",
                   help="Comma-separated priority names from "
                        f"{list(PRIORITY_MODES)}; all run per prompt with "
                        "shared prefill.")
    p.add_argument("--priority-seed", type=int, default=0)
    p.add_argument("--layernorm-placement", default="pre",
                   choices=["pre", "post"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    priorities = [s.strip() for s in args.priorities.split(",") if s.strip()]
    for pri in priorities:
        if pri not in PRIORITY_MODES:
            raise ValueError(
                f"unknown priority {pri!r}; expected one of "
                f"{list(PRIORITY_MODES)}"
            )
    records = load_manifest_records(args.manifest)
    print(f"[e2e] manifest {args.manifest}: {len(records)} prompts; "
          f"priorities={priorities} ratio={args.ratio} k_gen={args.k_gen}",
          flush=True)

    prefill_attn, masked_attn = attention_backends_for_model(args.model)
    print(f"[e2e] loading {args.model} (prefill={prefill_attn} / "
          f"masked={masked_attn})", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        args.model, device=args.device, attn_implementation=prefill_attn,
    )
    device = resolve_model_input_device(model, args.device)
    num_layers = int(model.config.num_hidden_layers)

    if args.layernorm_placement == "post":
        install_post_norm_hooks(model)

    eval_config = {
        "manifest": str(args.manifest),
        "model": args.model,
        "device": str(device),
        "k_gen": args.k_gen,
        "obs_window": args.obs_window,
        "num_groups": args.num_groups,
        "n_sink": args.n_sink,
        "n_recent": args.n_recent,
        "ratio": args.ratio,
        "priorities": priorities,
        "priority_seed": args.priority_seed,
        "layernorm_placement": args.layernorm_placement,
        "attn_backend": f"{prefill_attn} prefill / {masked_attn} masked decode",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.out / "eval_config.json", eval_config)
    results: list[dict[str, Any]] = []
    _write_json_atomic(
        args.out / "trajectories.json", _trajectories_payload(eval_config, results)
    )

    for i, record in enumerate(records):
        print(f"[e2e] {i + 1}/{len(records)}  L~{record.tier_length}  "
              f"{record.task}  {record.prompt_id}", flush=True)
        try:
            result = _run_prompt(
                record, model=model, tokenizer=tokenizer, device=device,
                num_layers=num_layers, num_groups=args.num_groups,
                obs_window=args.obs_window, k_gen=args.k_gen,
                n_sink=args.n_sink, n_recent=args.n_recent,
                ratio=args.ratio, priorities=priorities,
                priority_seed=args.priority_seed,
                masked_attn_implementation=masked_attn,
            )
            for pri, pri_out in result["per_priority"].items():
                print(f"[e2e]   {pri:14}  first_div={pri_out['first_divergence_step']}  "
                      f"match_full={pri_out['prefix_match_rate_full']:.3f}",
                      flush=True)
        except Exception as exc:
            status = type(exc).__name__
            print(f"[e2e]   FAIL ({status}): {exc}", flush=True)
            traceback.print_exc()
            result = {
                "prompt_id": record.prompt_id,
                "task": record.task,
                "tier_length": record.tier_length,
                "status": status,
                "error": repr(exc),
            }
        results.append(result)
        _write_json_atomic(
            args.out / "trajectories.json", _trajectories_payload(eval_config, results)
        )

    print(f"[e2e] done; wrote {len(results)} prompts to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
