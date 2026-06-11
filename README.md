# KV-PR: A Participation-Ratio Selector for Architecture-Dependent KV Cache Compression

Code companion to the paper *KV-PR: A Participation-Ratio Selector
for Architecture-Dependent KV Cache Compression*.

The paper audits the token-priority axis of KV-cache compression on
LLaMA-3.2-1B (GQA-4), Qwen3-1.7B (GQA-2), and Phi-3.5-mini (no GQA),
finds an architecture-dependent priority-ordering inversion on Phi at
long context, traces it to attention diffuseness via a two-regime
decomposition of masked-decode KL, and proposes a one-time per-model
selector that picks SnapKV or independent random per-group based on
the model's median participation ratio.

---

## Contents

1. [Layout](#layout)
2. [Install](#install)
3. [Reproduce a Single Audit Cell](#reproduce-a-single-audit-cell)
4. [Mechanism Probes](#mechanism-probes)
5. [Reproduce End-to-End Token-Level Fidelity](#reproduce-end-to-end-token-level-fidelity)
6. [Recompute Paper Numbers](#recompute-paper-numbers)
7. [HPC Notes](#hpc-notes)
8. [Code Architecture](#code-architecture)
9. [Tests](#tests)
10. [Citation](#citation)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

---

## Layout

| Path | Purpose |
|---|---|
| `cli.py` | CLI entry point. `python cli.py <command> --help` lists arguments per command. |
| `src/dart_pagedkv/` | Audit primitives: masked-decode session, priorities, allocation grid, GQA / layer-norm intervention hooks, direct cache-perturbation operator-norm estimator. |
| `src/benchmarks/`, `src/trace/` | Benchmark loaders, model loading, prompt tokenization, Q/K capture. |
| `experiments/scripts/` | Producer scripts that drive the audit primitives. |
| `analysis/` | Database plus analyzers that recompute the paper's numbers from `analysis/paper_data.db`. |
| `tests/` | Unit tests for the audit primitives and mechanism estimators. |

Model weights, RULER raw data, and producer outputs are not included
in the repository; they are large and regenerable.

The codebase has three layers, mirroring the paper's three
contributions:

1. **Per-cell measurement primitives** (`src/dart_pagedkv/`) build a
   KV cache, apply a priority's mask, run a masked decode, and
   compute the per-step KL divergence against an uncompressed
   reference.
2. **Mechanism probes** (also in `src/dart_pagedkv/`, exposed through
   producer flags) intervene on the model's forward pass (GQA
   emulation, layer-norm placement) or estimate the masked-decode
   operator norm directly via cache-perturbation power iteration.
3. **Producer scripts** (`experiments/scripts/`) loop over prompts,
   priorities, and allocations, drive the primitives and probes, and
   write per-prompt trajectories to disk.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

## Reproduce a Single Audit Cell

```bash
# 1. Build a RULER prompt manifest for a single tier.
python cli.py ruler-length-sweep-manifest \
    --tier 32768=data/ruler_l32768 \
    --prompts-per-tier 110 \
    --out experiments/manifests/demo_l32768.json

# 2. Run the long-context decode probe under a single priority.
python cli.py longcontext-decode-probe \
    --manifest experiments/manifests/demo_l32768.json \
    --model models/phi-3.5-mini-instruct \
    --priority snapkv \
    --ratios 0.04,0.08,0.16 \
    --out analyses/demo_phi35_snapkv_l32768

# 3. Compute alpha / beta / cell-level KL.
python cli.py longcontext-decode-probe-analyze \
    --in-dir analyses/demo_phi35_snapkv_l32768
```

`--priority` accepts `snapkv`, `random`, `accumulated`, `recent`,
`tova`.

## Mechanism Probes

Exposed as flags on the `longcontext-decode-probe` producer:

- `--gqa-emulation-factor F` averages the model's KV heads in groups
  of size `F`; used for the architectural-causality test (Phi at
  `F` in `{1,2,4,8}` and Qwen at the same factors).
- `--layernorm-placement {pre,post}` switches every decoder layer
  between pre-norm and post-norm forward; used for the layer-norm
  intervention.
- `--record-decoder-jacobian` enables the cache-perturbation
  power-iteration estimator of the per-step masked-decode operator's
  top singular value. Use `--jacobian-eps 0.05 --jacobian-n-iter 10`
  for the bf16-safe default.

## Reproduce End-to-End Token-Level Fidelity

```bash
python cli.py e-e2e-accuracy \
    --manifest experiments/manifests/demo_l32768.json \
    --model models/phi-3.5-mini-instruct \
    --priorities snapkv,random \
    --ratio 0.04 \
    --k-gen 16 \
    --out analyses/demo_e_phi35
```

The `e-e2e-accuracy` command shares prefill once per prompt and runs
autoregressive greedy decode under each requested priority against
the full-KV reference, recording match rates at `k` in `{2,4,8,16}`.

## Recompute Paper Numbers

The analyzers in `analysis/` read a SQLite database
`analysis/paper_data.db` holding every per-cell measurement.

```bash
python analysis/build_db.py             # build analysis/paper_data.db
python analysis/stats.py                # statistics tables
python analysis/paper_numbers.py        # headline numbers
python analysis/mediation.py            # SCI mediation regression
python analysis/gauss_newton.py         # Gauss-Newton expansion check on Phi
python analysis/operator_norm.py        # direct top singular value
python analysis/task_heterogeneity.py   # per-task family heterogeneity
python analysis/gqa_tova.py             # GQA-emulation + TOVA results
python analysis/bias_regime.py          # Phi cross-tier regime checks
python analysis/e2e_fidelity.py         # end-to-end fidelity summary
```

## HPC Notes

The producer scripts run unchanged on any GPU with sufficient memory.
Our reference runs used H100 80GB and, where noted, H200 141GB.

### Memory Budget per Tier

Phi-3.5-mini-Instruct is the most memory-hungry model in the panel
because its multi-head (non-GQA) attention gives a per-token KV cache
of approximately 51.5 GiB at 131,072 tokens, which exceeds the H100
80GB budget once model weights and the FlashAttention-2 prefill
workspace are added. The 131k Phi runs therefore require an H200
141GB; the other tiers fit on H100.

| Model | Tier | GPU Requirement |
|---|---|---|
| LLaMA-3.2-1B-Instruct | 2k–131k | H100 80GB |
| Qwen3-1.7B | 2k–131k | H100 80GB |
| Phi-3.5-mini-Instruct | 2k–32k | H100 80GB |
| Phi-3.5-mini-Instruct | 131k | H200 141GB |

### Attention Backends

GPU attention backends differ by model because the Phi-3 remote-code
modeling does not implement SDPA:

| Model | Prefill | Masked Decode |
|---|---|---|
| LLaMA-3.2-1B, Qwen3-1.7B | SDPA | FlexAttention |
| Phi-3.5-mini | FlashAttention-2 | eager |

The producer auto-selects the backend pair via
`attention_backends_for_model`; no user action is needed.

### Data Staging

RULER prompt sets must be present on the GPU node before any audit
run. The repository does not include them (they are licensed by
NVIDIA and produced by RULER's own generator scripts). To stage
them, clone RULER, then build the prompt set with the model's
tokenizer:

```bash
git clone https://github.com/NVIDIA/RULER third_party/RULER
python cli.py prepare-official-ruler \
    --max-seq-length 32768 \
    --tokenizer-path models/phi-3.5-mini-instruct \
    --out-dir data/ruler_l32768_phi
```

Manifests built by `cli.py ruler-length-sweep-manifest` reference
the staged prompt directories (`data/ruler_l<TIER>` or
model-specific suffixes such as `data/ruler_l<TIER>_phi`).

### Pull Conventions

Producer outputs land in `analyses/<YYYY-MM-DD>_<name>_*/`, one
directory per submitted job. The directory contains
`eval_config.json` (the run's full configuration),
`trajectories.json` (raw per-prompt output), `stats.json` (summary),
and `report.md` (a human-readable analyzer output). The analyzers
under `analysis/` read these directories directly.

The repository does not store `analyses/` (it is gitignored). A
typical paper-reproduction workflow runs the audit on the GPU node
and then rsyncs the output directories back to a local copy of the
repo before running the analyzers.

## Code Architecture

### `src/dart_pagedkv/` — Audit Framework

| Module | Responsibility |
|---|---|
| `flex_decode.py` | `FlexCachedDecodeSession` — holds one prompt's prefill cache; serves `full_kv_reference` greedy decode and `masked_decode` (per-step KL, teacher-forced) or `masked_decode_autoregressive` (argmax-feedback). |
| `flex_mask.py` | Per-layer block-mask construction for FlexAttention decode under a hot/cold prompt-key policy. |
| `logit_kl.py` | Per-layer mask installation for eager and FlashAttention-2 paths; `token_kl_divergence` for the per-step KL metric. |
| `windowed_priority.py` | `windowed_recompute_attention` (the SnapKV column-sum primitive) and the priority families: SnapKV, accumulated (H2O-style chunked), recent, random per-group, TOVA (single-row column-sum), k-norm. |
| `published_allocators.py` | PyramidKV, inverse-pyramid, AdaKV-style head-priority-weighted allocators. |
| `budget_spread.py` | The 12-vector allocation grid (uniform + four single-dominant + four single-starved + three published allocators). |
| `two_tier.py` | `group_hot_sets` — combine a per-group priority vector with an allocation to produce per-group retained sets. |
| `filler.py` | Protected sink + recent floor for every layer-group's retained set. |
| `trace.py` | `capture_prompt_trace` — post-RoPE Q/K capture for priority computation. |
| `effective_support.py` | Mass-coverage / participation-ratio / entropy estimators for the diffuseness measurement. |
| `service_cost.py` | Cold-attention demand model used by the allocation grid's cost-minimising vectors. |
| `ruler_manifest.py` | Loader for the manifests produced by `cli.py ruler-length-sweep-manifest`. |
| `gqa_emulation.py` | Forward-hook intervention: average key/value heads in groups of factor `f`. |
| `layernorm_intervention.py` | Forward-hook intervention: switch every decoder layer from pre-norm to post-norm. |
| `power_iteration.py` | Central-difference JVP + autograd VJP power iteration with unit-norm grad-output reformulation (bf16-safe). |
| `decoder_jacobian.py` | Binds `power_iteration` to a `FlexCachedDecodeSession` to recover the top singular value of the per-step masked-decode operator. |

### `experiments/scripts/` — Producers

| Script | Purpose |
|---|---|
| `ruler_length_sweep_manifest.py` | Build a RULER prompt manifest for a single tier. |
| `longcontext_decode_probe.py` | The main audit producer. Supports mechanism probe flags (`--gqa-emulation-factor`, `--layernorm-placement`, `--record-decoder-jacobian`). |
| `longcontext_decode_probe_analyze.py` | Aggregate a producer pull into cell-level statistics. |
| `longcontext_decode_probe_merge.py` | Merge multiple producer pulls (same cell, different prompt offsets). |
| `e_e2e_accuracy.py` | End-to-end token-level reference-fidelity probe. |
| `prepare_official_ruler.py` | Stage RULER prompts at a target tier with the model's tokenizer. |

## Citation

```bibtex
@article{singh2026kvpr,
  title  = {KV-PR: A Participation-Ratio Selector for
            Architecture-Dependent KV Cache Compression},
  author = {Singh, Pratyush},
  year   = {2026},
  note   = {Preprint.}
}
```

## License

Code is released under the MIT License (`LICENSE`).

## Acknowledgements

Compute was provided by the California Institute of Technology
Resnick High-Performance Computing Center. Phi-3.5-mini-Instruct,
LLaMA-3.2-1B-Instruct, and Qwen3-1.7B are released under their
respective licenses by Microsoft, Meta, and Alibaba.
