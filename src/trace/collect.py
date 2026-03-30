"""Attention capture and training-sample extraction for live inference."""

from __future__ import annotations

import importlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..predictor.model import build_feature_rows


def _install_transformers_cache_compat() -> None:
    """Patch old remote-code cache helpers onto newer Transformers caches.

    Phi-3.5's hosted ``modeling_phi3.py`` still calls the pre-5.x
    ``DynamicCache.from_legacy_cache()``, ``to_legacy_cache()``,
    ``get_usable_length()``, ``get_max_length()``, and ``seen_tokens`` APIs.
    Transformers 5.5 keeps equivalent primitives but removed those names. The
    shim is intentionally tiny and matches the dynamic-cache, no-max-length
    semantics used by our single-prompt decode probes.
    """
    try:
        from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
    except Exception:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def _from_legacy_cache(cls, past_key_values=None, *args, **kwargs):
            if past_key_values is None:
                return cls()
            layers = []
            for layer_idx, layer_past in enumerate(past_key_values):
                if layer_past is None:
                    continue
                key_states, value_states = layer_past[:2]
                if key_states is None or value_states is None:
                    continue
                layer = DynamicLayer()
                layer.dtype = key_states.dtype
                layer.device = key_states.device
                layer.keys = key_states
                layer.values = value_states
                layer.is_initialized = True
                layers.append(layer)
            if not layers:
                return cls()
            cache = object.__new__(cls)
            Cache.__init__(cache, layers=layers)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(Cache, "to_legacy_cache"):

        def _to_legacy_cache(self):
            legacy = []
            for layer in getattr(self, "layers", []):
                key_states = getattr(layer, "keys", None)
                value_states = getattr(layer, "values", None)
                if key_states is None or value_states is None:
                    continue
                legacy.append((key_states, value_states))
            return tuple(legacy)

        Cache.to_legacy_cache = _to_legacy_cache

    if not hasattr(Cache, "__getitem__"):

        def _cache_getitem(self, layer_idx: int):
            layer = self.layers[layer_idx]
            return getattr(layer, "keys", None), getattr(layer, "values", None)

        Cache.__getitem__ = _cache_getitem

    if not hasattr(Cache, "get_usable_length"):

        def _get_usable_length(self, new_seq_length=None, layer_idx: int = 0):
            return self.get_seq_length(layer_idx)

        Cache.get_usable_length = _get_usable_length

    if not hasattr(Cache, "get_max_length"):

        def _get_max_length(self):
            return None

        Cache.get_max_length = _get_max_length

    if not hasattr(Cache, "seen_tokens"):
        Cache.seen_tokens = property(lambda self: self.get_seq_length())


@dataclass(slots=True)
class LayerQKCapture:
    """Captured Q/K tensors for one attention layer."""

    query: torch.Tensor
    key: torch.Tensor
    scaling: float
    num_key_value_groups: int


@dataclass(slots=True)
class PrefillArtifacts:
    """Outputs shared by all evaluation runs for a prompt."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: Optional[torch.Tensor]
    past_key_values: Optional[object]
    qk_layers: Dict[int, LayerQKCapture]

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "cuda",
    dtype=None,
    attn_implementation: Optional[str] = None,
):
    """Load a HF model/tokenizer pair for causal LM inference.

    ``attn_implementation`` overrides the default attention backend. Leave it
    ``None`` for the usual behaviour (flash-attention on CUDA, SDPA otherwise);
    pass ``"eager"`` when a caller needs to inject custom per-layer additive
    attention masks (flash-attention rejects arbitrary 4D masks).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _install_transformers_cache_compat()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    preferred_attn = attn_implementation or (
        "flash_attention_2" if device.startswith("cuda") else "sdpa"
    )
    load_kwargs = {
        "dtype": model_dtype,
        "attn_implementation": preferred_attn,
        "trust_remote_code": True,
    }

    def _from_pretrained(**extra_kwargs):
        kwargs = dict(load_kwargs)
        kwargs.update(extra_kwargs)
        try:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        except (ImportError, ValueError) as exc:
            if kwargs.get("attn_implementation") != "flash_attention_2":
                raise
            # flash-attn not installed/compatible: retry with SDPA.
            kwargs["attn_implementation"] = "sdpa"
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        except TypeError as exc:
            # Backward compatibility with older transformers that still use torch_dtype.
            if "dtype" not in str(exc):
                raise
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    if device.startswith("cuda"):
        load_kwargs["low_cpu_mem_usage"] = True
        device_map = {"": 0} if device == "cuda" else {"": device}
        try:
            model = _from_pretrained(device_map=device_map)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                model = _from_pretrained(device_map="auto", offload_buffers=True)
            except TypeError as exc:
                if "offload_buffers" not in str(exc):
                    raise
                model = _from_pretrained(device_map="auto")
        except (ImportError, ValueError) as exc:
            # Older environments may not have accelerate/device_map support.
            if "accelerate" not in str(exc).lower() and "device_map" not in str(exc).lower():
                raise
            model = _from_pretrained().to(device)
    else:
        model = _from_pretrained().to(device)
    model.eval()
    return model, tokenizer


def resolve_model_input_device(model, fallback_device: str = "cuda") -> torch.device:
    """Best-effort input device resolution for sharded/offloaded HF models."""
    try:
        embed = model.get_input_embeddings()
        if embed is not None and hasattr(embed, "weight"):
            weight_device = embed.weight.device
            if weight_device.type != "meta":
                return weight_device
    except Exception:
        pass
    return torch.device(fallback_device)


def format_prompt_for_tokenizer(tokenizer, prompt_text: str, chat_template: str = "never") -> str:
    """Optionally wrap a benchmark prompt with the tokenizer's chat template."""
    if chat_template == "never":
        return prompt_text
    has_template = bool(getattr(tokenizer, "chat_template", None))
    if chat_template == "auto" and not has_template:
        return prompt_text
    if chat_template not in {"auto", "always"}:
        raise ValueError(f"Unsupported chat_template mode: {chat_template}")
    if chat_template == "always" and not has_template:
        raise ValueError("Tokenizer does not define a chat_template")
    if not hasattr(tokenizer, "apply_chat_template"):
        if chat_template == "always":
            raise ValueError("Tokenizer does not expose apply_chat_template")
        return prompt_text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def middle_truncate_inputs(encoded: Dict[str, torch.Tensor], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Keep the start and end of long prompts, matching LongBench evaluation."""
    if max_length is None or "input_ids" not in encoded:
        return encoded
    seq_len = int(encoded["input_ids"].shape[1])
    if seq_len <= max_length:
        return encoded

    head = max_length // 2
    tail = max_length - head
    truncated: Dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if isinstance(value, torch.Tensor) and value.ndim == 2 and int(value.shape[1]) == seq_len:
            truncated[key] = torch.cat([value[:, :head], value[:, -tail:]], dim=1)
        else:
            truncated[key] = value
    return truncated


def tokenize_prompt(
    tokenizer,
    prompt_text: str,
    device: str,
    max_length: Optional[int] = None,
    chat_template: str = "never",
) -> Dict[str, torch.Tensor]:
    prompt_text = format_prompt_for_tokenizer(tokenizer, prompt_text, chat_template=chat_template)
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=False)
    encoded = middle_truncate_inputs(encoded, max_length=max_length)
    return {key: value.to(device) for key, value in encoded.items()}


def _find_attention_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    modules = [(name, module) for name, module in model.named_modules() if name.endswith(".self_attn")]
    if modules:
        return modules
    modules = [(name, module) for name, module in model.named_modules() if name.endswith(".attn")]
    if modules:
        return modules
    raise RuntimeError("Unable to locate attention modules on the model.")


def _resolve_rotary_function(module) -> Optional[callable]:
    module_path = type(module).__module__
    candidates = [module_path]
    if "modeling_" in module_path:
        parts = module_path.split(".")
        if len(parts) >= 2:
            candidates.append(".".join(parts[:-1]) + ".modeling_llama")
            candidates.append(".".join(parts[:-1]) + ".modeling_qwen2")

    for candidate in candidates:
        try:
            imported = importlib.import_module(candidate)
        except Exception:
            continue
        fn = getattr(imported, "apply_rotary_pos_emb", None)
        if fn is not None:
            return fn
    return None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _slice_rotary_rows(
    tensor: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    if tensor.ndim >= 2:
        return tensor[..., start:stop, :]
    return tensor


def _apply_rotary_direct(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # HF Llama/Phi/Qwen attention tensors here are [B, H, T, D], while
    # position embeddings are [B, T, D] or [T, D]. Unsqueezing at dim=1
    # matches their apply_rotary_pos_emb convention.
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def _project_qk(
    module,
    hidden_states: torch.Tensor,
    *,
    position_ids: Optional[torch.Tensor] = None,
    position_embeddings=None,
    query_slice_start: Optional[int] = None,
):
    batch_size, seq_len, _ = hidden_states.shape
    if query_slice_start is None:
        q_start = 0
    else:
        q_start = max(0, min(int(query_slice_start), seq_len))
    q_hidden = hidden_states[:, q_start:, :]
    q_len = int(q_hidden.shape[1])
    if hasattr(module, "q_proj") and hasattr(module, "k_proj"):
        q = module.q_proj(q_hidden).view(batch_size, q_len, -1, module.head_dim).transpose(1, 2)
        k = module.k_proj(hidden_states).view(batch_size, seq_len, -1, module.head_dim).transpose(1, 2)
    elif hasattr(module, "qkv_proj"):
        query_width = int(module.num_heads) * int(module.head_dim)
        kv_width = int(module.num_key_value_heads) * int(module.head_dim)
        weight = module.qkv_proj.weight
        bias = module.qkv_proj.bias
        q_bias = bias[:query_width] if bias is not None else None
        k_bias = bias[query_width: query_width + kv_width] if bias is not None else None
        q = F.linear(q_hidden, weight[:query_width], q_bias)
        k = F.linear(
            hidden_states,
            weight[query_width: query_width + kv_width],
            k_bias,
        )
        q = q.view(batch_size, q_len, int(module.num_heads), int(module.head_dim)).transpose(1, 2)
        k = k.view(batch_size, seq_len, int(module.num_key_value_heads), int(module.head_dim)).transpose(1, 2)
    else:
        raise AttributeError(
            f"{type(module).__name__} exposes neither q_proj/k_proj nor qkv_proj"
        )

    if position_embeddings is not None:
        cos, sin = position_embeddings
        if q_start == 0 and q_len == seq_len:
            rotary = _resolve_rotary_function(module)
            if rotary is not None:
                try:
                    q, k = rotary(q, k, cos, sin)
                except TypeError:
                    q, k = rotary(q, k, cos, sin, position_ids)
        else:
            q = _apply_rotary_direct(
                q,
                _slice_rotary_rows(cos, q_start, seq_len),
                _slice_rotary_rows(sin, q_start, seq_len),
            )
            k = _apply_rotary_direct(k, cos, sin)
    elif position_ids is not None and hasattr(module, "rotary_emb"):
        rotary = _resolve_rotary_function(module)
        if rotary is not None:
            rotary_seq_len = max(int(seq_len), int(position_ids[:, -1].max().item()) + 1)
            cos, sin = module.rotary_emb(k, position_ids, seq_len=rotary_seq_len)
            if q_start == 0 and q_len == seq_len:
                try:
                    q, k = rotary(q, k, cos, sin)
                except TypeError:
                    q, k = rotary(q, k, cos, sin, position_ids)
            else:
                q = _apply_rotary_direct(
                    q,
                    _slice_rotary_rows(cos, q_start, seq_len),
                    _slice_rotary_rows(sin, q_start, seq_len),
                )
                k = _apply_rotary_direct(k, cos, sin)
    return q, k


def run_prefill_with_qk_capture(
    model,
    inputs: Dict[str, torch.Tensor],
    *,
    n_layers_to_track: Optional[int] = None,
    return_past_key_values: bool = True,
    return_logits: bool = True,
    query_slice_start: Optional[int] = None,
) -> PrefillArtifacts:
    """Run a prompt prefill, returning logits, KV cache, and captured Q/K tensors."""
    qk_layers: Dict[int, LayerQKCapture] = {}
    attn_modules = _find_attention_modules(model)
    n_keep = min(n_layers_to_track or len(attn_modules), len(attn_modules))
    keep_layers = set(range(len(attn_modules) - n_keep, len(attn_modules)))

    def make_hook(layer_idx: int):
        def hook_fn(module, args, kwargs, output):
            if layer_idx not in keep_layers:
                return
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            position_ids = kwargs.get("position_ids")
            if hidden_states is None:
                return

            query, key = _project_qk(
                module,
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                query_slice_start=query_slice_start,
            )

            qk_layers[layer_idx] = LayerQKCapture(
                query=query.detach().cpu(),
                key=key.detach().cpu(),
                scaling=float(getattr(module, "scaling", 1.0 / np.sqrt(module.head_dim))),
                num_key_value_groups=int(getattr(module, "num_key_value_groups", 1)),
            )
        return hook_fn

    hooks = [
        module.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
        for layer_idx, (_, module) in enumerate(attn_modules)
    ]
    try:
        with torch.inference_mode():
            if return_logits:
                outputs = model(
                    **inputs,
                    use_cache=return_past_key_values,
                    output_attentions=False,
                    return_dict=True,
                )
                logits = outputs.logits
            else:
                base_model = getattr(model, "model", None)
                runner = base_model if callable(base_model) else model
                forward_kwargs = {
                    **inputs,
                    "use_cache": return_past_key_values,
                    "output_attentions": False,
                    "return_dict": True,
                }
                try:
                    outputs = runner(**forward_kwargs)
                except TypeError:
                    # Some base models do not accept output_attentions in this call path.
                    forward_kwargs.pop("output_attentions", None)
                    outputs = runner(**forward_kwargs)
                logits = outputs.logits if runner is model else None
    finally:
        for hook in hooks:
            hook.remove()

    return PrefillArtifacts(
        input_ids=inputs["input_ids"].detach().cpu(),
        attention_mask=inputs["attention_mask"].detach().cpu(),
        logits=logits.detach().cpu() if logits is not None else None,
        past_key_values=getattr(outputs, "past_key_values", None) if return_past_key_values else None,
        qk_layers=qk_layers,
    )


def recompute_attention_matrix(
    qk_layers: Dict[int, LayerQKCapture],
    device: str = "cuda",
    heads_per_batch: int = 8,
) -> np.ndarray:
    """Recompute the mean attention matrix from captured Q/K tensors."""
    if not qk_layers:
        raise ValueError("No Q/K captures available.")

    first_capture = next(iter(qk_layers.values()))
    seq_len = int(first_capture.query.shape[2])
    accumulator = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32),
        diagonal=1,
    )

    for capture in qk_layers.values():
        query = capture.query
        key = capture.key
        n_heads = int(query.shape[1])
        layer_acc = torch.zeros(seq_len, seq_len, dtype=torch.float32)

        for start in range(0, n_heads, heads_per_batch):
            stop = min(start + heads_per_batch, n_heads)
            query_batch = query[:, start:stop].to(device)
            key_indices = [head // capture.num_key_value_groups for head in range(start, stop)]
            key_batch = key[:, key_indices].to(device)
            scores = torch.matmul(query_batch, key_batch.transpose(-2, -1)) * capture.scaling
            scores = scores + causal_mask
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
            layer_acc += weights[0].sum(dim=0).cpu()
            del query_batch, key_batch, scores, weights

        accumulator += layer_acc / max(n_heads, 1)

    return (accumulator / max(len(qk_layers), 1)).numpy().astype(np.float32, copy=False)


def summarize_prefill_attention(attention_matrix: np.ndarray, window_size: int = 5) -> Dict[str, np.ndarray]:
    """Summaries consumed by initial live policy selection."""
    seq_len = attention_matrix.shape[0]
    window_rows = attention_matrix[max(0, seq_len - window_size):].copy()
    return {
        "attention_matrix": attention_matrix,
        "cumulative_attention": attention_matrix.sum(axis=0),
        "recent_window": window_rows,
    }


def _sample_position_indices(n_pos: int, position_stride: int) -> np.ndarray:
    if position_stride <= 1 or n_pos <= 1:
        return np.arange(n_pos, dtype=np.int64)
    sample_idx = np.arange(0, n_pos, position_stride, dtype=np.int64)
    if sample_idx[-1] != (n_pos - 1):
        sample_idx = np.append(sample_idx, n_pos - 1)
    return sample_idx


def extract_training_samples(
    attention_matrix: np.ndarray,
    *,
    window_size: int = 5,
    stride: int = 8,
    position_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (X, Y) training samples from a full attention matrix."""
    seq_len = attention_matrix.shape[0]
    cumulative = np.cumsum(attention_matrix, axis=0)
    total = cumulative[-1]

    features = []
    targets = []
    for step_idx in range(0, seq_len, max(1, stride)):
        n_pos = step_idx + 1
        start = max(0, step_idx - window_size + 1)
        block = attention_matrix[start: step_idx + 1, :n_pos]
        recent = np.zeros((n_pos, window_size), dtype=np.float32)
        recent[:, -block.shape[0]:] = block.T
        cumulative_snapshot = cumulative[step_idx, :n_pos]
        sample_idx = _sample_position_indices(n_pos, position_stride)
        sampled_recent = recent[sample_idx]
        sampled_cumulative = cumulative_snapshot[sample_idx]
        position_ids = sample_idx.astype(np.float32, copy=False)
        features.append(build_feature_rows(sampled_recent, sampled_cumulative, position_ids, current_step=n_pos))
        targets.append((total[sample_idx] - sampled_cumulative).astype(np.float32))

    if not features:
        return (
            np.zeros((0, window_size + 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


def stream_training_samples_from_qk_layers(
    captures: Dict[int, LayerQKCapture] | Iterable[LayerQKCapture],
    *,
    window_size: int = 5,
    stride: int = 64,
    position_stride: int = 1,
    row_batch_size: int = 128,
    heads_per_batch: int = 8,
    device: str = "cuda",
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract training samples from one or more captured layers without materializing T x T."""
    if isinstance(captures, dict):
        ordered_captures = [captures[layer_idx] for layer_idx in sorted(captures)]
    else:
        ordered_captures = list(captures)
    if not ordered_captures:
        return (
            np.zeros((0, window_size + 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    seq_len = int(ordered_captures[0].query.shape[2])
    for capture in ordered_captures[1:]:
        if int(capture.query.shape[2]) != seq_len:
            raise ValueError("All captured layers must share the same sequence length")

    cumulative = np.zeros(seq_len, dtype=np.float32)
    recent_rows: deque[np.ndarray] = deque(maxlen=window_size)

    emitted_features: List[np.ndarray] = []
    emitted_snapshots: List[np.ndarray] = []
    emitted_indices: List[np.ndarray] = []
    emitted_rows = 0

    keys_on_device = [capture.key[0].to(device) for capture in ordered_captures]
    try:
        for batch_start in range(0, seq_len, row_batch_size):
            batch_stop = min(batch_start + row_batch_size, seq_len)
            if progress_callback is not None:
                progress_callback(
                    phase="row_batch_start",
                    batch_start=int(batch_start),
                    batch_stop=int(batch_stop),
                    seq_len=int(seq_len),
                    emitted_rows=int(emitted_rows),
                )

            row_positions = torch.arange(batch_start, batch_stop, device=device)
            col_positions = torch.arange(seq_len, device=device)
            batch_acc = None

            for capture_idx, capture in enumerate(ordered_captures):
                query = capture.query[0]
                n_heads = int(query.shape[0])
                batch_queries = query[:, batch_start:batch_stop].to(device)
                layer_acc = None

                for head_start in range(0, n_heads, max(1, heads_per_batch)):
                    head_stop = min(head_start + max(1, heads_per_batch), n_heads)
                    query_batch = batch_queries[head_start:head_stop]
                    kv_indices = torch.arange(
                        head_start,
                        head_stop,
                        device=device,
                        dtype=torch.long,
                    ) // max(capture.num_key_value_groups, 1)
                    key_batch = keys_on_device[capture_idx].index_select(0, kv_indices)
                    score = torch.matmul(query_batch, key_batch.transpose(-2, -1)) * capture.scaling
                    score = score.masked_fill(
                        col_positions.unsqueeze(0) > row_positions.unsqueeze(1),
                        float("-inf"),
                    )
                    weights = torch.softmax(score, dim=-1, dtype=torch.float32)
                    if layer_acc is None:
                        layer_acc = weights.sum(dim=0)
                    else:
                        layer_acc += weights.sum(dim=0)
                    del query_batch, key_batch, score, weights

                layer_acc = layer_acc / max(n_heads, 1)
                if batch_acc is None:
                    batch_acc = layer_acc
                else:
                    batch_acc += layer_acc
                del batch_queries, layer_acc

            attn_batch = (batch_acc / max(len(ordered_captures), 1)).cpu().numpy()
            del batch_acc, row_positions, col_positions

            for offset, row in enumerate(attn_batch):
                step_idx = batch_start + offset
                n_pos = step_idx + 1
                row = row[:n_pos].astype(np.float32, copy=False)
                full_row = np.zeros(seq_len, dtype=np.float32)
                full_row[:n_pos] = row
                cumulative[:n_pos] += row
                recent_rows.append(full_row)

                if step_idx % max(1, stride) != 0:
                    continue

                recent = np.zeros((n_pos, window_size), dtype=np.float32)
                stacked = list(recent_rows)
                for idx, snapshot in enumerate(stacked):
                    recent[:, window_size - len(stacked) + idx] = snapshot[:n_pos]

                sample_idx = _sample_position_indices(n_pos, position_stride)
                sampled_cumulative = cumulative[:n_pos][sample_idx].copy()
                positions = sample_idx.astype(np.float32, copy=False)
                emitted_features.append(
                    build_feature_rows(recent[sample_idx], sampled_cumulative, positions, current_step=n_pos)
                )
                emitted_snapshots.append(sampled_cumulative)
                emitted_indices.append(sample_idx)
                emitted_rows += int(len(sample_idx))

            if progress_callback is not None:
                progress_callback(
                    phase="row_batch_done",
                    batch_start=int(batch_start),
                    batch_stop=int(batch_stop),
                    seq_len=int(seq_len),
                    emitted_rows=int(emitted_rows),
                )
    finally:
        del keys_on_device

    total = cumulative.copy()
    targets = [
        (total[idx] - snapshot).astype(np.float32)
        for idx, snapshot in zip(emitted_indices, emitted_snapshots)
    ]
    if not emitted_features:
        return (
            np.zeros((0, window_size + 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return np.concatenate(emitted_features, axis=0), np.concatenate(targets, axis=0)


def stream_last_layer_training_samples(
    capture: LayerQKCapture,
    *,
    window_size: int = 5,
    stride: int = 64,
    position_stride: int = 1,
    row_batch_size: int = 128,
    heads_per_batch: int = 8,
    device: str = "cuda",
    progress_callback=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for the streamed single-layer training path."""
    return stream_training_samples_from_qk_layers(
        [capture],
        window_size=window_size,
        stride=stride,
        position_stride=position_stride,
        row_batch_size=row_batch_size,
        heads_per_batch=heads_per_batch,
        device=device,
        progress_callback=progress_callback,
    )
