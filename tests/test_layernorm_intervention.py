"""Unit tests for the P27 post-norm layernorm intervention.

We mock a minimal pre-norm decoder layer and verify:
  1. The patch picks the right forward (Phi-3 vs LLaMA-style) based on
     whether ``resid_*_dropout`` modules are present.
  2. The patched forward produces a *different* output from the
     unpatched (pre-norm) forward when norms / attention / mlp are
     non-trivial -- i.e. the patch actually swaps placement.
  3. Reverting via ``uninstall_post_norm_hooks`` restores the original
     output exactly.
  4. ``install_post_norm_hooks`` on a model with no recognisable
     decoder layer raises.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.dart_pagedkv.layernorm_intervention import (
    install_post_norm_hooks,
    uninstall_post_norm_hooks,
    temporary_post_norm,
)


class _MockAttention(nn.Module):
    """Toy self-attention: linear + return (out, None) like Phi-3 does."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states), None


class _MockMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return torch.relu(self.fc(hidden_states))


class _Phi3LikeDecoderLayer(nn.Module):
    """Mirrors the transformers Phi3DecoderLayer surface: input_layernorm,
    post_attention_layernorm, self_attn, mlp, resid_*_dropout."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = _MockAttention(hidden_size)
        self.mlp = _MockMLP(hidden_size)
        self.resid_attn_dropout = nn.Dropout(0.0)
        self.resid_mlp_dropout = nn.Dropout(0.0)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(hidden_states)
        hidden_states = residual + self.resid_attn_dropout(attn_out)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        return hidden_states


class _LlamaLikeDecoderLayer(nn.Module):
    """Mirrors the LLaMA-style decoder block surface: no resid dropouts."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = _MockAttention(hidden_size)
        self.mlp = _MockMLP(hidden_size)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(hidden_states)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _StackedModel(nn.Module):
    """A small model holding N decoder layers, like a transformer's
    ``model.layers``. The intervention is applied across all of them."""

    def __init__(self, layer_cls, n_layers: int = 2, hidden_size: int = 16):
        super().__init__()
        self.layers = nn.ModuleList(
            [layer_cls(hidden_size) for _ in range(n_layers)]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TestLayernormInterventionStructure(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2031)
        self.x = torch.randn(2, 4, 16)

    def test_phi3_style_post_norm_changes_output(self):
        model = _StackedModel(_Phi3LikeDecoderLayer, n_layers=2)
        model.eval()
        with torch.no_grad():
            pre_out = model(self.x)
        patches = install_post_norm_hooks(model)
        self.assertEqual(len(patches), 2)
        with torch.no_grad():
            post_out = model(self.x)
        uninstall_post_norm_hooks(patches)
        # Output must differ -- pre-norm and post-norm of a non-trivial
        # network with random init produce different activations.
        diff = (pre_out - post_out).norm().item()
        self.assertGreater(diff, 0.1)

    def test_llama_style_post_norm_changes_output(self):
        model = _StackedModel(_LlamaLikeDecoderLayer, n_layers=2)
        model.eval()
        with torch.no_grad():
            pre_out = model(self.x)
        patches = install_post_norm_hooks(model)
        self.assertEqual(len(patches), 2)
        with torch.no_grad():
            post_out = model(self.x)
        uninstall_post_norm_hooks(patches)
        diff = (pre_out - post_out).norm().item()
        self.assertGreater(diff, 0.1)

    def test_uninstall_restores_original_output_exactly(self):
        model = _StackedModel(_Phi3LikeDecoderLayer, n_layers=3)
        model.eval()
        with torch.no_grad():
            original = model(self.x)
        patches = install_post_norm_hooks(model)
        uninstall_post_norm_hooks(patches)
        with torch.no_grad():
            after_revert = model(self.x)
        # Bit-exact restoration: the forward is the same object we saved
        torch.testing.assert_close(original, after_revert, rtol=0, atol=0)

    def test_context_manager_revert_on_exit(self):
        model = _StackedModel(_Phi3LikeDecoderLayer, n_layers=2)
        model.eval()
        with torch.no_grad():
            pre = model(self.x)
        with temporary_post_norm(model, enabled=True):
            with torch.no_grad():
                inside = model(self.x)
            self.assertGreater((pre - inside).norm().item(), 0.1)
        with torch.no_grad():
            after = model(self.x)
        torch.testing.assert_close(pre, after, rtol=0, atol=0)

    def test_context_manager_disabled_noop(self):
        model = _StackedModel(_Phi3LikeDecoderLayer, n_layers=2)
        model.eval()
        with torch.no_grad():
            pre = model(self.x)
        with temporary_post_norm(model, enabled=False):
            with torch.no_grad():
                inside = model(self.x)
        torch.testing.assert_close(pre, inside, rtol=0, atol=0)

    def test_no_recognised_layer_raises(self):
        class _NotADecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(x)

        with self.assertRaises(ValueError):
            install_post_norm_hooks(_NotADecoder())


if __name__ == "__main__":
    unittest.main()
