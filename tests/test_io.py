"""Tests for vlora.io — adapter parsing and I/O."""

import json

import pytest
import torch
from safetensors.torch import save_file

from vlora.io import LoRAWeights, load_adapter, parse_state_dict, save_adapter, stack_lora_weights


def _make_peft_state_dict(layers, rank=4, in_features=64, out_features=64):
    """Create a PEFT-style state dict."""
    sd = {}
    for layer in layers:
        sd[f"base_model.model.{layer}.lora_A.weight"] = torch.randn(rank, in_features)
        sd[f"base_model.model.{layer}.lora_B.weight"] = torch.randn(out_features, rank)
    return sd


class TestParseStateDict:
    def test_basic_parse(self):
        layers = ["model.layers.0.self_attn.q_proj"]
        sd = _make_peft_state_dict(layers, rank=8)
        lora_a, lora_b, names = parse_state_dict(sd)
        assert names == layers
        assert lora_a[layers[0]].shape == (8, 64)

    def test_multiple_layers(self):
        layers = ["model.layers.0.q_proj", "model.layers.0.v_proj", "model.layers.1.q_proj"]
        sd = _make_peft_state_dict(layers)
        _, _, names = parse_state_dict(sd)
        assert names == sorted(layers)

    def test_ignores_non_lora_keys(self):
        sd = {"some.random.key": torch.randn(10)}
        sd.update(_make_peft_state_dict(["layer.0.q_proj"]))
        _, _, names = parse_state_dict(sd)
        assert len(names) == 1

    def test_requires_both_A_and_B(self):
        sd = {"base_model.model.layer.0.lora_A.weight": torch.randn(4, 64)}
        _, _, names = parse_state_dict(sd)
        assert len(names) == 0  # No layer has both A and B


class TestLoadSaveAdapter:
    def test_roundtrip(self, tmp_path):
        layers = ["layers.0.q_proj", "layers.0.v_proj"]
        rank = 4

        original = LoRAWeights(
            layer_names=layers,
            lora_a={l: torch.randn(rank, 64) for l in layers},
            lora_b={l: torch.randn(64, rank) for l in layers},
            rank=rank,
            metadata={"r": rank, "peft_type": "LORA"},
        )

        save_adapter(original, tmp_path / "adapter")
        loaded = load_adapter(tmp_path / "adapter")

        assert loaded.layer_names == original.layer_names
        assert loaded.rank == rank
        for layer in layers:
            assert torch.allclose(loaded.lora_a[layer], original.lora_a[layer])
            assert torch.allclose(loaded.lora_b[layer], original.lora_b[layer])

    def test_load_from_safetensors(self, tmp_path):
        layers = ["model.layers.0.q_proj"]
        sd = _make_peft_state_dict(layers, rank=8, in_features=32)
        save_file(sd, str(tmp_path / "adapter_model.safetensors"))

        config = {"r": 8, "peft_type": "LORA"}
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(config, f)

        adapter = load_adapter(tmp_path)
        assert adapter.rank == 8
        assert len(adapter.layer_names) == 1


class TestStackWeights:
    def test_stacks_correctly(self):
        layers = ["layer.0"]
        adapters = [
            LoRAWeights(layers, {layers[0]: torch.randn(4, 64)}, {layers[0]: torch.randn(64, 4)}, 4),
            LoRAWeights(layers, {layers[0]: torch.randn(4, 64)}, {layers[0]: torch.randn(64, 4)}, 4),
        ]
        stacked = stack_lora_weights(adapters, side="A")
        assert stacked["layer.0"].shape == (2, 4 * 64)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stack_lora_weights([], side="A")
