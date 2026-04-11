"""Integration tests for vlora.pipeline."""

import torch

from vlora.io import LoRAWeights, save_adapter
from vlora.pipeline import absorb_task, extract_adapter, init_subspace


def _save_synthetic_adapters(tmp_path, n=3, layers=None, rank=4, dim=64):
    """Save synthetic adapters to disk and return paths."""
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

    paths = []
    for i in range(n):
        adapter = LoRAWeights(
            layer_names=layers,
            lora_a={l: torch.randn(rank, dim) * 0.01 for l in layers},
            lora_b={l: torch.randn(dim, rank) * 0.01 for l in layers},
            rank=rank,
            metadata={"r": rank},
        )
        path = tmp_path / f"adapter_{i}"
        save_adapter(adapter, path)
        paths.append(path)

    return paths, layers


class TestInitSubspace:
    def test_basic(self, tmp_path):
        paths, _ = _save_synthetic_adapters(tmp_path)
        sub = init_subspace(paths, num_components=2)
        assert sub.num_components == 2
        assert len(sub.tasks) == 3


class TestAbsorbTask:
    def test_absorb(self, tmp_path):
        paths, layers = _save_synthetic_adapters(tmp_path, n=3)
        sub = init_subspace(paths, num_components=2)

        # Save a new adapter to absorb
        new_adapter = LoRAWeights(
            layer_names=layers,
            lora_a={l: torch.randn(4, 64) * 0.01 for l in layers},
            lora_b={l: torch.randn(64, 4) * 0.01 for l in layers},
            rank=4,
            metadata={"r": 4},
        )
        new_path = tmp_path / "new_adapter"
        save_adapter(new_adapter, new_path)

        absorb_task(sub, new_path, "new")
        assert "new" in sub.tasks


class TestExtractAdapter:
    def test_extract(self, tmp_path):
        paths, layers = _save_synthetic_adapters(tmp_path)
        sub = init_subspace(paths, task_ids=["a", "b", "c"], num_components=2)

        output = tmp_path / "extracted"
        weights = extract_adapter(sub, "a", output)

        assert (output / "adapter_model.safetensors").exists()
        assert (output / "adapter_config.json").exists()
        assert weights.rank == 4
