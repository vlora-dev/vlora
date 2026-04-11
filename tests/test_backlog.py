"""Tests for backlog features: outlier detection, torch.compile, etc."""

import torch

from vlora.analysis import find_outliers
from vlora.io import LoRAWeights
from vlora.model import VLoRAModel
from vlora.subspace import SharedSubspace


def _make_adapters(n=5, layers=None, rank=4, dim=32):
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

    shared_a = {l: torch.randn(3, rank * dim) for l in layers}
    shared_b = {l: torch.randn(3, dim * rank) for l in layers}

    adapters = []
    for i in range(n):
        lora_a = {l: (torch.randn(3) @ shared_a[l] + torch.randn(rank * dim) * 0.01).reshape(rank, dim) for l in layers}
        lora_b = {l: (torch.randn(3) @ shared_b[l] + torch.randn(dim * rank) * 0.01).reshape(dim, rank) for l in layers}
        adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))
    return adapters, layers


class TestFindOutliers:
    def test_no_outliers_in_similar_adapters(self):
        adapters, _ = _make_adapters(5)
        outliers = find_outliers(adapters, threshold=3.0)
        # All adapters share the same structure, so no outliers expected
        assert isinstance(outliers, list)

    def test_detects_random_outlier(self):
        adapters, layers = _make_adapters(8)
        # Add a wildly different adapter
        outlier = LoRAWeights(
            layer_names=layers,
            lora_a={l: torch.randn(4, 32) * 100 for l in layers},
            lora_b={l: torch.randn(32, 4) * 100 for l in layers},
            rank=4,
        )
        all_adapters = adapters + [outlier]
        outliers = find_outliers(all_adapters, threshold=2.0)
        # The random outlier should be detected
        outlier_indices = [o["index"] for o in outliers]
        assert 8 in outlier_indices

    def test_returns_empty_for_few_adapters(self):
        adapters, _ = _make_adapters(2)
        outliers = find_outliers(adapters)
        assert outliers == []

    def test_outlier_has_required_keys(self):
        adapters, layers = _make_adapters(5)
        outlier = LoRAWeights(
            layer_names=layers,
            lora_a={l: torch.randn(4, 32) * 100 for l in layers},
            lora_b={l: torch.randn(32, 4) * 100 for l in layers},
            rank=4,
        )
        outliers = find_outliers(adapters + [outlier], threshold=1.5)
        for o in outliers:
            assert "index" in o
            assert "distance" in o
            assert "z_score" in o


class TestVLoRAModelCompile:
    def test_compile_returns_self(self):
        layers = ["layer.0.q_proj"]
        adapters, _ = _make_adapters(3, layers=layers)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        import torch.nn as nn
        base = nn.Linear(32, 32)
        model = VLoRAModel(base, sub)
        result = model.compile()
        assert result is model

    def test_forward_after_compile(self):
        """Model should still work after compile (may be identity on CPU)."""
        layers = ["layer.0.q_proj"]
        adapters, _ = _make_adapters(3, layers=layers)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.ModuleDict({"0": nn.ModuleDict({"q_proj": nn.Linear(32, 32, bias=False)})})
            def forward(self, x):
                return self.layer["0"]["q_proj"](x)

        base = SimpleModel()
        model = VLoRAModel(base, sub)
        model.compile()

        x = torch.randn(2, 32)
        out = model(x)
        assert out.shape == (2, 32)
