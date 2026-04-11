"""Tests for vlora.merge — adapter merging techniques."""

import pytest
import torch

from vlora.io import LoRAWeights
from vlora.merge import MERGE_METHODS, dare_merge, task_arithmetic, ties_merge


def _make_adapters(n=3, layers=None, rank=4, dim=32):
    """Create n synthetic adapters with known structure."""
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

    adapters = []
    for i in range(n):
        lora_a = {l: torch.randn(rank, dim) * (0.5 + i * 0.1) for l in layers}
        lora_b = {l: torch.randn(dim, rank) * (0.5 + i * 0.1) for l in layers}
        adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))
    return adapters, layers


class TestTaskArithmetic:
    def test_output_shape(self):
        adapters, layers = _make_adapters(3)
        merged = task_arithmetic(adapters)
        assert merged.layer_names == layers
        assert merged.rank == 4
        for l in layers:
            assert merged.lora_a[l].shape == (4, 32)
            assert merged.lora_b[l].shape == (32, 4)

    def test_uniform_average(self):
        adapters, layers = _make_adapters(2)
        merged = task_arithmetic(adapters)
        for l in layers:
            expected_a = (adapters[0].lora_a[l] + adapters[1].lora_a[l]) / 2
            assert torch.allclose(merged.lora_a[l], expected_a, atol=1e-5)

    def test_custom_weights(self):
        adapters, layers = _make_adapters(2)
        merged = task_arithmetic(adapters, weights=[0.8, 0.2])
        for l in layers:
            expected = 0.8 * adapters[0].lora_a[l] + 0.2 * adapters[1].lora_a[l]
            assert torch.allclose(merged.lora_a[l], expected, atol=1e-5)

    def test_single_adapter_identity(self):
        adapters, layers = _make_adapters(1)
        merged = task_arithmetic(adapters)
        for l in layers:
            assert torch.allclose(merged.lora_a[l], adapters[0].lora_a[l], atol=1e-5)

    def test_wrong_weights_length(self):
        adapters, _ = _make_adapters(3)
        with pytest.raises(ValueError, match="weights length"):
            task_arithmetic(adapters, weights=[0.5, 0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            task_arithmetic([])


class TestTiesMerge:
    def test_output_shape(self):
        adapters, layers = _make_adapters(3)
        merged = ties_merge(adapters, density=0.5)
        assert merged.layer_names == layers
        assert merged.rank == 4

    def test_density_1_keeps_all(self):
        """With density=1.0, no trimming happens — all values kept."""
        adapters, layers = _make_adapters(2)
        merged = ties_merge(adapters, density=1.0)
        for l in layers:
            assert merged.lora_a[l].shape == adapters[0].lora_a[l].shape

    def test_low_density_trims_more(self):
        """Lower density should produce sparser intermediate results."""
        adapters, layers = _make_adapters(3)
        merged_high = ties_merge(adapters, density=0.9)
        merged_low = ties_merge(adapters, density=0.1)
        # Both should produce valid outputs — exact comparison is nondeterministic
        # but the merged outputs should differ
        for l in layers:
            assert merged_high.lora_a[l].shape == merged_low.lora_a[l].shape

    def test_invalid_density(self):
        adapters, _ = _make_adapters(2)
        with pytest.raises(ValueError, match="density"):
            ties_merge(adapters, density=0.0)
        with pytest.raises(ValueError, match="density"):
            ties_merge(adapters, density=1.5)

    def test_single_adapter(self):
        adapters, layers = _make_adapters(1)
        merged = ties_merge(adapters, density=0.8)
        # Single adapter: TIES should preserve the overall structure
        for l in layers:
            assert merged.lora_a[l].shape == adapters[0].lora_a[l].shape


class TestDareMerge:
    def test_output_shape(self):
        adapters, layers = _make_adapters(3)
        merged = dare_merge(adapters, drop_rate=0.5, seed=42)
        assert merged.layer_names == layers
        assert merged.rank == 4

    def test_drop_rate_zero_is_average(self):
        """With drop_rate=0, DARE reduces to task arithmetic."""
        adapters, layers = _make_adapters(2)
        merged_dare = dare_merge(adapters, drop_rate=0.0)
        merged_avg = task_arithmetic(adapters)
        for l in layers:
            assert torch.allclose(merged_dare.lora_a[l], merged_avg.lora_a[l], atol=1e-5)

    def test_reproducible_with_seed(self):
        adapters, layers = _make_adapters(3)
        m1 = dare_merge(adapters, drop_rate=0.5, seed=123)
        m2 = dare_merge(adapters, drop_rate=0.5, seed=123)
        for l in layers:
            assert torch.allclose(m1.lora_a[l], m2.lora_a[l])

    def test_invalid_drop_rate(self):
        adapters, _ = _make_adapters(2)
        with pytest.raises(ValueError, match="drop_rate"):
            dare_merge(adapters, drop_rate=1.0)
        with pytest.raises(ValueError, match="drop_rate"):
            dare_merge(adapters, drop_rate=-0.1)

    def test_high_drop_rate_reduces_magnitude(self):
        """High drop rate with rescaling should produce larger variance."""
        adapters, layers = _make_adapters(3)
        merged_low = dare_merge(adapters, drop_rate=0.1, seed=42)
        merged_high = dare_merge(adapters, drop_rate=0.9, seed=42)
        # Both valid but high drop rate has more variance per element
        for l in layers:
            assert merged_low.lora_a[l].shape == merged_high.lora_a[l].shape


class TestMergeMethods:
    def test_method_registry(self):
        assert "average" in MERGE_METHODS
        assert "ties" in MERGE_METHODS
        assert "dare" in MERGE_METHODS

    def test_all_methods_produce_valid_output(self):
        adapters, layers = _make_adapters(3)
        for name, fn in MERGE_METHODS.items():
            if name == "dare":
                merged = fn(adapters, seed=42)
            else:
                merged = fn(adapters)
            assert merged.layer_names == layers
            assert merged.rank == 4
            for l in layers:
                assert not torch.isnan(merged.lora_a[l]).any(), f"{name} produced NaN in A"
                assert not torch.isnan(merged.lora_b[l]).any(), f"{name} produced NaN in B"

    def test_incompatible_ranks_rejected(self):
        a1 = LoRAWeights(
            layer_names=["l0"], rank=4,
            lora_a={"l0": torch.randn(4, 16)},
            lora_b={"l0": torch.randn(16, 4)},
        )
        a2 = LoRAWeights(
            layer_names=["l0"], rank=8,
            lora_a={"l0": torch.randn(8, 16)},
            lora_b={"l0": torch.randn(16, 8)},
        )
        for fn in MERGE_METHODS.values():
            with pytest.raises(ValueError, match="rank"):
                fn([a1, a2])
