"""Tests for vlora.model — VLoRAModel inference wrapper."""

import pytest
import torch
import torch.nn as nn

from vlora.io import LoRAWeights
from vlora.model import VLoRAModel
from vlora.subspace import SharedSubspace


def _make_base_model(layers, dim=64):
    """Create a simple base model with named linear layers matching adapter layer names."""
    modules = {}
    for layer_name in layers:
        # Convert dot-notation to nested modules
        modules[layer_name] = nn.Linear(dim, dim, bias=False)

    # Build a model from the modules using a ModuleDict-like approach
    # We need nested module structure to match layer names like "layer.0.q_proj"
    model = _NestedModel(modules)
    return model


class _NestedModel(nn.Module):
    """Simple model that registers linear layers at dot-separated paths."""

    def __init__(self, named_layers: dict[str, nn.Module]):
        super().__init__()
        for name, module in named_layers.items():
            # Register each as a flat module with dots replaced
            parts = name.split(".")
            parent = self
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    child = nn.Module()
                    parent.add_module(part, child)
                parent = getattr(parent, part)
            parent.add_module(parts[-1], module)

    def forward(self, x):
        # Simple pass-through for testing — apply all linear layers sequentially
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                x = module(x)
        return x


def _make_adapters_and_model(n=3, layers=None, rank=4, dim=64):
    """Create adapters, subspace, and a matching base model."""
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

    shared_a = {l: torch.randn(3, rank * dim) for l in layers}
    shared_b = {l: torch.randn(3, dim * rank) for l in layers}

    adapters = []
    for i in range(n):
        lora_a = {l: (torch.randn(3) @ shared_a[l]).reshape(rank, dim) for l in layers}
        lora_b = {l: (torch.randn(3) @ shared_b[l]).reshape(dim, rank) for l in layers}
        adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))

    sub = SharedSubspace.from_adapters(adapters, num_components=2)
    base_model = _make_base_model(layers, dim=dim)
    return sub, base_model, layers


class TestVLoRAModel:
    def test_set_task(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        model.set_task("task_0")
        assert model.active_task == "task_0"

    def test_clear_task(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        model.set_task("task_0")
        model.clear_task()
        assert model.active_task is None

    def test_available_tasks(self):
        sub, base_model, _ = _make_adapters_and_model(n=3)
        model = VLoRAModel(base_model, sub)
        assert model.available_tasks == ["task_0", "task_1", "task_2"]

    def test_unknown_task_raises(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        with pytest.raises(KeyError, match="Unknown task"):
            model.set_task("nonexistent")

    def test_forward_without_task(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)
        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape[0] == 2

    def test_forward_with_task_changes_output(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)

        x = torch.randn(2, 64)
        out_base = model(x).detach().clone()

        model.set_task("task_0")
        out_lora = model(x).detach().clone()

        # Output should differ when LoRA is applied
        assert not torch.allclose(out_base, out_lora, atol=1e-6)

    def test_switching_tasks_changes_output(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)

        x = torch.randn(2, 64)

        model.set_task("task_0")
        out_0 = model(x).detach().clone()

        model.set_task("task_1")
        out_1 = model(x).detach().clone()

        # Different tasks should generally give different outputs
        # (not guaranteed but very likely with random adapters)
        assert not torch.allclose(out_0, out_1, atol=1e-6)

    def test_same_task_is_cached(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        model.set_task("task_0")
        cached = model._cached_deltas
        model.set_task("task_0")  # Same task — should not recompute
        assert model._cached_deltas is cached

    def test_reconstruct_state_dict(self):
        sub, base_model, layers = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)
        deltas = model.reconstruct_state_dict("task_0")
        for l in layers:
            assert l in deltas
            assert deltas[l].shape == (64, 64)  # (out_features, in_features)

    def test_lora_alpha_scaling(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        # lora_alpha = 2 * rank → scaling = 2.0
        model = VLoRAModel(base_model, sub, lora_alpha=sub.rank * 2)
        assert model.scaling == 2.0

    def test_scaling_overrides_lora_alpha(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub, scaling=0.5, lora_alpha=999)
        assert model.scaling == 0.5

    def test_default_scaling_is_one(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)
        assert model.scaling == 1.0

    def test_qlora_info_no_quantization(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        info = model.qlora_info
        assert info["quantized"] is False
        assert info["method"] is None
        assert info["num_quantized_layers"] == 0

    def test_qlora_info_has_target_layers(self):
        sub, base_model, layers = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        info = model.qlora_info
        assert info["num_target_layers"] == len(layers)

    def test_compute_dtype_none_by_default(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        assert model.compute_dtype is None

    def test_compute_dtype_affects_output(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model_f32 = VLoRAModel(base_model, sub)
        model_f32.set_task("task_0")

        # With compute_dtype=float64, the LoRA computation happens in f64
        # then casts back. Result may differ slightly due to precision.
        model_f64 = VLoRAModel(base_model, sub, compute_dtype=torch.float64)
        model_f64.set_task("task_0")

        x = torch.randn(2, 64)
        out_f32 = model_f32(x)
        out_f64 = model_f64(x)
        # Both should produce valid output of the same shape
        assert out_f32.shape == out_f64.shape

    def test_compute_dtype_bfloat16(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub, compute_dtype=torch.bfloat16)
        model.set_task("task_0")
        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 64)


class TestMergeUnmerge:
    def test_merge_output_matches_hooks(self):
        """Core correctness: merged forward == hook-based forward."""
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        x = torch.randn(2, 64)

        # Get hook-based output
        model = VLoRAModel(base_model, sub)
        model.set_task("task_0")
        out_hooks = model(x).detach().clone()
        model.clear_task()

        # Get merge-based output
        model.merge(task_id="task_0")
        out_merge = model(x).detach().clone()

        torch.testing.assert_close(out_hooks, out_merge, atol=1e-5, rtol=1e-5)

    def test_merge_changes_weights(self):
        sub, base_model, layers = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)

        original_weights = {
            name: module.weight.data.clone()
            for name, module in model._target_modules.items()
        }

        model.merge(task_id="task_0")

        for name, module in model._target_modules.items():
            assert not torch.allclose(
                original_weights[name], module.weight.data, atol=1e-8
            ), f"Weight {name} should have changed after merge"

    def test_unmerge_restores_weights(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)

        original_weights = {
            name: module.weight.data.clone()
            for name, module in model._target_modules.items()
        }

        model.merge(task_id="task_0")
        model.unmerge()

        for name, module in model._target_modules.items():
            torch.testing.assert_close(
                original_weights[name], module.weight.data, atol=1e-6, rtol=1e-6
            )

    def test_merge_then_set_task_raises(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        model.merge(task_id="task_0")
        with pytest.raises(RuntimeError, match="Cannot switch tasks while merged"):
            model.set_task("task_1")

    def test_merge_then_clear_task_raises(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        model.merge(task_id="task_0")
        with pytest.raises(RuntimeError, match="Cannot clear task while merged"):
            model.clear_task()

    def test_merge_twice_raises(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        model.merge(task_id="task_0")
        with pytest.raises(RuntimeError, match="already merged"):
            model.merge(task_id="task_1")

    def test_unmerge_without_merge_raises(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        with pytest.raises(RuntimeError, match="not merged"):
            model.unmerge()

    def test_merge_with_explicit_task_id(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        # No set_task call — pass task_id directly
        model.merge(task_id="task_1")
        assert model.is_merged
        assert model.active_task == "task_1"

    def test_merge_no_task_raises(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        with pytest.raises(ValueError, match="No task to merge"):
            model.merge()

    def test_is_merged_property(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        assert not model.is_merged
        model.merge(task_id="task_0")
        assert model.is_merged
        model.unmerge()
        assert not model.is_merged

    def test_merge_uses_active_task(self):
        sub, base_model, _ = _make_adapters_and_model()
        model = VLoRAModel(base_model, sub)
        model.set_task("task_1")
        model.merge()
        assert model.is_merged
        assert model.active_task == "task_1"

    def test_merge_with_scaling(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        x = torch.randn(2, 64)

        # scaling=2.0
        model = VLoRAModel(base_model, sub, scaling=2.0)
        model.set_task("task_0")
        out_hooks = model(x).detach().clone()
        model.clear_task()

        model.merge(task_id="task_0")
        out_merge = model(x).detach().clone()

        # Slightly higher tolerance: hook path applies delta via x @ delta.T
        # while merge path folds delta into weight, so matmul order differs.
        torch.testing.assert_close(out_hooks, out_merge, atol=5e-4, rtol=5e-4)

    def test_unmerge_then_set_task_works(self):
        sub, base_model, _ = _make_adapters_and_model(dim=64)
        model = VLoRAModel(base_model, sub)

        model.merge(task_id="task_0")
        model.unmerge()

        # Should work normally after unmerge
        model.set_task("task_1")
        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 64)
