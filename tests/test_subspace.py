"""Tests for vlora.subspace — SharedSubspace core class."""


import pytest
import torch

from vlora.io import LoRAWeights
from vlora.subspace import SharedSubspace, TaskProjection


def _make_adapters(n=5, layers=None, rank=4, dim=64):
    """Create n synthetic adapters that share structure."""
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

    # Create a shared basis so adapters are correlated (realistic scenario)
    shared_a = {l: torch.randn(3, rank * dim) for l in layers}
    shared_b = {l: torch.randn(3, dim * rank) for l in layers}

    adapters = []
    for i in range(n):
        lora_a = {}
        lora_b = {}
        for l in layers:
            coeffs_a = torch.randn(3)
            coeffs_b = torch.randn(3)
            lora_a[l] = (coeffs_a @ shared_a[l] + torch.randn(rank * dim) * 0.01).reshape(rank, dim)
            lora_b[l] = (coeffs_b @ shared_b[l] + torch.randn(dim * rank) * 0.01).reshape(dim, rank)
        adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))

    return adapters, layers


class TestFromAdapters:
    def test_basic_init(self):
        adapters, layers = _make_adapters(5)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        assert sub.num_components == 2
        assert len(sub.tasks) == 5
        assert set(sub.layer_names) == set(layers)

    def test_auto_component_selection(self):
        adapters, _ = _make_adapters(5)
        sub = SharedSubspace.from_adapters(adapters, variance_threshold=0.5)
        assert sub.num_components >= 1

    def test_custom_task_ids(self):
        adapters, _ = _make_adapters(3)
        ids = ["alpha", "beta", "gamma"]
        sub = SharedSubspace.from_adapters(adapters, task_ids=ids, num_components=2)
        assert set(sub.tasks.keys()) == set(ids)

    def test_mismatched_ids_raises(self):
        adapters, _ = _make_adapters(3)
        with pytest.raises(ValueError):
            SharedSubspace.from_adapters(adapters, task_ids=["a", "b"])


class TestProjectAndReconstruct:
    def test_project_returns_task_projection(self):
        adapters, layers = _make_adapters(5)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)
        new_adapter = _make_adapters(1)[0][0]
        proj = sub.project(new_adapter, "new")
        assert isinstance(proj, TaskProjection)
        assert proj.task_id == "new"
        for l in layers:
            assert proj.loadings_a[l].shape == (3,)

    def test_reconstruct_shape(self):
        adapters, layers = _make_adapters(5, rank=4, dim=64)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)
        recon = sub.reconstruct("task_0")
        for l in layers:
            assert recon.lora_a[l].shape == (4, 64)
            assert recon.lora_b[l].shape == (64, 4)

    def test_in_subspace_reconstruction_is_good(self):
        """Adapters used to build the subspace should reconstruct well."""
        adapters, layers = _make_adapters(5)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)

        recon = sub.reconstruct("task_0")
        orig = adapters[0]
        for l in layers:
            error = (orig.lora_a[l].flatten() - recon.lora_a[l].flatten()).norm()
            orig_norm = orig.lora_a[l].flatten().norm()
            relative_error = error / (orig_norm + 1e-8)
            # With 3 components for data that lives in ~3D subspace, error should be small
            assert relative_error < 0.5, f"Reconstruction error too high: {relative_error}"


class TestAbsorb:
    def test_absorb_adds_task(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        new = _make_adapters(1)[0][0]
        sub.absorb(new, "new_task")
        assert "new_task" in sub.tasks
        assert len(sub.tasks) == 4

    def test_absorb_preserves_existing_tasks(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        new = _make_adapters(1)[0][0]
        sub.absorb(new, "new_task")
        # All original tasks should still be present
        for i in range(3):
            assert f"task_{i}" in sub.tasks


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        sub.save(tmp_path / "subspace")
        loaded = SharedSubspace.load(tmp_path / "subspace")

        assert loaded.layer_names == sub.layer_names
        assert loaded.num_components == sub.num_components
        assert set(loaded.tasks.keys()) == set(sub.tasks.keys())

        for l in layers:
            assert torch.allclose(loaded.components_a[l], sub.components_a[l])
            assert torch.allclose(loaded.components_b[l], sub.components_b[l])


class TestSaveLoadQuantized:
    def test_nf4_roundtrip(self, tmp_path):
        adapters, layers = _make_adapters(3, rank=4, dim=64)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        # Save quantized
        sub.save_quantized(tmp_path / "subspace_nf4")
        loaded = SharedSubspace.load(tmp_path / "subspace_nf4")

        assert loaded.layer_names == sub.layer_names
        assert loaded.num_components == sub.num_components
        assert set(loaded.tasks.keys()) == set(sub.tasks.keys())

        # Components should be NF4-approximated (not exact)
        for l in layers:
            assert loaded.components_a[l].shape == sub.components_a[l].shape

    def test_nf4_reconstruction_close(self, tmp_path):
        adapters, layers = _make_adapters(3, rank=4, dim=64)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        recon_orig = sub.reconstruct("task_0")

        sub.save_quantized(tmp_path / "subspace_nf4")
        loaded = SharedSubspace.load(tmp_path / "subspace_nf4")
        recon_nf4 = loaded.reconstruct("task_0")

        for l in layers:
            diff = (recon_orig.lora_a[l] - recon_nf4.lora_a[l]).abs().max()
            assert diff < 2.0  # NF4 introduces some error

    def test_nf4_files_smaller(self, tmp_path):
        adapters, _ = _make_adapters(3, rank=4, dim=64)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        sub.save(tmp_path / "float")
        sub.save_quantized(tmp_path / "nf4")

        float_size = (tmp_path / "float" / "subspace.safetensors").stat().st_size
        nf4_size = (tmp_path / "nf4" / "subspace_nf4.safetensors").stat().st_size
        # Packed should be significantly smaller
        assert nf4_size < float_size


class TestNonSquareLoRA:
    """Test with in_features != out_features (realistic LoRA shapes)."""

    def test_asymmetric_shapes_roundtrip(self):
        layers = ["layer.0.q_proj"]
        rank = 4
        in_features = 512
        out_features = 128
        adapters = []
        for _ in range(3):
            lora_a = {layers[0]: torch.randn(rank, in_features)}
            lora_b = {layers[0]: torch.randn(out_features, rank)}
            adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))

        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        recon = sub.reconstruct("task_0")
        assert recon.lora_a[layers[0]].shape == (rank, in_features)
        assert recon.lora_b[layers[0]].shape == (out_features, rank)

    def test_shapes_stored_correctly(self):
        layers = ["layer.0.q_proj"]
        rank = 8
        in_features = 1024
        out_features = 256
        adapters = []
        for _ in range(3):
            lora_a = {layers[0]: torch.randn(rank, in_features)}
            lora_b = {layers[0]: torch.randn(out_features, rank)}
            adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))

        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        assert sub.shapes_a[layers[0]] == (rank, in_features)
        assert sub.shapes_b[layers[0]] == (out_features, rank)

    def test_shapes_survive_save_load(self, tmp_path):
        layers = ["layer.0.q_proj"]
        rank = 4
        in_features = 512
        out_features = 128
        adapters = []
        for _ in range(3):
            lora_a = {layers[0]: torch.randn(rank, in_features)}
            lora_b = {layers[0]: torch.randn(out_features, rank)}
            adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))

        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        sub.save(tmp_path / "subspace")
        loaded = SharedSubspace.load(tmp_path / "subspace")

        recon = loaded.reconstruct("task_0")
        assert recon.lora_a[layers[0]].shape == (rank, in_features)
        assert recon.lora_b[layers[0]].shape == (out_features, rank)


class TestSpecialCharTaskIds:
    """Test task IDs with filesystem-unsafe characters."""

    def test_save_load_with_slashes(self, tmp_path):
        adapters, layers = _make_adapters(2)
        ids = ["model/v1", "model/v2"]
        sub = SharedSubspace.from_adapters(adapters, task_ids=ids, num_components=2)
        sub.save(tmp_path / "subspace")
        loaded = SharedSubspace.load(tmp_path / "subspace")
        assert set(loaded.tasks.keys()) == set(ids)

    def test_save_load_with_colons_and_spaces(self, tmp_path):
        adapters, _ = _make_adapters(2)
        ids = ["task:v2", "my task"]
        sub = SharedSubspace.from_adapters(adapters, task_ids=ids, num_components=2)
        sub.save(tmp_path / "subspace")
        loaded = SharedSubspace.load(tmp_path / "subspace")
        assert set(loaded.tasks.keys()) == set(ids)
        # Verify reconstruction works
        recon = loaded.reconstruct("task:v2")
        assert recon.rank == sub.rank


class TestRepr:
    def test_subspace_repr(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        r = repr(sub)
        assert "SharedSubspace" in r
        assert "k=2" in r
        assert "tasks=3" in r

    def test_task_projection_repr(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        proj = sub.tasks["task_0"]
        r = repr(proj)
        assert "TaskProjection" in r
        assert "task_0" in r

    def test_lora_weights_repr(self):
        adapters, _ = _make_adapters(1)
        r = repr(adapters[0])
        assert "LoRAWeights" in r
        assert "rank=4" in r


class TestAdaptiveKPreserved:
    def test_absorb_preserves_adaptive_k(self):
        adapters, _ = _make_adapters(5)
        sub = SharedSubspace.from_adapters(adapters, adaptive_k=True, variance_threshold=0.8)
        assert sub.adaptive_k is True
        assert sub.variance_threshold == 0.8

        new = _make_adapters(1)[0][0]
        sub.absorb(new, "new_task")
        assert sub.adaptive_k is True
        assert sub.variance_threshold == 0.8


class TestTrainableParams:
    def test_returns_params_with_grad(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        params = sub.get_trainable_params("task_0")
        for name, p in params.items():
            assert p.requires_grad
        assert len(params) == len(layers) * 2  # A and B per layer

    def test_expand_adds_dimensions(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        params = sub.get_trainable_params("task_0", num_expand=2)
        # Each loading should now have 2 + 2 = 4 dimensions (or more if Gram-Schmidt found more)
        for name, p in params.items():
            assert p.shape[0] >= 4
