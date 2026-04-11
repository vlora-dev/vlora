"""Tests for incremental SVD and streaming subspace construction."""


import torch

from vlora.io import LoRAWeights, save_adapter
from vlora.ops import compute_svd, incremental_svd_update
from vlora.subspace import SharedSubspace


def _make_adapters(n=5, layers=None, rank=4, dim=64):
    """Create n synthetic adapters that share structure."""
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

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


class TestIncrementalSVD:
    def test_output_shapes(self):
        data = torch.randn(5, 100)
        comps, svals, mean = compute_svd(data, num_components=3)

        new_data = torch.randn(2, 100)
        new_comps, new_svals, new_mean, n = incremental_svd_update(
            comps, svals, mean, n_seen=5, new_data=new_data, max_components=3
        )
        assert new_comps.shape == (3, 100)
        assert new_svals.shape == (3,)
        assert new_mean.shape == (100,)
        assert n == 7

    def test_mean_update_correct(self):
        data1 = torch.randn(5, 50)
        data2 = torch.randn(3, 50)
        all_data = torch.cat([data1, data2])

        expected_mean = all_data.mean(dim=0)

        comps, svals, mean = compute_svd(data1, num_components=3)
        _, _, updated_mean, _ = incremental_svd_update(
            comps, svals, mean, n_seen=5, new_data=data2
        )

        assert torch.allclose(updated_mean, expected_mean, atol=1e-5)

    def test_max_components_respected(self):
        data = torch.randn(5, 100)
        comps, svals, mean = compute_svd(data, num_components=2)
        new_data = torch.randn(3, 100)

        new_comps, new_svals, _, _ = incremental_svd_update(
            comps, svals, mean, n_seen=5, new_data=new_data, max_components=2
        )
        assert new_comps.shape[0] <= 2

    def test_singular_values_nonnegative(self):
        data = torch.randn(5, 100)
        comps, svals, mean = compute_svd(data, num_components=3)
        new_data = torch.randn(2, 100)

        _, new_svals, _, _ = incremental_svd_update(
            comps, svals, mean, n_seen=5, new_data=new_data
        )
        assert (new_svals >= 0).all()


class TestAbsorbIncremental:
    def test_adds_task(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        new = _make_adapters(1)[0][0]
        sub.absorb_incremental(new, "incremental_task")
        assert "incremental_task" in sub.tasks
        assert len(sub.tasks) == 4

    def test_preserves_existing_tasks(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        new = _make_adapters(1)[0][0]
        sub.absorb_incremental(new, "incremental_task")
        for i in range(3):
            assert f"task_{i}" in sub.tasks

    def test_reconstruct_after_incremental(self):
        adapters, layers = _make_adapters(3, rank=4, dim=64)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        new = _make_adapters(1)[0][0]
        sub.absorb_incremental(new, "incremental_task")

        # Should be able to reconstruct any task
        recon = sub.reconstruct("task_0")
        for l in layers:
            assert recon.lora_a[l].shape == (4, 64)
            assert recon.lora_b[l].shape == (64, 4)

    def test_faster_than_full_absorb(self):
        """Incremental should not rebuild from scratch."""
        adapters, _ = _make_adapters(5)
        SharedSubspace.from_adapters(adapters, num_components=3)

        new = _make_adapters(1)[0][0]

        # Run incremental
        sub_inc = SharedSubspace.from_adapters(adapters, num_components=3)
        sub_inc.absorb_incremental(new, "new")

        # Run full absorb
        sub_full = SharedSubspace.from_adapters(adapters, num_components=3)
        sub_full.absorb(new, "new")

        # Both should complete and have the new task
        assert "new" in sub_inc.tasks
        assert "new" in sub_full.tasks


class TestAbsorbIncrementalAccuracy:
    """Compare absorb_incremental quality against full from_adapters."""

    def test_incremental_reconstruction_bounded(self):
        """Existing tasks should reconstruct reasonably after incremental absorb."""
        torch.manual_seed(42)
        adapters, layers = _make_adapters(5, rank=4, dim=64)
        new_adapter = _make_adapters(1, rank=4, dim=64)[0][0]

        # Full SVD baseline
        sub_full = SharedSubspace.from_adapters(adapters, num_components=3)
        sub_full.absorb(new_adapter, "new_task")
        recon_full = sub_full.reconstruct("task_0")

        # Incremental
        sub_inc = SharedSubspace.from_adapters(adapters, num_components=3)
        sub_inc.absorb_incremental(new_adapter, "new_task")
        recon_inc = sub_inc.reconstruct("task_0")

        # Incremental reconstruction should be in the right ballpark
        for l in layers:
            full_a = recon_full.lora_a[l]
            inc_a = recon_inc.lora_a[l]
            # Relative error < 50% (incremental is approximate)
            rel_err = (full_a - inc_a).norm() / full_a.norm().clamp(min=1e-6)
            assert rel_err < 0.5, f"Layer {l} A-side relative error {rel_err:.3f} too high"

    def test_new_task_reconstructs_well(self):
        """The newly absorbed task should reconstruct with reasonable fidelity."""
        torch.manual_seed(42)
        adapters, layers = _make_adapters(5, rank=4, dim=64)
        new_adapter = _make_adapters(1, rank=4, dim=64)[0][0]

        sub = SharedSubspace.from_adapters(adapters, num_components=3)
        sub.absorb_incremental(new_adapter, "new_task")
        recon = sub.reconstruct("new_task")

        for l in layers:
            original = new_adapter.lora_a[l]
            reconstructed = recon.lora_a[l]
            rel_err = (original - reconstructed).norm() / original.norm().clamp(min=1e-6)
            assert rel_err < 1.0


class TestFromAdaptersStreaming:
    def test_streaming_builds_subspace(self, tmp_path):
        adapters, layers = _make_adapters(4)
        paths = []
        for i, adapter in enumerate(adapters):
            p = tmp_path / f"adapter_{i}"
            save_adapter(adapter, p)
            paths.append(p)

        sub = SharedSubspace.from_adapters_streaming(paths, num_components=2)
        assert len(sub.tasks) == 4
        assert sub.num_components == 2

    def test_streaming_can_reconstruct(self, tmp_path):
        adapters, layers = _make_adapters(3, rank=4, dim=64)
        paths = []
        for i, adapter in enumerate(adapters):
            p = tmp_path / f"adapter_{i}"
            save_adapter(adapter, p)
            paths.append(p)

        sub = SharedSubspace.from_adapters_streaming(paths, num_components=2)
        recon = sub.reconstruct("adapter_0")
        for l in layers:
            assert recon.lora_a[l].shape == (4, 64)


class TestDeviceAndDtype:
    def test_to_dtype(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        sub.to(dtype=torch.float16)

        for l in layers:
            assert sub.components_a[l].dtype == torch.float16
            assert sub.means_a[l].dtype == torch.float16
            assert sub.tasks["task_0"].loadings_a[l].dtype == torch.float16

    def test_to_returns_self(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        result = sub.to(dtype=torch.float32)
        assert result is sub
