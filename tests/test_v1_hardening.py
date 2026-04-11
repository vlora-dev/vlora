"""v1.0 hardening tests — edge cases, numerical precision, serialization, API surface."""

import tempfile
import warnings

import torch

from vlora.io import LoRAWeights
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


class TestEdgeCases:
    def test_rank_1_adapters(self):
        """rank=1 adapters through full pipeline."""
        adapters, layers = _make_adapters(n=3, rank=1, dim=32)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        assert sub.rank == 1
        assert len(sub.tasks) == 3

        # project + reconstruct should work
        sub.project(adapters[0], "test")
        weights = sub.reconstruct("task_0")
        assert weights.rank == 1
        for l in layers:
            assert weights.lora_a[l].shape[0] == 1

    def test_single_layer_adapters(self):
        """Adapters with only 1 layer."""
        adapters, layers = _make_adapters(n=3, layers=["single_layer"], rank=4, dim=32)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        assert len(sub.layer_names) == 1
        assert sub.layer_names[0] == "single_layer"

        weights = sub.reconstruct("task_0")
        assert "single_layer" in weights.lora_a

    def test_two_adapter_subspace(self):
        """from_adapters with n=2 (minimum for meaningful SVD)."""
        adapters, _ = _make_adapters(n=2)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        assert len(sub.tasks) == 2
        assert sub.num_components == 2

    def test_non_square_dimensions(self):
        """Adapters where in_features != out_features."""
        layers = ["proj"]
        rank = 4
        in_dim, out_dim = 32, 64

        adapters = []
        for _ in range(3):
            lora_a = {"proj": torch.randn(rank, in_dim)}
            lora_b = {"proj": torch.randn(out_dim, rank)}
            adapters.append(LoRAWeights(
                layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank,
            ))

        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        weights = sub.reconstruct("task_0")
        assert weights.lora_a["proj"].shape == (rank, in_dim)
        assert weights.lora_b["proj"].shape == (out_dim, rank)


class TestNumericalPrecision:
    def test_project_reconstruct_roundtrip(self):
        """Adapters used to build the subspace should reconstruct well."""
        adapters, layers = _make_adapters(n=5)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)

        for i in range(5):
            original = adapters[i]
            reconstructed = sub.reconstruct(f"task_{i}")

            for layer in layers:
                # Reconstruction error should be small (most variance captured)
                error_a = (original.lora_a[layer] - reconstructed.lora_a[layer]).norm()
                orig_norm_a = original.lora_a[layer].norm()
                if orig_norm_a > 1e-6:
                    relative_error = (error_a / orig_norm_a).item()
                    assert relative_error < 0.5, (
                        f"Reconstruction error too high: {relative_error:.3f}"
                    )

    def test_in_subspace_adapter_zero_error(self):
        """An adapter built as exact linear combo of basis should have ~0 error."""
        adapters, layers = _make_adapters(n=5)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)

        # Build adapter as exact linear combo of components
        lora_a = {}
        lora_b = {}
        coeffs = torch.randn(3)
        for layer in layers:
            flat_a = coeffs @ sub.components_a[layer] + sub.means_a[layer]
            flat_b = coeffs @ sub.components_b[layer] + sub.means_b[layer]
            lora_a[layer] = flat_a.reshape(sub.shapes_a[layer])
            lora_b[layer] = flat_b.reshape(sub.shapes_b[layer])

        synth = LoRAWeights(
            layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=sub.rank,
        )

        proj = sub.project(synth, "synth")
        sub.add_task(proj)
        reconstructed = sub.reconstruct("synth")

        for layer in layers:
            torch.testing.assert_close(
                synth.lora_a[layer], reconstructed.lora_a[layer],
                atol=1e-4, rtol=1e-4,
            )
            torch.testing.assert_close(
                synth.lora_b[layer], reconstructed.lora_b[layer],
                atol=1e-4, rtol=1e-4,
            )


class TestSerializationRoundtrip:
    def test_save_load_all_state_identical(self):
        """Build subspace, save, load, verify every tensor matches."""
        adapters, layers = _make_adapters(n=4)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            sub.save(tmpdir)
            loaded = SharedSubspace.load(tmpdir)

        assert loaded.rank == sub.rank
        assert loaded.num_components == sub.num_components
        assert loaded.layer_names == sub.layer_names
        assert loaded.adaptive_k == sub.adaptive_k
        assert loaded.variance_threshold == sub.variance_threshold
        assert set(loaded.tasks.keys()) == set(sub.tasks.keys())

        for layer in layers:
            torch.testing.assert_close(
                loaded.components_a[layer], sub.components_a[layer]
            )
            torch.testing.assert_close(
                loaded.components_b[layer], sub.components_b[layer]
            )
            torch.testing.assert_close(
                loaded.means_a[layer], sub.means_a[layer]
            )
            torch.testing.assert_close(
                loaded.means_b[layer], sub.means_b[layer]
            )

        for tid in sub.tasks:
            for layer in layers:
                torch.testing.assert_close(
                    loaded.tasks[tid].loadings_a[layer],
                    sub.tasks[tid].loadings_a[layer],
                )
                torch.testing.assert_close(
                    loaded.tasks[tid].loadings_b[layer],
                    sub.tasks[tid].loadings_b[layer],
                )

    def test_save_load_preserves_shapes(self):
        """shapes_a and shapes_b survive round-trip."""
        adapters, layers = _make_adapters(n=3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            sub.save(tmpdir)
            loaded = SharedSubspace.load(tmpdir)

        for layer in layers:
            assert loaded.shapes_a[layer] == sub.shapes_a[layer]
            assert loaded.shapes_b[layer] == sub.shapes_b[layer]

    def test_quantized_save_load_close(self):
        """save_quantized -> load -> reconstructions should be close."""
        adapters, layers = _make_adapters(n=3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        # Reconstruct before quantized save
        before = sub.reconstruct("task_0")

        with tempfile.TemporaryDirectory() as tmpdir:
            sub.save_quantized(tmpdir)
            loaded = SharedSubspace.load(tmpdir)

        after = loaded.reconstruct("task_0")

        for layer in layers:
            # NF4 introduces quantization error, but should be bounded
            error_a = (before.lora_a[layer] - after.lora_a[layer]).norm()
            orig_a = before.lora_a[layer].norm()
            if orig_a > 1e-6:
                assert (error_a / orig_a).item() < 0.5


class TestProjectWarning:
    def test_warns_on_low_coverage(self):
        """Random adapter unrelated to subspace should trigger warning."""
        adapters, layers = _make_adapters(n=5)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        # Create a random adapter (not in the shared subspace)
        random_adapter = LoRAWeights(
            layer_names=layers,
            lora_a={l: torch.randn(4, 64) * 10 for l in layers},
            lora_b={l: torch.randn(64, 4) * 10 for l in layers},
            rank=4,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sub.project(random_adapter, "random", warn_threshold=0.99)
            # Should warn since random adapter has low coverage
            coverage_warnings = [x for x in w if "Low subspace coverage" in str(x.message)]
            assert len(coverage_warnings) >= 1

    def test_no_warn_on_good_coverage(self):
        """Adapter used to build subspace should not trigger warning."""
        adapters, _ = _make_adapters(n=5)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sub.project(adapters[0], "test", warn_threshold=0.3)
            coverage_warnings = [x for x in w if "Low subspace coverage" in str(x.message)]
            assert len(coverage_warnings) == 0

    def test_no_warn_when_threshold_none(self):
        """Default behavior: no warning regardless of coverage."""
        adapters, layers = _make_adapters(n=5)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        random_adapter = LoRAWeights(
            layer_names=layers,
            lora_a={l: torch.randn(4, 64) * 10 for l in layers},
            lora_b={l: torch.randn(64, 4) * 10 for l in layers},
            rank=4,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sub.project(random_adapter, "random")  # no warn_threshold
            coverage_warnings = [x for x in w if "Low subspace coverage" in str(x.message)]
            assert len(coverage_warnings) == 0


class TestAPICleanup:
    def test_ops_not_in_top_level_all(self):
        """Internal ops should not be in vlora.__all__."""
        import vlora

        removed_ops = [
            "compute_svd", "project_onto_subspace", "reconstruct_from_subspace",
            "gram_schmidt", "explained_variance_ratio", "select_num_components",
            "incremental_svd_update", "NF4_QUANT_TABLE", "nf4_quantize_dequantize",
            "nf4_pack", "nf4_unpack",
        ]
        for name in removed_ops:
            assert name not in vlora.__all__, f"{name} should not be in __all__"

    def test_ops_accessible_via_module(self):
        """Low-level ops should still be importable from vlora.ops."""
        from vlora.ops import (
            NF4_QUANT_TABLE,
            compute_svd,
            gram_schmidt,
        )
        # Verify they're callable/accessible
        assert callable(compute_svd)
        assert callable(gram_schmidt)
        assert isinstance(NF4_QUANT_TABLE, torch.Tensor)

    def test_version_is_1_0_0(self):
        import vlora

        assert vlora.__version__ == "1.0.0"

    def test_public_api_symbols(self):
        """Verify the expected public API is present."""
        import vlora

        expected = {
            "SharedSubspace", "TaskProjection", "LoRAWeights",
            "load_adapter", "load_adapter_from_hub", "save_adapter",
            "init_subspace", "absorb_task", "extract_adapter",
            "compute_similarity_matrix", "find_clusters", "adapter_diff",
            "subspace_coverage", "find_outliers",
            "VLoRAModel", "TaskRouter",
            "SubspaceTrainer", "orthogonal_init",
            "task_arithmetic", "ties_merge", "dare_merge",
        }
        assert set(vlora.__all__) == expected
