"""Tests for vlora.ops — pure math operations."""

import torch
import pytest

from vlora.ops import (
    NF4_QUANT_TABLE,
    compute_svd,
    explained_variance_ratio,
    gram_schmidt,
    nf4_quantize_dequantize,
    project_onto_subspace,
    reconstruct_from_subspace,
    select_num_components,
)


class TestComputeSVD:
    def test_basic_shape(self):
        data = torch.randn(10, 50)
        comps, svals, mean = compute_svd(data, num_components=3)
        assert comps.shape == (3, 50)
        assert svals.shape == (3,)
        assert mean.shape == (50,)

    def test_singular_values_descending(self):
        data = torch.randn(20, 100)
        _, svals, _ = compute_svd(data)
        for i in range(len(svals) - 1):
            assert svals[i] >= svals[i + 1]

    def test_components_orthonormal(self):
        data = torch.randn(10, 50)
        comps, _, _ = compute_svd(data, num_components=5)
        gram = comps @ comps.T
        assert torch.allclose(gram, torch.eye(5), atol=1e-5)

    def test_no_center(self):
        data = torch.randn(10, 50)
        _, _, mean = compute_svd(data, center=False)
        assert torch.all(mean == 0)

    def test_num_components_capped(self):
        data = torch.randn(3, 50)  # Only 3 observations
        comps, svals, _ = compute_svd(data, num_components=10)
        assert comps.shape[0] == 3  # Can't have more than min(N, D)


class TestProjectAndReconstruct:
    def test_roundtrip_perfect_in_subspace(self):
        """Vectors in the subspace should reconstruct perfectly."""
        basis = torch.randn(5, 100)
        # Orthonormalize
        basis, _ = torch.linalg.qr(basis.T)
        basis = basis.T[:5]  # (5, 100)

        # Create a vector that lives in the subspace
        coeffs = torch.randn(5)
        vec = coeffs @ basis

        loadings = project_onto_subspace(vec, basis)
        recon = reconstruct_from_subspace(basis, loadings)
        assert torch.allclose(recon, vec, atol=1e-5)

    def test_batch_projection(self):
        basis = torch.eye(3, 10)  # Simple basis
        batch = torch.randn(5, 10)
        loadings = project_onto_subspace(batch, basis)
        assert loadings.shape == (5, 3)

    def test_reconstruction_error_decreases_with_more_components(self):
        data = torch.randn(10, 50)
        comps_2, _, _ = compute_svd(data, num_components=2)
        comps_5, _, _ = compute_svd(data, num_components=5)

        vec = data[0]
        recon_2 = reconstruct_from_subspace(comps_2, project_onto_subspace(vec, comps_2))
        recon_5 = reconstruct_from_subspace(comps_5, project_onto_subspace(vec, comps_5))

        error_2 = (vec - recon_2).norm()
        error_5 = (vec - recon_5).norm()
        assert error_5 <= error_2 + 1e-6


class TestGramSchmidt:
    def test_expands_basis(self):
        basis = torch.eye(2, 5)  # 2 vectors in 5D
        new = torch.randn(2, 5)
        expanded = gram_schmidt(basis, new)
        assert expanded.shape[0] >= 3  # Should add at least 1

    def test_result_is_orthonormal(self):
        basis = torch.eye(2, 10)
        new = torch.randn(3, 10)
        expanded = gram_schmidt(basis, new)
        gram = expanded @ expanded.T
        assert torch.allclose(gram, torch.eye(expanded.shape[0]), atol=1e-5)

    def test_rejects_redundant_vectors(self):
        basis = torch.eye(3, 5)
        # These are linear combinations of the basis — should be rejected
        redundant = basis[:2] * 2.0
        expanded = gram_schmidt(basis, redundant)
        assert expanded.shape[0] == 3  # No new vectors added


class TestVariance:
    def test_explained_variance_sums_to_one(self):
        svals = torch.tensor([5.0, 3.0, 1.0, 0.5])
        ratios = explained_variance_ratio(svals)
        assert torch.allclose(ratios[-1], torch.tensor(1.0), atol=1e-5)

    def test_select_num_components(self):
        svals = torch.tensor([10.0, 5.0, 1.0, 0.1])
        k = select_num_components(svals, threshold=0.6)
        assert k >= 1
        ratios = explained_variance_ratio(svals)
        assert ratios[k - 1] >= 0.6

    def test_select_returns_all_if_threshold_too_high(self):
        svals = torch.tensor([1.0, 1.0])
        k = select_num_components(svals, threshold=0.99999)
        assert k == 2


class TestNF4Quantization:
    def test_table_has_16_levels(self):
        assert NF4_QUANT_TABLE.shape == (16,)

    def test_table_is_sorted(self):
        for i in range(len(NF4_QUANT_TABLE) - 1):
            assert NF4_QUANT_TABLE[i] < NF4_QUANT_TABLE[i + 1]

    def test_table_range(self):
        assert NF4_QUANT_TABLE[0] == -1.0
        assert NF4_QUANT_TABLE[-1] == 1.0

    def test_output_shape_preserved(self):
        t = torch.randn(4, 8)
        result = nf4_quantize_dequantize(t, block_size=8)
        assert result.shape == t.shape

    def test_output_dtype_preserved(self):
        for dtype in [torch.float32, torch.float64]:
            t = torch.randn(16, dtype=dtype)
            result = nf4_quantize_dequantize(t, block_size=16)
            assert result.dtype == dtype

    def test_zero_tensor_unchanged(self):
        t = torch.zeros(64)
        result = nf4_quantize_dequantize(t, block_size=64)
        assert torch.allclose(result, t, atol=1e-8)

    def test_values_snap_to_nf4_levels(self):
        # A single block — after quantization, normalized values
        # should be exactly NF4 levels.
        t = torch.randn(64)
        result = nf4_quantize_dequantize(t, block_size=64)
        absmax = t.abs().max().clamp(min=1e-10)
        normalized = result / absmax
        table = NF4_QUANT_TABLE
        for val in normalized:
            dists = (val - table).abs()
            assert dists.min() < 1e-5

    def test_small_error_for_normal_data(self):
        torch.manual_seed(42)
        t = torch.randn(1024)
        result = nf4_quantize_dequantize(t, block_size=64)
        # Relative error should be small for normally distributed data
        error = (t - result).norm() / t.norm()
        assert error < 0.15  # < 15% relative error

    def test_nf4_beats_symmetric_int4_for_normal_data(self):
        torch.manual_seed(42)
        t = torch.randn(1024)

        # NF4 error
        nf4_result = nf4_quantize_dequantize(t, block_size=64)
        nf4_error = (t - nf4_result).norm()

        # Symmetric int4 per-tensor error (same as existing quantize)
        qmax = 7
        scale = t.abs().max() / qmax
        sym_result = ((t / scale).round().clamp(-qmax, qmax)) * scale
        sym_error = (t - sym_result).norm()

        assert nf4_error < sym_error

    def test_handles_non_multiple_of_block_size(self):
        # 100 elements with block_size=64 → needs padding
        t = torch.randn(100)
        result = nf4_quantize_dequantize(t, block_size=64)
        assert result.shape == (100,)

    def test_multidimensional_input(self):
        t = torch.randn(3, 5, 64)
        result = nf4_quantize_dequantize(t, block_size=64)
        assert result.shape == (3, 5, 64)

    def test_smaller_block_size_lower_error(self):
        torch.manual_seed(42)
        t = torch.randn(1024)
        err_64 = (t - nf4_quantize_dequantize(t, block_size=64)).norm()
        err_32 = (t - nf4_quantize_dequantize(t, block_size=32)).norm()
        # Smaller blocks should generally give equal or lower error
        assert err_32 <= err_64 + 1e-4
