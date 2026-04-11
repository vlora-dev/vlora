"""Tests for vlora.analysis — adapter analysis functions."""

import pytest
import torch

from vlora.analysis import (
    adapter_diff,
    compute_similarity_matrix,
    find_clusters,
    subspace_coverage,
)
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


class TestSimilarityMatrix:
    def test_shape(self):
        adapters, _ = _make_adapters(4)
        sim = compute_similarity_matrix(adapters)
        assert sim.shape == (4, 4)

    def test_diagonal_is_one(self):
        adapters, _ = _make_adapters(3)
        sim = compute_similarity_matrix(adapters)
        for i in range(3):
            assert abs(sim[i, i].item() - 1.0) < 1e-5

    def test_symmetric(self):
        adapters, _ = _make_adapters(4)
        sim = compute_similarity_matrix(adapters)
        assert torch.allclose(sim, sim.T, atol=1e-5)

    def test_identical_adapters_have_high_similarity(self):
        adapters, _ = _make_adapters(1)
        # Duplicate the same adapter
        both = [adapters[0], adapters[0]]
        sim = compute_similarity_matrix(both)
        assert sim[0, 1].item() > 0.99

    def test_needs_at_least_two(self):
        adapters, _ = _make_adapters(1)
        with pytest.raises(ValueError):
            compute_similarity_matrix(adapters)


class TestFindClusters:
    def test_identical_adapters_cluster_together(self):
        adapters, _ = _make_adapters(1)
        both = [adapters[0], adapters[0]]
        sim = compute_similarity_matrix(both)
        clusters = find_clusters(sim, threshold=0.99)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_low_threshold_groups_all(self):
        adapters, _ = _make_adapters(4)
        sim = compute_similarity_matrix(adapters)
        clusters = find_clusters(sim, threshold=-1.0)
        assert len(clusters) == 1

    def test_high_threshold_separates(self):
        adapters, _ = _make_adapters(4)
        sim = compute_similarity_matrix(adapters)
        clusters = find_clusters(sim, threshold=0.9999)
        # With random adapters, each should be its own cluster
        assert len(clusters) >= 2


class TestAdapterDiff:
    def test_same_adapter_zero_distance(self):
        adapters, layers = _make_adapters(1)
        diff = adapter_diff(adapters[0], adapters[0])
        for layer in layers:
            assert diff[layer]["l2_distance"] < 1e-5
            assert diff[layer]["cosine_sim"] > 0.99

    def test_different_adapters_have_distance(self):
        adapters, layers = _make_adapters(2)
        diff = adapter_diff(adapters[0], adapters[1])
        for layer in layers:
            assert diff[layer]["l2_distance"] > 0
            assert "cosine_sim" in diff[layer]

    def test_returns_common_layers(self):
        adapters, layers = _make_adapters(2)
        diff = adapter_diff(adapters[0], adapters[1])
        assert set(diff.keys()) == set(layers)


class TestSubspaceCoverage:
    def test_in_subspace_adapter_has_high_coverage(self):
        adapters, _ = _make_adapters(5)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)
        coverage = subspace_coverage(sub, adapters[0])
        for layer, cov in coverage.items():
            # Adapter used to build subspace should be well-covered
            assert cov["coverage_mean"] > 0.5

    def test_returns_all_layers(self):
        adapters, layers = _make_adapters(5)
        sub = SharedSubspace.from_adapters(adapters, num_components=3)
        coverage = subspace_coverage(sub, adapters[0])
        assert set(coverage.keys()) == set(layers)

    def test_coverage_keys(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        coverage = subspace_coverage(sub, adapters[0])
        for layer, cov in coverage.items():
            assert "coverage_a" in cov
            assert "coverage_b" in cov
            assert "coverage_mean" in cov
