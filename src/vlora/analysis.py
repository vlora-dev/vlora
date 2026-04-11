"""Adapter analysis — similarity, clustering, diffing, and coverage."""

from __future__ import annotations

__all__ = [
    "adapter_diff",
    "compute_similarity_matrix",
    "find_clusters",
    "find_outliers",
    "subspace_coverage",
]

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from vlora.io import LoRAWeights
from vlora.ops import project_onto_subspace

if TYPE_CHECKING:
    from vlora.subspace import SharedSubspace


def _flatten_adapter(adapter: LoRAWeights) -> Tensor:
    """Flatten all A and B weights into a single 1D vector."""
    parts = []
    for layer in adapter.layer_names:
        parts.append(adapter.lora_a[layer].flatten())
        parts.append(adapter.lora_b[layer].flatten())
    return torch.cat(parts)


def compute_similarity_matrix(adapters: list[LoRAWeights]) -> Tensor:
    """Compute pairwise cosine similarity between adapters.

    Returns:
        (N, N) similarity matrix where entry [i,j] is the cosine
        similarity between adapter i and adapter j.
    """
    if len(adapters) < 2:
        raise ValueError("Need at least 2 adapters for similarity")

    vectors = torch.stack([_flatten_adapter(a) for a in adapters])
    # Normalize rows
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = vectors / norms
    return normalized @ normalized.T


def find_clusters(
    similarity_matrix: Tensor,
    threshold: float = 0.9,
) -> list[list[int]]:
    """Group adapters into clusters based on similarity threshold.

    Uses simple greedy clustering: iterate through adapters, assign each
    to the first cluster where similarity to all members >= threshold,
    or create a new cluster.

    Returns:
        List of clusters, where each cluster is a list of adapter indices.
    """
    n = similarity_matrix.shape[0]
    clusters: list[list[int]] = []

    for i in range(n):
        placed = False
        for cluster in clusters:
            # Check similarity to all members
            if all(similarity_matrix[i, j].item() >= threshold for j in cluster):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    return clusters


def adapter_diff(
    adapter_a: LoRAWeights,
    adapter_b: LoRAWeights,
) -> dict[str, dict[str, float]]:
    """Per-layer comparison of two adapters.

    Returns:
        Dict mapping layer_name -> {"l2_distance": float, "cosine_sim": float}
        for each layer present in both adapters.
    """
    common_layers = sorted(set(adapter_a.layer_names) & set(adapter_b.layer_names))
    result: dict[str, dict[str, float]] = {}

    for layer in common_layers:
        vec_a = torch.cat([
            adapter_a.lora_a[layer].flatten(),
            adapter_a.lora_b[layer].flatten(),
        ])
        vec_b = torch.cat([
            adapter_b.lora_a[layer].flatten(),
            adapter_b.lora_b[layer].flatten(),
        ])

        l2 = (vec_a - vec_b).norm().item()
        cos = torch.nn.functional.cosine_similarity(
            vec_a.unsqueeze(0), vec_b.unsqueeze(0)
        ).item()

        result[layer] = {"l2_distance": l2, "cosine_sim": cos}

    return result


def subspace_coverage(
    subspace: SharedSubspace,
    adapter: LoRAWeights,
) -> dict[str, dict[str, float]]:
    """Measure how well a subspace represents a given adapter.

    For each layer and side (A/B), projects the adapter onto the subspace
    and measures the fraction of the adapter's norm that is captured.

    Returns:
        Dict mapping layer_name -> {"coverage_a": float, "coverage_b": float,
        "coverage_mean": float}
    """
    result: dict[str, dict[str, float]] = {}

    for layer in subspace.layer_names:
        if layer not in adapter.lora_a:
            continue

        coverages = {}
        for side, weights_dict, components, means in [
            ("a", adapter.lora_a, subspace.components_a, subspace.means_a),
            ("b", adapter.lora_b, subspace.components_b, subspace.means_b),
        ]:
            flat = weights_dict[layer].flatten() - means[layer]
            original_norm = flat.norm().item()
            if original_norm < 1e-8:
                coverages[f"coverage_{side}"] = 1.0
                continue

            loadings = project_onto_subspace(flat, components[layer])
            reconstructed = loadings @ components[layer]
            residual_norm = (flat - reconstructed).norm().item()
            coverages[f"coverage_{side}"] = 1.0 - (residual_norm / original_norm)

        coverages["coverage_mean"] = (
            coverages["coverage_a"] + coverages["coverage_b"]
        ) / 2.0
        result[layer] = coverages

    return result


def find_outliers(
    adapters: list[LoRAWeights],
    threshold: float = 2.0,
) -> list[dict]:
    """Detect adapter outliers based on distance from the group mean.

    Computes each adapter's flattened weight vector, measures the L2
    distance from the group centroid, and flags adapters whose distance
    exceeds `threshold` standard deviations above the mean distance.

    Args:
        adapters: List of adapters to analyze.
        threshold: Number of standard deviations above mean distance
            to consider an outlier. Default 2.0.

    Returns:
        List of dicts with keys: {"index", "distance", "z_score"} for
        each outlier adapter.
    """
    if len(adapters) < 3:
        return []

    vectors = torch.stack([_flatten_adapter(a) for a in adapters])
    centroid = vectors.mean(dim=0)
    distances = (vectors - centroid).norm(dim=1)

    mean_dist = distances.mean().item()
    std_dist = distances.std().item()

    if std_dist < 1e-8:
        return []

    outliers = []
    for i in range(len(adapters)):
        z = (distances[i].item() - mean_dist) / std_dist
        if z > threshold:
            outliers.append({
                "index": i,
                "distance": distances[i].item(),
                "z_score": z,
            })

    return outliers
