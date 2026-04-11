"""Adapter merging — task arithmetic, TIES, and DARE.

These techniques operate on LoRA weight matrices directly, producing
a single merged adapter from multiple inputs. All three methods work
on a per-layer basis and return a new LoRAWeights object.

References:
    - Task Arithmetic: Ilharco et al., "Editing Models with Task Arithmetic" (2023)
    - TIES: Yadav et al., "TIES-Merging: Resolving Interference When Merging Models" (2023)
    - DARE: Yu et al., "Language Models are Super Mario" (2024)
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from vlora._validate import check_adapters_compatible
from vlora.io import LoRAWeights

logger = logging.getLogger("vlora")


def task_arithmetic(
    adapters: list[LoRAWeights],
    weights: list[float] | None = None,
) -> LoRAWeights:
    """Merge adapters via weighted average of their weight matrices.

    Each adapter's LoRA A and B matrices are averaged (optionally weighted)
    per-layer to produce a single merged adapter.

    Args:
        adapters: List of adapters to merge (must share layers and rank).
        weights: Per-adapter weights. Defaults to uniform (1/N each).

    Returns:
        Merged LoRAWeights.
    """
    check_adapters_compatible(adapters)
    n = len(adapters)
    if n == 0:
        raise ValueError("Need at least one adapter to merge.")

    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights length ({len(weights)}) must match adapters ({n})")

    # Use intersection of layers
    layer_names = sorted(set.intersection(*(set(a.layer_names) for a in adapters)))
    if not layer_names:
        raise ValueError("Adapters share no common layers.")

    logger.info("Task arithmetic merge: %d adapters, %d layers", n, len(layer_names))

    lora_a: dict[str, Tensor] = {}
    lora_b: dict[str, Tensor] = {}

    for layer in layer_names:
        merged_a = sum(w * adapters[i].lora_a[layer] for i, w in enumerate(weights))
        merged_b = sum(w * adapters[i].lora_b[layer] for i, w in enumerate(weights))
        lora_a[layer] = merged_a
        lora_b[layer] = merged_b

    return LoRAWeights(
        layer_names=layer_names,
        lora_a=lora_a,
        lora_b=lora_b,
        rank=adapters[0].rank,
    )


def ties_merge(
    adapters: list[LoRAWeights],
    density: float = 0.5,
    weights: list[float] | None = None,
) -> LoRAWeights:
    """Merge adapters using TIES: Trim, Elect sign, and merge.

    1. Trim: zero out the smallest elements per adapter (keep top `density` fraction)
    2. Elect sign: for each position, choose the majority sign across adapters
    3. Merge: average only the values that agree with the elected sign

    Args:
        adapters: List of adapters to merge.
        density: Fraction of elements to keep per adapter (0, 1]. Default 0.5.
        weights: Per-adapter weights for the final average.

    Returns:
        Merged LoRAWeights.
    """
    check_adapters_compatible(adapters)
    n = len(adapters)
    if n == 0:
        raise ValueError("Need at least one adapter to merge.")
    if not 0 < density <= 1:
        raise ValueError(f"density must be in (0, 1], got {density}")

    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights length ({len(weights)}) must match adapters ({n})")

    layer_names = sorted(set.intersection(*(set(a.layer_names) for a in adapters)))
    if not layer_names:
        raise ValueError("Adapters share no common layers.")

    logger.info("TIES merge: %d adapters, density=%.2f, %d layers", n, density, len(layer_names))

    lora_a: dict[str, Tensor] = {}
    lora_b: dict[str, Tensor] = {}

    for layer in layer_names:
        for side, out_dict, attr in [("a", lora_a, "lora_a"), ("b", lora_b, "lora_b")]:
            # Stack all adapters for this layer/side
            tensors = [getattr(adapters[i], attr)[layer].clone() for i in range(n)]

            # Step 1: Trim — zero out smallest elements per adapter
            for t in tensors:
                flat = t.flatten().abs()
                k = max(1, int(density * flat.numel()))
                threshold = flat.topk(k).values[-1]
                t[t.abs() < threshold] = 0.0

            stacked = torch.stack(tensors)  # (N, *shape)

            # Step 2: Elect sign — majority vote at each position
            sign_votes = (stacked > 0).float().sum(dim=0) - (stacked < 0).float().sum(dim=0)
            elected_sign = sign_votes.sign()
            # Ties go positive (convention)
            elected_sign[elected_sign == 0] = 1.0

            # Step 3: Merge — weighted average of values matching elected sign
            mask = (stacked.sign() == elected_sign.unsqueeze(0))
            # Apply weights
            w = torch.tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
            weighted = stacked * w
            # Zero out values with wrong sign
            weighted = weighted * mask.float()
            # Average over contributors that match elected sign
            contributor_count = mask.float().sum(dim=0).clamp(min=1)
            merged = weighted.sum(dim=0) / contributor_count

            out_dict[layer] = merged

    return LoRAWeights(
        layer_names=layer_names,
        lora_a=lora_a,
        lora_b=lora_b,
        rank=adapters[0].rank,
    )


def dare_merge(
    adapters: list[LoRAWeights],
    drop_rate: float = 0.5,
    weights: list[float] | None = None,
    seed: int | None = None,
) -> LoRAWeights:
    """Merge adapters using DARE: Drop And REscale.

    For each adapter, randomly drop elements with probability `drop_rate`
    and rescale survivors by 1/(1-drop_rate). Then average the results.

    Args:
        adapters: List of adapters to merge.
        drop_rate: Probability of dropping each element. Default 0.5.
        weights: Per-adapter weights for the final average.
        seed: Random seed for reproducibility.

    Returns:
        Merged LoRAWeights.
    """
    check_adapters_compatible(adapters)
    n = len(adapters)
    if n == 0:
        raise ValueError("Need at least one adapter to merge.")
    if not 0 <= drop_rate < 1:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")

    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"weights length ({len(weights)}) must match adapters ({n})")

    layer_names = sorted(set.intersection(*(set(a.layer_names) for a in adapters)))
    if not layer_names:
        raise ValueError("Adapters share no common layers.")

    logger.info("DARE merge: %d adapters, drop_rate=%.2f, %d layers", n, drop_rate, len(layer_names))

    if seed is not None:
        torch.manual_seed(seed)

    rescale = 1.0 / (1.0 - drop_rate) if drop_rate > 0 else 1.0

    lora_a: dict[str, Tensor] = {}
    lora_b: dict[str, Tensor] = {}

    for layer in layer_names:
        for side, out_dict, attr in [("a", lora_a, "lora_a"), ("b", lora_b, "lora_b")]:
            merged = torch.zeros_like(getattr(adapters[0], attr)[layer])

            for i, adapter in enumerate(adapters):
                t = getattr(adapter, attr)[layer].clone()
                if drop_rate > 0:
                    mask = torch.bernoulli(torch.full_like(t, 1.0 - drop_rate))
                    t = t * mask * rescale
                merged = merged + weights[i] * t

            out_dict[layer] = merged

    return LoRAWeights(
        layer_names=layer_names,
        lora_a=lora_a,
        lora_b=lora_b,
        rank=adapters[0].rank,
    )


MERGE_METHODS = {
    "average": task_arithmetic,
    "ties": ties_merge,
    "dare": dare_merge,
}
