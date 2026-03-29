"""SharedSubspace — core state container and 3-step algorithm.

Step 1: from_adapters  — build shared basis via SVD
Step 2: project        — project new adapter onto basis
Step 3: absorb         — incorporate new adapter, recompute basis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger("vlora")
from safetensors.torch import load_file, save_file
from torch import Tensor

from vlora._validate import (
    check_adapter_matches_subspace,
    check_adapters_compatible,
    check_task_exists,
    check_tensor_health,
)
from vlora.io import LoRAWeights, stack_lora_weights
from vlora.ops import (
    compute_svd,
    explained_variance_ratio,
    gram_schmidt,
    incremental_svd_update,
    nf4_quantize_dequantize,
    project_onto_subspace,
    reconstruct_from_subspace,
    select_num_components,
)


@dataclass
class TaskProjection:
    """A single task's representation in the shared subspace."""

    task_id: str
    loadings_a: dict[str, Tensor]  # layer_name -> (k,)
    loadings_b: dict[str, Tensor]  # layer_name -> (k,)


class SharedSubspace:
    """Shared low-rank subspace for LoRA adapters.

    Maintains per-layer orthonormal basis vectors (components) and
    per-task coefficient vectors (loadings).
    """

    def __init__(
        self,
        layer_names: list[str],
        components_a: dict[str, Tensor],
        components_b: dict[str, Tensor],
        singular_values_a: dict[str, Tensor],
        singular_values_b: dict[str, Tensor],
        means_a: dict[str, Tensor],
        means_b: dict[str, Tensor],
        tasks: dict[str, TaskProjection],
        rank: int,
        num_components: int,
    ):
        self.layer_names = layer_names
        self.components_a = components_a
        self.components_b = components_b
        self.singular_values_a = singular_values_a
        self.singular_values_b = singular_values_b
        self.means_a = means_a
        self.means_b = means_b
        self.tasks = tasks
        self.rank = rank
        self.num_components = num_components

    @classmethod
    def from_adapters(
        cls,
        adapters: list[LoRAWeights],
        task_ids: list[str] | None = None,
        variance_threshold: float = 0.6,
        num_components: int | None = None,
        adaptive_k: bool = False,
    ) -> SharedSubspace:
        """Step 1: Build shared subspace from existing adapters.

        Stacks each adapter's flattened weights, runs SVD per layer,
        and projects all adapters onto the resulting basis.

        Args:
            adapters: List of LoRA adapters to initialize from.
            task_ids: Names for each adapter. Defaults to "task_0", "task_1", etc.
            variance_threshold: Minimum cumulative variance to explain (used if
                num_components is None).
            num_components: Explicit number of basis vectors per layer.
                Overrides variance_threshold if set.
            adaptive_k: If True, select k independently per layer based on
                variance_threshold. Each layer gets the minimal k that explains
                the threshold. Overrides num_components.
        """
        check_adapters_compatible(adapters)
        logger.info("Building subspace from %d adapters", len(adapters))

        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(len(adapters))]
        if len(task_ids) != len(adapters):
            raise ValueError("task_ids length must match adapters length")

        # Intersect layer names across all adapters for safety
        layer_set = set(adapters[0].layer_names)
        for adapter in adapters[1:]:
            layer_set &= set(adapter.layer_names)
        layer_names = sorted(layer_set)

        if not layer_names:
            raise ValueError("Adapters share no common layers")

        if len(layer_names) < len(adapters[0].layer_names):
            import warnings
            dropped = set(adapters[0].layer_names) - layer_set
            warnings.warn(
                f"Adapters have different layer sets. Using {len(layer_names)} "
                f"common layers (dropped {len(dropped)}: {sorted(dropped)[:3]}...)"
            )

        rank = adapters[0].rank

        # Stack weights into data matrices
        stacked_a = stack_lora_weights(adapters, side="A")
        stacked_b = stack_lora_weights(adapters, side="B")

        components_a: dict[str, Tensor] = {}
        components_b: dict[str, Tensor] = {}
        sv_a: dict[str, Tensor] = {}
        sv_b: dict[str, Tensor] = {}
        means_a: dict[str, Tensor] = {}
        means_b: dict[str, Tensor] = {}

        resolved_k: int | None = None

        for layer in layer_names:
            for side, stacked, comp_dict, sv_dict, mean_dict in [
                ("A", stacked_a, components_a, sv_a, means_a),
                ("B", stacked_b, components_b, sv_b, means_b),
            ]:
                data = stacked[layer]
                comps, svals, mean = compute_svd(data, num_components=None, center=True)
                check_tensor_health(comps, f"{layer}.components_{side.lower()}")
                check_tensor_health(mean, f"{layer}.mean_{side.lower()}")

                if adaptive_k:
                    # Per-layer: each layer/side gets its own k
                    k = select_num_components(svals, variance_threshold)
                elif num_components is not None:
                    k = min(num_components, len(svals))
                else:
                    k = select_num_components(svals, variance_threshold)

                if not adaptive_k:
                    if resolved_k is None:
                        resolved_k = k
                    # Use consistent k across layers for simplicity
                    k = resolved_k

                comp_dict[layer] = comps[:k]
                sv_dict[layer] = svals[:k]
                mean_dict[layer] = mean

        # For adaptive_k, use the max per-layer k as the reported num_components
        if adaptive_k:
            resolved_k = max(
                max(components_a[l].shape[0], components_b[l].shape[0])
                for l in layer_names
            )
        resolved_k = resolved_k or 1

        # Project all input adapters onto the basis
        tasks: dict[str, TaskProjection] = {}
        for i, (adapter, tid) in enumerate(zip(adapters, task_ids)):
            loadings_a: dict[str, Tensor] = {}
            loadings_b: dict[str, Tensor] = {}
            for layer in layer_names:
                wa = adapter.lora_a[layer].flatten() - means_a[layer]
                wb = adapter.lora_b[layer].flatten() - means_b[layer]
                loadings_a[layer] = project_onto_subspace(wa, components_a[layer])
                loadings_b[layer] = project_onto_subspace(wb, components_b[layer])
            tasks[tid] = TaskProjection(
                task_id=tid, loadings_a=loadings_a, loadings_b=loadings_b
            )

        logger.info(
            "Subspace built: k=%d, layers=%d, tasks=%d, rank=%d",
            resolved_k, len(layer_names), len(tasks), rank,
        )

        return cls(
            layer_names=layer_names,
            components_a=components_a,
            components_b=components_b,
            singular_values_a=sv_a,
            singular_values_b=sv_b,
            means_a=means_a,
            means_b=means_b,
            tasks=tasks,
            rank=rank,
            num_components=resolved_k,
        )

    def project(self, adapter: LoRAWeights, task_id: str) -> TaskProjection:
        """Step 2a: Project a new adapter onto the existing basis."""
        check_adapter_matches_subspace(adapter, self, "project")
        loadings_a: dict[str, Tensor] = {}
        loadings_b: dict[str, Tensor] = {}

        for layer in self.layer_names:
            wa = adapter.lora_a[layer].flatten() - self.means_a[layer]
            wb = adapter.lora_b[layer].flatten() - self.means_b[layer]
            loadings_a[layer] = project_onto_subspace(wa, self.components_a[layer])
            loadings_b[layer] = project_onto_subspace(wb, self.components_b[layer])

        return TaskProjection(
            task_id=task_id, loadings_a=loadings_a, loadings_b=loadings_b
        )

    def add_task(self, projection: TaskProjection) -> None:
        """Register a projected task in the subspace."""
        self.tasks[projection.task_id] = projection

    def reconstruct(self, task_id: str) -> LoRAWeights:
        """Reconstruct full LoRA weights for a task from its loadings."""
        check_task_exists(self, task_id)

        proj = self.tasks[task_id]
        lora_a: dict[str, Tensor] = {}
        lora_b: dict[str, Tensor] = {}

        for layer in self.layer_names:
            flat_a = reconstruct_from_subspace(
                self.components_a[layer], proj.loadings_a[layer]
            ) + self.means_a[layer]
            flat_b = reconstruct_from_subspace(
                self.components_b[layer], proj.loadings_b[layer]
            ) + self.means_b[layer]

            # Recover original matrix shapes from the adapter's rank
            # A: (rank, in_features), B: (out_features, rank)
            ref_a_shape = (self.rank, flat_a.numel() // self.rank)
            ref_b_shape = (flat_b.numel() // self.rank, self.rank)
            lora_a[layer] = flat_a.reshape(ref_a_shape)
            lora_b[layer] = flat_b.reshape(ref_b_shape)

        return LoRAWeights(
            layer_names=self.layer_names,
            lora_a=lora_a,
            lora_b=lora_b,
            rank=self.rank,
        )

    def absorb(self, new_adapter: LoRAWeights, new_task_id: str) -> None:
        """Step 3: Absorb a new adapter, recomputing the shared basis.

        Reconstructs all existing tasks, adds the new adapter, then
        reruns SVD to produce an updated basis.
        """
        check_adapter_matches_subspace(new_adapter, self, "absorb")
        if new_task_id in self.tasks:
            import warnings
            warnings.warn(
                f"Task '{new_task_id}' already exists and will be overwritten by absorb.",
                stacklevel=2,
            )
        logger.info("Absorbing adapter '%s' (full SVD recompute, %d existing tasks)", new_task_id, len(self.tasks))
        # Reconstruct all existing tasks as full adapters
        all_adapters = []
        all_ids = []
        for tid, _ in self.tasks.items():
            all_adapters.append(self.reconstruct(tid))
            all_ids.append(tid)

        all_adapters.append(new_adapter)
        all_ids.append(new_task_id)

        # Rebuild subspace from scratch
        new_sub = SharedSubspace.from_adapters(
            all_adapters,
            task_ids=all_ids,
            num_components=self.num_components,
        )

        # Update self in-place
        self.layer_names = new_sub.layer_names
        self.components_a = new_sub.components_a
        self.components_b = new_sub.components_b
        self.singular_values_a = new_sub.singular_values_a
        self.singular_values_b = new_sub.singular_values_b
        self.means_a = new_sub.means_a
        self.means_b = new_sub.means_b
        self.tasks = new_sub.tasks
        self.num_components = new_sub.num_components

    def absorb_incremental(self, new_adapter: LoRAWeights, new_task_id: str) -> None:
        """Absorb a new adapter incrementally without full SVD recompute.

        Instead of reconstructing all tasks and re-running SVD, this projects
        the new adapter onto the existing basis, measures the residual, and
        expands the basis with any significant new directions.

        Existing tasks are properly re-projected onto the updated basis
        by reconstructing their weights from the old basis first.

        Much faster than absorb() for large collections, with a small
        approximation trade-off.
        """
        check_adapter_matches_subspace(new_adapter, self, "absorb_incremental")
        if new_task_id in self.tasks:
            import warnings
            warnings.warn(
                f"Task '{new_task_id}' already exists and will be overwritten by absorb_incremental.",
                stacklevel=2,
            )
        logger.debug("Absorbing adapter '%s' incrementally", new_task_id)
        loadings_a: dict[str, Tensor] = {}
        loadings_b: dict[str, Tensor] = {}

        # Snapshot old basis per layer so we can reconstruct existing tasks
        old_comps_a = {l: self.components_a[l].clone() for l in self.layer_names}
        old_comps_b = {l: self.components_b[l].clone() for l in self.layer_names}
        old_means_a = {l: self.means_a[l].clone() for l in self.layer_names}
        old_means_b = {l: self.means_b[l].clone() for l in self.layer_names}

        for layer in self.layer_names:
            for side, weights_dict, comp_attr, sv_attr, mean_attr, load_dict in [
                ("a", new_adapter.lora_a, "components_a", "singular_values_a", "means_a", loadings_a),
                ("b", new_adapter.lora_b, "components_b", "singular_values_b", "means_b", loadings_b),
            ]:
                components = getattr(self, comp_attr)[layer]
                svals = getattr(self, sv_attr)[layer]
                mean = getattr(self, mean_attr)[layer]
                flat = weights_dict[layer].flatten().unsqueeze(0)  # (1, D)

                new_comps, new_svals, new_mean, _ = incremental_svd_update(
                    components, svals, mean,
                    n_seen=len(self.tasks),
                    new_data=flat,
                    max_components=self.num_components,
                )

                getattr(self, comp_attr)[layer] = new_comps
                getattr(self, sv_attr)[layer] = new_svals
                getattr(self, mean_attr)[layer] = new_mean

                # Project new adapter with updated basis
                centered = flat.squeeze(0) - new_mean
                load_dict[layer] = project_onto_subspace(centered, new_comps)

        # Re-project existing tasks: reconstruct from OLD basis, project onto NEW
        for tid, proj in self.tasks.items():
            for layer in self.layer_names:
                for old_comps, old_means, old_loads, comp_attr, mean_attr in [
                    (old_comps_a, old_means_a, proj.loadings_a, "components_a", "means_a"),
                    (old_comps_b, old_means_b, proj.loadings_b, "components_b", "means_b"),
                ]:
                    # Reconstruct flat weights from old basis
                    flat_weights = reconstruct_from_subspace(
                        old_comps[layer], old_loads[layer]
                    ) + old_means[layer]
                    # Re-project onto updated basis
                    new_comps = getattr(self, comp_attr)[layer]
                    new_mean = getattr(self, mean_attr)[layer]
                    old_loads[layer] = project_onto_subspace(
                        flat_weights - new_mean, new_comps
                    )

        self.tasks[new_task_id] = TaskProjection(
            task_id=new_task_id, loadings_a=loadings_a, loadings_b=loadings_b
        )

    @classmethod
    def from_adapters_streaming(
        cls,
        adapter_paths: list[str | Path],
        task_ids: list[str] | None = None,
        num_components: int = 4,
    ) -> SharedSubspace:
        """Build a subspace by streaming adapters one at a time from disk.

        Only loads one adapter into memory at a time, unlike from_adapters
        which loads all simultaneously. Uses incremental SVD updates.

        Args:
            adapter_paths: Paths to adapter directories on disk.
            task_ids: Names for each adapter.
            num_components: Number of basis components.
        """
        from vlora.io import load_adapter

        if not adapter_paths:
            raise ValueError("Need at least one adapter path")

        paths = [Path(p) for p in adapter_paths]
        if task_ids is None:
            task_ids = [p.name for p in paths]
        if len(task_ids) != len(paths):
            raise ValueError(
                f"task_ids length ({len(task_ids)}) must match "
                f"adapter_paths length ({len(paths)})"
            )

        # Initialize from first adapter(s) — use first two if available
        # so SVD has enough samples to find >1 component
        if len(paths) >= 2:
            init_adapters = [load_adapter(paths[0]), load_adapter(paths[1])]
            init_ids = task_ids[:2]
            remaining = list(zip(paths[2:], task_ids[2:]))
        else:
            init_adapters = [load_adapter(paths[0])]
            init_ids = [task_ids[0]]
            remaining = []

        sub = cls.from_adapters(init_adapters, task_ids=init_ids, num_components=num_components)
        # Ensure target num_components is preserved even if initial SVD
        # had fewer samples than requested components
        sub.num_components = num_components

        # Stream remaining adapters
        for path, tid in remaining:
            adapter = load_adapter(path)
            sub.absorb_incremental(adapter, tid)

        return sub

    def to(self, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> SharedSubspace:
        """Move all tensors to a device and/or dtype. Returns self."""
        for layer in self.layer_names:
            for attr in ["components_a", "components_b", "singular_values_a",
                         "singular_values_b", "means_a", "means_b"]:
                d = getattr(self, attr)
                t = d[layer]
                if device is not None:
                    t = t.to(device=device)
                if dtype is not None:
                    t = t.to(dtype=dtype)
                d[layer] = t

        for proj in self.tasks.values():
            for layer in self.layer_names:
                for loads in [proj.loadings_a, proj.loadings_b]:
                    t = loads[layer]
                    if device is not None:
                        t = t.to(device=device)
                    if dtype is not None:
                        t = t.to(dtype=dtype)
                    loads[layer] = t

        return self

    def quantize(
        self,
        bits: int = 8,
        method: Literal["symmetric", "nf4"] = "symmetric",
        block_size: int = 64,
        quantize_loadings: bool = False,
    ) -> SharedSubspace:
        """Quantize components (and optionally loadings) to reduce memory.

        Two methods are available:

        - ``"symmetric"`` (default): uniform per-tensor quantization at the
          specified bit width (4 or 8). Simple and fast.
        - ``"nf4"``: 4-bit NormalFloat quantization from QLoRA (Dettmers et
          al., 2023). Uses 16 quantile levels optimized for normally-
          distributed data with per-block absmax scaling. Lower error than
          symmetric int4 when weights are approximately normal.

        This is a lossy, in-place operation.

        Args:
            bits: Bit width for symmetric quantization (4 or 8). Ignored
                when method is ``"nf4"`` (always 4-bit).
            method: Quantization method — ``"symmetric"`` or ``"nf4"``.
            block_size: Elements per quantization block for NF4. Smaller
                blocks give finer granularity. Default 64. Ignored for
                symmetric quantization.
            quantize_loadings: If True, also quantize per-task loadings
                (not just components). Loadings from dot products of
                normal-ish weights tend to be approximately normal, so
                NF4 is especially effective here.

        Returns:
            self (modified in-place).
        """
        if method == "symmetric":
            if bits not in (4, 8):
                raise ValueError(f"bits must be 4 or 8, got {bits}")
            self._quantize_symmetric(bits, quantize_loadings)
        elif method == "nf4":
            self._quantize_nf4(block_size, quantize_loadings)
        else:
            raise ValueError(f"method must be 'symmetric' or 'nf4', got {method!r}")

        return self

    def _quantize_symmetric(self, bits: int, quantize_loadings: bool) -> None:
        """Symmetric per-tensor quantization."""
        qmax = (1 << (bits - 1)) - 1  # 127 for int8, 7 for int4

        for layer in self.layer_names:
            for attr in ["components_a", "components_b"]:
                d = getattr(self, attr)
                t = d[layer].float()
                scale = t.abs().max() / qmax
                if scale == 0:
                    continue
                quantized = (t / scale).round().clamp(-qmax, qmax)
                d[layer] = (quantized * scale).to(t.dtype)

        if quantize_loadings:
            for proj in self.tasks.values():
                for loads in [proj.loadings_a, proj.loadings_b]:
                    for layer in self.layer_names:
                        t = loads[layer].float()
                        scale = t.abs().max() / qmax
                        if scale == 0:
                            continue
                        quantized = (t / scale).round().clamp(-qmax, qmax)
                        loads[layer] = (quantized * scale).to(t.dtype)

    def _quantize_nf4(self, block_size: int, quantize_loadings: bool) -> None:
        """NF4 per-block quantization (QLoRA-style)."""
        for layer in self.layer_names:
            for attr in ["components_a", "components_b"]:
                d = getattr(self, attr)
                d[layer] = nf4_quantize_dequantize(d[layer], block_size)

        if quantize_loadings:
            for proj in self.tasks.values():
                for loads in [proj.loadings_a, proj.loadings_b]:
                    for layer in self.layer_names:
                        loads[layer] = nf4_quantize_dequantize(
                            loads[layer], block_size
                        )

    def compression_stats(self) -> dict:
        """Compute compression statistics for the current subspace.

        Returns a dict with per-layer and aggregate stats including:
        - components_per_layer: dict of layer -> (k_a, k_b)
        - total_params: total parameters in compressed representation
        - total_original: estimated original parameters (N adapters)
        - compression_ratio: original / compressed
        """
        n_tasks = len(self.tasks)
        total_compressed = 0
        total_original = 0
        per_layer = {}

        for layer in self.layer_names:
            k_a = self.components_a[layer].shape[0]
            k_b = self.components_b[layer].shape[0]
            dim_a = self.components_a[layer].shape[1]
            dim_b = self.components_b[layer].shape[1]

            # Compressed: components + means + per-task loadings
            layer_compressed = (
                k_a * dim_a + k_b * dim_b  # components
                + dim_a + dim_b  # means
                + n_tasks * (k_a + k_b)  # loadings
            )
            # Original: N full adapter matrices
            layer_original = n_tasks * (dim_a + dim_b)

            per_layer[layer] = {
                "k_a": k_a, "k_b": k_b,
                "compressed": layer_compressed,
                "original": layer_original,
            }
            total_compressed += layer_compressed
            total_original += layer_original

        return {
            "components_per_layer": {l: (d["k_a"], d["k_b"]) for l, d in per_layer.items()},
            "total_params_compressed": total_compressed,
            "total_params_original": total_original,
            "compression_ratio": total_original / total_compressed if total_compressed > 0 else 0,
            "num_tasks": n_tasks,
            "num_layers": len(self.layer_names),
        }

    def get_trainable_params(
        self, task_id: str, num_expand: int = 0
    ) -> dict[str, Tensor]:
        """Get trainable loading parameters for a task.

        Useful for integrating with a training loop: freeze the components,
        train only the loadings.

        Args:
            task_id: Task whose loadings to return.
            num_expand: Number of extra orthogonal directions to add via
                Gram-Schmidt (gives the optimizer room to escape the subspace).

        Returns:
            Dict of parameter name -> tensor (with requires_grad=True).
        """
        if num_expand > 0:
            import warnings
            warnings.warn(
                f"get_trainable_params(num_expand={num_expand}) will permanently "
                "expand the subspace basis via Gram-Schmidt. This modifies the "
                "subspace in-place and cannot be undone.",
                stacklevel=2,
            )
            for layer in self.layer_names:
                random_a = torch.randn(num_expand, self.components_a[layer].shape[1])
                random_b = torch.randn(num_expand, self.components_b[layer].shape[1])
                self.components_a[layer] = gram_schmidt(self.components_a[layer], random_a)
                self.components_b[layer] = gram_schmidt(self.components_b[layer], random_b)

            # Re-project the task onto the expanded basis
            proj = self.tasks.get(task_id)
            if proj is not None:
                for layer in self.layer_names:
                    old_k_a = proj.loadings_a[layer].shape[0]
                    new_k_a = self.components_a[layer].shape[0]
                    if new_k_a > old_k_a:
                        proj.loadings_a[layer] = torch.cat([
                            proj.loadings_a[layer],
                            torch.zeros(new_k_a - old_k_a),
                        ])
                    old_k_b = proj.loadings_b[layer].shape[0]
                    new_k_b = self.components_b[layer].shape[0]
                    if new_k_b > old_k_b:
                        proj.loadings_b[layer] = torch.cat([
                            proj.loadings_b[layer],
                            torch.zeros(new_k_b - old_k_b),
                        ])

        if task_id not in self.tasks:
            raise KeyError(f"Unknown task: {task_id}")

        params = {}
        proj = self.tasks[task_id]
        for layer in self.layer_names:
            la = proj.loadings_a[layer].clone().detach().requires_grad_(True)
            lb = proj.loadings_b[layer].clone().detach().requires_grad_(True)
            params[f"{layer}.loadings_a"] = la
            params[f"{layer}.loadings_b"] = lb

        return params

    @staticmethod
    def _safe_filename(task_id: str) -> str:
        """Convert a task ID to a filesystem-safe filename component."""
        import re
        return re.sub(r'[^\w\-.]', '_', task_id)

    def save(self, path: str | Path) -> None:
        """Serialize the subspace to disk."""
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save components and means (contiguous() needed for safetensors)
        tensors = {}
        for layer in self.layer_names:
            tensors[f"{layer}.components_a"] = self.components_a[layer].contiguous()
            tensors[f"{layer}.components_b"] = self.components_b[layer].contiguous()
            tensors[f"{layer}.sv_a"] = self.singular_values_a[layer].contiguous()
            tensors[f"{layer}.sv_b"] = self.singular_values_b[layer].contiguous()
            tensors[f"{layer}.mean_a"] = self.means_a[layer].contiguous()
            tensors[f"{layer}.mean_b"] = self.means_b[layer].contiguous()

        save_file(tensors, str(path / "subspace.safetensors"))

        # Save per-task loadings (with sanitized filenames)
        tid_to_filename: dict[str, str] = {}
        for tid, proj in self.tasks.items():
            safe_name = self._safe_filename(tid)
            tid_to_filename[tid] = safe_name
            task_tensors = {}
            for layer in self.layer_names:
                task_tensors[f"{layer}.loadings_a"] = proj.loadings_a[layer].contiguous()
                task_tensors[f"{layer}.loadings_b"] = proj.loadings_b[layer].contiguous()
            save_file(task_tensors, str(path / f"task_{safe_name}.safetensors"))

        # Save metadata (includes filename mapping for safe round-trip)
        meta = {
            "layer_names": self.layer_names,
            "task_ids": list(self.tasks.keys()),
            "task_filenames": tid_to_filename,
            "rank": self.rank,
            "num_components": self.num_components,
        }
        with open(path / "subspace_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SharedSubspace:
        """Deserialize a subspace from disk."""
        import json

        path = Path(path)

        with open(path / "subspace_meta.json") as f:
            meta = json.load(f)

        layer_names = meta["layer_names"]
        task_ids = meta["task_ids"]
        rank = meta["rank"]
        num_components = meta["num_components"]
        # Support both old format (no mapping) and new format
        tid_to_filename = meta.get("task_filenames", {})

        tensors = load_file(str(path / "subspace.safetensors"))
        components_a = {l: tensors[f"{l}.components_a"] for l in layer_names}
        components_b = {l: tensors[f"{l}.components_b"] for l in layer_names}
        sv_a = {l: tensors[f"{l}.sv_a"] for l in layer_names}
        sv_b = {l: tensors[f"{l}.sv_b"] for l in layer_names}
        means_a = {l: tensors[f"{l}.mean_a"] for l in layer_names}
        means_b = {l: tensors[f"{l}.mean_b"] for l in layer_names}

        tasks = {}
        for tid in task_ids:
            safe_name = tid_to_filename.get(tid, tid)
            task_tensors = load_file(str(path / f"task_{safe_name}.safetensors"))
            loadings_a = {l: task_tensors[f"{l}.loadings_a"] for l in layer_names}
            loadings_b = {l: task_tensors[f"{l}.loadings_b"] for l in layer_names}
            tasks[tid] = TaskProjection(
                task_id=tid, loadings_a=loadings_a, loadings_b=loadings_b
            )

        return cls(
            layer_names=layer_names,
            components_a=components_a,
            components_b=components_b,
            singular_values_a=sv_a,
            singular_values_b=sv_b,
            means_a=means_a,
            means_b=means_b,
            tasks=tasks,
            rank=rank,
            num_components=num_components,
        )
