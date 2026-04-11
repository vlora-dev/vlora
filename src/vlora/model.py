"""VLoRAModel — inference wrapper that applies reconstructed LoRA deltas."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from vlora._validate import check_task_exists
from vlora.subspace import SharedSubspace


def _is_linear_layer(module: nn.Module) -> bool:
    """Check if a module is a linear layer, including quantized variants.

    Handles standard nn.Linear and bitsandbytes quantized layers
    (Linear4bit, Linear8bitLt) which are used by QLoRA.
    """
    if isinstance(module, nn.Linear):
        return True
    # bitsandbytes quantized layers inherit from nn.Linear, so the
    # check above covers them. This fallback handles non-standard
    # quantization libraries whose linear layers don't inherit nn.Linear.
    cls_name = type(module).__name__
    return cls_name in ("Linear4bit", "Linear8bitLt", "LinearNF4")


class VLoRAModel(nn.Module):
    """Wraps a base model with a shared subspace for multi-task LoRA inference.

    Reconstructs task-specific LoRA deltas on demand and applies them to
    the base model's linear layers during forward pass.

    Supports both standard and QLoRA-quantized base models. When using a
    quantized model (e.g. loaded with ``load_in_4bit=True``), set
    ``compute_dtype`` to match the model's compute precision (typically
    ``torch.bfloat16``).

    Usage:
        subspace = SharedSubspace.load("shared_subspace/")
        base_model = AutoModelForCausalLM.from_pretrained("model-name")
        model = VLoRAModel(base_model, subspace)

        model.set_task("task_0")
        output = model(input_ids)

        model.set_task("task_1")  # switches adapter, cached if same task
        output = model(input_ids)

    QLoRA usage:
        base_model = AutoModelForCausalLM.from_pretrained(
            "model-name", load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = VLoRAModel(base_model, subspace, compute_dtype=torch.bfloat16)
    """

    def __init__(
        self,
        base_model: nn.Module,
        subspace: SharedSubspace,
        scaling: float | None = None,
        lora_alpha: float | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.subspace = subspace
        self.compute_dtype = compute_dtype

        # Resolve scaling: explicit scaling > lora_alpha/rank > 1.0
        if scaling is not None:
            self.scaling = scaling
        elif lora_alpha is not None:
            self.scaling = lora_alpha / subspace.rank
        else:
            self.scaling = 1.0
        self._active_task: str | None = None
        self._cached_deltas: dict[str, Tensor] | None = None
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._merged: bool = False
        self._merge_deltas: dict[str, Tensor] | None = None
        self._merged_task_id: str | None = None
        # Cache module handles once to avoid O(M) scan on every task switch
        self._target_modules: dict[str, nn.Module] = {
            name: module
            for name, module in self.base_model.named_modules()
            if name in self.subspace.layer_names and _is_linear_layer(module)
        }
        self._qlora_info = self._detect_quantization()

    def set_task(self, task_id: str) -> None:
        """Set the active task adapter. Reconstructs and caches if changed."""
        if self._merged:
            raise RuntimeError(
                "Cannot switch tasks while merged. Call unmerge() first."
            )
        if task_id == self._active_task:
            return

        check_task_exists(self.subspace, task_id)

        # Reconstruct and cache the LoRA deltas
        weights = self.subspace.reconstruct(task_id)
        self._cached_deltas = {}
        for layer_name in weights.layer_names:
            # delta_W = B @ A
            delta = weights.lora_b[layer_name] @ weights.lora_a[layer_name]
            self._cached_deltas[layer_name] = delta

        self._active_task = task_id
        self._apply_hooks()

    def clear_task(self) -> None:
        """Remove the active task adapter."""
        if self._merged:
            raise RuntimeError(
                "Cannot clear task while merged. Call unmerge() first."
            )
        self._remove_hooks()
        self._active_task = None
        self._cached_deltas = None

    def _apply_hooks(self) -> None:
        """Register forward hooks on matching linear layers."""
        self._remove_hooks()

        if self._cached_deltas is None:
            return

        for name, module in self._target_modules.items():
            if name in self._cached_deltas:
                delta = self._cached_deltas[name]
                hook = module.register_forward_hook(
                    self._make_lora_hook(delta)
                )
                self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_lora_hook(self, delta: Tensor):
        """Create a forward hook that adds LoRA delta to the output.

        Wrapped in torch.no_grad() since the delta is a pre-computed
        constant — no gradients should flow through inference hooks.
        (Differentiable hooks for training are handled separately by
        VLoRACallback.)
        """
        scaling = self.scaling
        compute_dtype = self.compute_dtype

        def hook(module: nn.Module, input: Any, output: Tensor) -> Tensor:
            x = input[0] if isinstance(input, tuple) else input
            with torch.no_grad():
                dtype = compute_dtype if compute_dtype is not None else x.dtype
                lora_out = x.to(dtype) @ delta.T.to(x.device, dtype)
            return output + scaling * lora_out.to(output.dtype)

        return hook

    def forward(self, *args, **kwargs):
        """Forward pass through the base model with active LoRA adapter."""
        return self.base_model(*args, **kwargs)

    @property
    def active_task(self) -> str | None:
        """Currently active task ID, or None."""
        return self._active_task

    @property
    def available_tasks(self) -> list[str]:
        """List of available task IDs."""
        return sorted(self.subspace.tasks.keys())

    def reconstruct_state_dict(self, task_id: str) -> dict[str, Tensor]:
        """Get the LoRA delta weight dict for a task without applying hooks.

        Returns dict of {layer_name: delta_W} where delta_W = B @ A.
        Useful for manual integration with custom model architectures.
        """
        weights = self.subspace.reconstruct(task_id)
        deltas = {}
        for layer_name in weights.layer_names:
            deltas[layer_name] = weights.lora_b[layer_name] @ weights.lora_a[layer_name]
        return deltas

    def _detect_quantization(self) -> dict:
        """Introspect base model for quantized layers."""
        info: dict[str, Any] = {
            "quantized": False,
            "method": None,
            "num_quantized_layers": 0,
            "num_target_layers": len(self._target_modules),
        }
        try:
            import bitsandbytes as bnb

            for module in self._target_modules.values():
                if isinstance(module, bnb.nn.Linear4bit):
                    info["quantized"] = True
                    info["method"] = info["method"] or "nf4"
                    info["num_quantized_layers"] += 1
                elif isinstance(module, bnb.nn.Linear8bitLt):
                    info["quantized"] = True
                    info["method"] = info["method"] or "int8"
                    info["num_quantized_layers"] += 1
        except ImportError:
            pass

        return info

    @property
    def qlora_info(self) -> dict:
        """Quantization info about the base model.

        Returns a dict with keys:
        - ``quantized``: whether bitsandbytes quantized layers were detected
        - ``method``: ``"nf4"``, ``"int8"``, or ``None``
        - ``num_quantized_layers``: count of quantized linear layers
        - ``num_target_layers``: subspace layers matched in the base model
        """
        return dict(self._qlora_info)

    def merge(self, task_id: str | None = None) -> None:
        """Bake LoRA deltas into base model weights for hook-free inference.

        After merging, the model runs without hooks — the adapter effect
        is permanent in ``weight.data`` until ``unmerge()`` is called.
        This eliminates per-layer hook overhead for serving a single adapter.

        Args:
            task_id: Task to merge. Uses the active task if None.

        Raises:
            RuntimeError: If the model is already merged, or if the base
                model has quantized layers (cannot modify quantized weights).
            ValueError: If no task is specified and no task is active.
        """
        if self._merged:
            raise RuntimeError(
                "Model is already merged. Call unmerge() first."
            )
        if self._qlora_info["quantized"]:
            raise RuntimeError(
                "Cannot merge into quantized base model weights. "
                "Use hook-based inference with set_task() instead."
            )

        tid = task_id or self._active_task
        if tid is None:
            raise ValueError(
                "No task to merge. Pass task_id or call set_task() first."
            )
        check_task_exists(self.subspace, tid)

        # Compute deltas (reuse cache if available for this task)
        if self._cached_deltas is not None and self._active_task == tid:
            deltas = self._cached_deltas
        else:
            deltas = self.reconstruct_state_dict(tid)

        # Bake deltas into base model weights
        with torch.no_grad():
            for name, module in self._target_modules.items():
                if name in deltas:
                    delta = deltas[name]
                    module.weight.data += (
                        self.scaling * delta
                    ).to(module.weight.device, module.weight.dtype)

        # Cache deltas for unmerge and clean up hooks
        self._merge_deltas = {
            name: delta.clone() for name, delta in deltas.items()
        }
        self._remove_hooks()
        self._merged = True
        self._merged_task_id = tid
        self._active_task = tid
        self._cached_deltas = None

    def unmerge(self) -> None:
        """Reverse a previous ``merge()``, restoring original base weights.

        Raises:
            RuntimeError: If the model is not currently merged.
        """
        if not self._merged:
            raise RuntimeError("Model is not merged. Nothing to unmerge.")

        assert self._merge_deltas is not None  # guaranteed by _merged check
        with torch.no_grad():
            for name, module in self._target_modules.items():
                if name in self._merge_deltas:
                    delta = self._merge_deltas[name]
                    module.weight.data -= (
                        self.scaling * delta
                    ).to(module.weight.device, module.weight.dtype)

        self._merged = False
        self._merge_deltas = None
        self._merged_task_id = None
        self._active_task = None

    @property
    def is_merged(self) -> bool:
        """Whether LoRA deltas are currently baked into base weights."""
        return self._merged

    def compile(self, **kwargs) -> VLoRAModel:
        """Compile the base model with torch.compile for faster inference.

        Passes all kwargs to torch.compile(). The LoRA hooks remain
        uncompiled (they're lightweight matmuls) while the base model
        benefits from fusion and kernel optimization.

        Returns self for chaining.
        """
        self.base_model = torch.compile(self.base_model, **kwargs)
        return self
