"""High-level convenience wrappers for the 3-step pipeline."""

from __future__ import annotations

__all__ = ["init_subspace", "absorb_task", "extract_adapter"]

from pathlib import Path

from vlora.io import LoRAWeights, load_adapter, save_adapter
from vlora.subspace import SharedSubspace


def init_subspace(
    adapter_paths: list[str | Path],
    task_ids: list[str] | None = None,
    variance_threshold: float = 0.6,
    num_components: int | None = None,
) -> SharedSubspace:
    """Load adapters from disk and build a shared subspace in one call.

    Args:
        adapter_paths: Directories containing PEFT adapters.
        task_ids: Names for each adapter.
        variance_threshold: Variance threshold for auto component selection.
        num_components: Explicit number of components (overrides threshold).

    Returns:
        Initialized SharedSubspace.
    """
    adapters = [load_adapter(p) for p in adapter_paths]
    return SharedSubspace.from_adapters(
        adapters,
        task_ids=task_ids,
        variance_threshold=variance_threshold,
        num_components=num_components,
    )


def absorb_task(
    subspace: SharedSubspace,
    adapter_path: str | Path,
    task_id: str,
) -> None:
    """Load a new adapter and absorb it into the subspace.

    Args:
        subspace: Existing shared subspace (modified in-place).
        adapter_path: Directory containing the new PEFT adapter.
        task_id: Name for the new task.
    """
    adapter = load_adapter(adapter_path)
    subspace.absorb(adapter, task_id)


def extract_adapter(
    subspace: SharedSubspace,
    task_id: str,
    output_path: str | Path,
) -> LoRAWeights:
    """Reconstruct a task's adapter and save it to disk.

    Args:
        subspace: Shared subspace containing the task.
        task_id: Task to reconstruct.
        output_path: Directory to save the PEFT adapter to.

    Returns:
        The reconstructed LoRAWeights.
    """
    weights = subspace.reconstruct(task_id)
    save_adapter(weights, output_path)
    return weights
