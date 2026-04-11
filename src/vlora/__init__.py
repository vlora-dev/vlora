"""vLoRA — Shared low-rank subspaces for LoRA adapter management.

Based on the Share paper (arXiv:2602.06043): LoRA adapters across tasks
share a common low-rank subspace. Instead of storing N separate adapters,
maintain one shared basis and per-task coefficient vectors.
"""

__version__ = "1.0.0"

from vlora.analysis import (
    adapter_diff,
    compute_similarity_matrix,
    find_clusters,
    find_outliers,
    subspace_coverage,
)
from vlora.io import LoRAWeights, load_adapter, load_adapter_from_hub, save_adapter
from vlora.merge import dare_merge, task_arithmetic, ties_merge
from vlora.model import VLoRAModel
from vlora.pipeline import absorb_task, extract_adapter, init_subspace
from vlora.router import TaskRouter
from vlora.subspace import SharedSubspace, TaskProjection
from vlora.training import SubspaceTrainer, orthogonal_init

__all__ = [
    # Core
    "SharedSubspace",
    "TaskProjection",
    "LoRAWeights",
    # I/O
    "load_adapter",
    "load_adapter_from_hub",
    "save_adapter",
    # Pipeline
    "init_subspace",
    "absorb_task",
    "extract_adapter",
    # Analysis
    "compute_similarity_matrix",
    "find_clusters",
    "adapter_diff",
    "subspace_coverage",
    "find_outliers",
    # Model
    "VLoRAModel",
    # Router
    "TaskRouter",
    # Training
    "SubspaceTrainer",
    "orthogonal_init",
    # Merging
    "task_arithmetic",
    "ties_merge",
    "dare_merge",
]
