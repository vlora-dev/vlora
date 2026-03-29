"""vLoRA — Shared low-rank subspaces for LoRA adapter management.

Based on the Share paper (arXiv:2602.06043): LoRA adapters across tasks
share a common low-rank subspace. Instead of storing N separate adapters,
maintain one shared basis and per-task coefficient vectors.
"""

__version__ = "0.2.1"

from vlora.io import LoRAWeights, load_adapter, load_adapter_from_hub, save_adapter
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
from vlora.model import VLoRAModel
from vlora.ops import incremental_svd_update
from vlora.analysis import (
    adapter_diff,
    compute_similarity_matrix,
    find_clusters,
    find_outliers,
    subspace_coverage,
)
from vlora.pipeline import absorb_task, extract_adapter, init_subspace
from vlora.router import TaskRouter
from vlora.subspace import SharedSubspace, TaskProjection
from vlora.training import SubspaceTrainer, orthogonal_init
from vlora.merge import task_arithmetic, ties_merge, dare_merge

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
    # Ops
    "compute_svd",
    "project_onto_subspace",
    "reconstruct_from_subspace",
    "gram_schmidt",
    "explained_variance_ratio",
    "select_num_components",
    # NF4 quantization (QLoRA-style)
    "NF4_QUANT_TABLE",
    "nf4_quantize_dequantize",
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
    # Incremental
    "incremental_svd_update",
    # Merging
    "task_arithmetic",
    "ties_merge",
    "dare_merge",
]
