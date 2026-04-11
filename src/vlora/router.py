"""TaskRouter — lightweight routing over adapter loadings.

Routes inputs to a soft blend of task adapters by producing per-task
weights. Since adapters are represented as small loading vectors in the
shared subspace, blending is a cheap linear combination rather than
reconstructing and merging full LoRA matrices.

Usage:
    subspace = SharedSubspace.load("shared_subspace/")
    router = TaskRouter.from_subspace(subspace, hidden_dim=64)

    # During inference:
    model = VLoRAModel(base_model, subspace)
    x = get_input_embedding(batch)  # (B, embed_dim)
    blended = router.blend_loadings(x)  # per-input blended loadings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vlora.subspace import SharedSubspace, TaskProjection


class TaskRouter(nn.Module):
    """Small MLP that produces soft task-blend weights from input features.

    Given input embeddings (B, input_dim), outputs (B, num_tasks) blend
    weights that sum to 1 (via softmax). These weights define a per-input
    mixture of task loadings in the shared subspace.
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        hidden_dim: int = 64,
        task_ids: list[str] | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.task_ids = task_ids or [f"task_{i}" for i in range(num_tasks)]
        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute task blend weights.

        Args:
            x: (B, input_dim) input features.

        Returns:
            weights: (B, num_tasks) softmax blend weights.
        """
        logits = self.net(x) / self.temperature
        return F.softmax(logits, dim=-1)

    def blend_loadings(
        self,
        x: Tensor,
        subspace: SharedSubspace,
    ) -> TaskProjection:
        """Produce blended loadings for a batch by mixing task loadings.

        Computes a weighted average of all task loadings using the
        router's output weights. Returns a single TaskProjection whose
        loadings are the blend. For batched inference, uses the mean
        blend across the batch.

        Args:
            x: (B, input_dim) input features.
            subspace: SharedSubspace containing the tasks to blend.

        Returns:
            TaskProjection with blended loadings (one set for the batch).
        """
        weights = self.forward(x)  # (B, num_tasks)
        # Average across batch for a single blended adapter
        avg_weights = weights.mean(dim=0)  # (num_tasks,)

        blended_a: dict[str, Tensor] = {}
        blended_b: dict[str, Tensor] = {}

        for layer in subspace.layer_names:
            # Stack all task loadings: (num_tasks, k)
            stack_a = torch.stack([
                subspace.tasks[tid].loadings_a[layer]
                for tid in self.task_ids
            ])
            stack_b = torch.stack([
                subspace.tasks[tid].loadings_b[layer]
                for tid in self.task_ids
            ])

            # Weighted combination: (k,)
            blended_a[layer] = avg_weights @ stack_a
            blended_b[layer] = avg_weights @ stack_b

        return TaskProjection(
            task_id="__routed__",
            loadings_a=blended_a,
            loadings_b=blended_b,
        )

    @classmethod
    def from_subspace(
        cls,
        subspace: SharedSubspace,
        input_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        init_from_loadings: bool = True,
    ) -> TaskRouter:
        """Create a router matched to a subspace's task structure.

        Optionally initializes the final linear layer's weights so that
        the router starts biased toward separating tasks based on their
        loading similarity (warm start for fine-tuning).

        Args:
            subspace: SharedSubspace with registered tasks.
            input_dim: Dimension of input features the router will see.
            hidden_dim: Router hidden layer size.
            temperature: Softmax temperature (higher = softer blending).
            init_from_loadings: If True, use task loading similarity to
                bias the output layer (helps convergence).
        """
        task_ids = sorted(subspace.tasks.keys())
        num_tasks = len(task_ids)

        router = cls(
            input_dim=input_dim,
            num_tasks=num_tasks,
            hidden_dim=hidden_dim,
            task_ids=task_ids,
            temperature=temperature,
        )

        if init_from_loadings and num_tasks > 1:
            # Use task loading norms as output bias (tasks with larger
            # loadings get slightly higher initial routing weight)
            with torch.no_grad():
                biases = []
                for tid in task_ids:
                    proj = subspace.tasks[tid]
                    total_norm = sum(
                        proj.loadings_a[l].norm() + proj.loadings_b[l].norm()
                        for l in subspace.layer_names
                    )
                    biases.append(total_norm)
                bias_tensor = torch.stack(biases)
                bias_tensor = bias_tensor / (bias_tensor.max() + 1e-8)
                router.net[-1].bias.data.copy_(bias_tensor)

        return router

    @property
    def num_params(self) -> int:
        """Total number of router parameters."""
        return sum(p.numel() for p in self.parameters())
