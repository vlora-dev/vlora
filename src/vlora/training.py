"""Training within the shared subspace — train only loadings, not full LoRA.

Instead of optimizing rank × dim parameters per layer (standard LoRA),
train just k scalar loadings per layer (where k is the number of subspace
components). This gives 100x+ parameter reduction while staying in the
space of known-good adapter directions.

Usage:
    subspace = SharedSubspace.load("shared_subspace/")
    trainer = SubspaceTrainer(subspace, "new_task")

    for batch in dataloader:
        loss = compute_loss(trainer.model, batch)
        trainer.step(loss)

    # Loadings are updated in-place on the subspace
    subspace.save("updated_subspace/")
"""

from __future__ import annotations

__all__ = ["SubspaceTrainer", "orthogonal_init"]

import torch
from torch import Tensor

from vlora.subspace import SharedSubspace, TaskProjection


def orthogonal_init(
    subspace: SharedSubspace,
    task_id: str,
    scale: float = 0.01,
) -> TaskProjection:
    """Initialize a new task with small random loadings.

    Uses normally-distributed loadings scaled down so the initial adapter
    is near-zero (similar to LoRA's Kaiming + zero init strategy).

    Args:
        subspace: The shared subspace to initialize within.
        task_id: Name for the new task.
        scale: Standard deviation of initial loadings. Small values mean
            the adapter starts near-identity.

    Returns:
        TaskProjection registered in the subspace.
    """
    loadings_a = {}
    loadings_b = {}

    for layer in subspace.layer_names:
        actual_k = subspace.components_a[layer].shape[0]
        device = subspace.components_a[layer].device
        dtype = subspace.components_a[layer].dtype
        loadings_a[layer] = torch.randn(actual_k, device=device, dtype=dtype) * scale
        # Initialize B-side to zero (like standard LoRA) so initial delta is zero
        loadings_b[layer] = torch.zeros(actual_k, device=device, dtype=dtype)

    proj = TaskProjection(task_id=task_id, loadings_a=loadings_a, loadings_b=loadings_b)
    subspace.tasks[task_id] = proj
    return proj


class SubspaceTrainer:
    """Minimal training loop for learning task loadings within a subspace.

    Freezes the shared basis (components) and only optimizes the per-task
    loadings vector. Works with any PyTorch model and loss function.

    The trainer creates parameters with requires_grad=True from the task's
    loadings and provides an optimizer + step method. Compatible with
    standard PyTorch training patterns and HuggingFace Trainer via
    get_trainable_params().
    """

    def __init__(
        self,
        subspace: SharedSubspace,
        task_id: str,
        lr: float = 1e-3,
        num_expand: int = 0,
        optimizer_cls: type = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
    ):
        """
        Args:
            subspace: Shared subspace (must already contain the task).
            task_id: Task whose loadings to train.
            lr: Learning rate.
            num_expand: Extra orthogonal directions to add to the basis
                via Gram-Schmidt. Gives the optimizer room to escape the
                existing subspace if needed.
            optimizer_cls: PyTorch optimizer class.
            optimizer_kwargs: Extra kwargs for the optimizer.
        """
        if task_id not in subspace.tasks:
            raise KeyError(
                f"Task '{task_id}' not in subspace. "
                "Use orthogonal_init() or subspace.project() first."
            )

        self.subspace = subspace
        self.task_id = task_id

        # Get trainable parameter tensors
        self.params = subspace.get_trainable_params(task_id, num_expand=num_expand)

        # Build optimizer
        param_list = list(self.params.values())
        kwargs = dict(optimizer_kwargs or {})
        kwargs["lr"] = lr
        self.optimizer = optimizer_cls(param_list, **kwargs)

        self._step_count = 0

    def step(self, loss: Tensor) -> float:
        """Backprop and update loadings from a scalar loss.

        Args:
            loss: Scalar loss tensor (must have grad_fn).

        Returns:
            Loss value as float.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._step_count += 1
        return loss.item()  # type: ignore[no-any-return]

    def write_back(self) -> None:
        """Write trained parameters back to the subspace's TaskProjection.

        Call this after training is done to persist the learned loadings
        back into the subspace object.
        """
        proj = self.subspace.tasks[self.task_id]
        for layer in self.subspace.layer_names:
            proj.loadings_a[layer] = self.params[f"{layer}.loadings_a"].detach().clone()
            proj.loadings_b[layer] = self.params[f"{layer}.loadings_b"].detach().clone()

    @property
    def num_trainable_params(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.params.values())

    @property
    def step_count(self) -> int:
        """Number of optimizer steps taken."""
        return self._step_count

    def __repr__(self) -> str:
        return (
            f"SubspaceTrainer(task={self.task_id!r}, "
            f"params={self.num_trainable_params}, "
            f"steps={self._step_count})"
        )
