"""HuggingFace Trainer integration — VLoRACallback for training-in-subspace.

Registers differentiable forward hooks on the base model so that the
Trainer's backward pass flows gradients to subspace loadings. A separate
optimizer steps the loadings each training step.

Usage with HuggingFace Trainer:
    from vlora import SharedSubspace, orthogonal_init
    from vlora.integrations.huggingface import VLoRACallback

    subspace = SharedSubspace.load("shared_subspace/")
    orthogonal_init(subspace, "new_task")

    callback = VLoRACallback(subspace, "new_task", lr=1e-3)
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[callback],
    )
    trainer.train()
    subspace.save("updated_subspace/")
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn
from torch import Tensor

from vlora.ops import reconstruct_from_subspace
from vlora.subspace import SharedSubspace
from vlora.training import SubspaceTrainer

logger = logging.getLogger("vlora")

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _require_transformers():
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is required for HuggingFace integration. "
            "Install with: pip install vlora-dev[hf]"
        )


if HAS_TRANSFORMERS:

    class VLoRACallback(TrainerCallback):
        """HuggingFace Trainer callback for training-in-subspace.

        Registers differentiable forward hooks on the base model so the
        Trainer's backward pass produces gradients on the subspace loadings.
        A separate optimizer steps the loadings each training step.

        The hooks compute: ``output += (x @ (B @ A).T)`` where A and B are
        differentiably reconstructed from the trainable loadings, allowing
        gradients to flow from the loss through to the loading parameters.

        Args:
            subspace: Shared subspace (task must already exist).
            task_id: Task whose loadings to train.
            lr: Learning rate for loadings optimizer.
            num_expand: Extra orthogonal directions for the optimizer.
            log_every: Log adapter metrics every N steps.
            save_on_end: Whether to call write_back() when training ends.
        """

        def __init__(
            self,
            subspace: SharedSubspace,
            task_id: str,
            lr: float = 1e-3,
            num_expand: int = 0,
            log_every: int = 50,
            save_on_end: bool = True,
        ):
            _require_transformers()
            self.subspace = subspace
            self.task_id = task_id
            self.lr = lr
            self.num_expand = num_expand
            self.log_every = log_every
            self.save_on_end = save_on_end
            self._trainer: SubspaceTrainer | None = None
            self._fwd_hooks: list = []

        def _make_differentiable_hook(self, layer_name: str):
            """Create a forward hook that reconstructs LoRA delta from loadings.

            The reconstruction is differentiable — gradients flow from the
            loss through the hook back to the loadings parameters.
            """
            assert self._trainer is not None
            params = self._trainer.params
            la = params[f"{layer_name}.loadings_a"]
            lb = params[f"{layer_name}.loadings_b"]
            comp_a = self.subspace.components_a[layer_name]
            comp_b = self.subspace.components_b[layer_name]
            mean_a = self.subspace.means_a[layer_name]
            mean_b = self.subspace.means_b[layer_name]
            rank = self.subspace.rank

            def hook(module: nn.Module, input: Any, output: Tensor) -> Tensor:
                x = input[0] if isinstance(input, tuple) else input
                # Differentiable reconstruction: loadings → flat → matrix
                flat_a = reconstruct_from_subspace(comp_a, la) + mean_a
                flat_b = reconstruct_from_subspace(comp_b, lb) + mean_b
                A = flat_a.reshape(rank, -1)
                B = flat_b.reshape(-1, rank)
                delta = B @ A
                lora_out = x.to(delta.dtype) @ delta.T.to(x.device)
                return output + lora_out.to(output.dtype)

            return hook

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ):
            self._trainer = SubspaceTrainer(
                self.subspace,
                self.task_id,
                lr=self.lr,
                num_expand=self.num_expand,
            )

            # Register differentiable hooks on the base model
            model = kwargs.get("model")
            if model is not None:
                for name, module in model.named_modules():
                    if name in self.subspace.layer_names and isinstance(module, nn.Linear):
                        hook = module.register_forward_hook(
                            self._make_differentiable_hook(name)
                        )
                        self._fwd_hooks.append(hook)

            logger.info(
                "VLoRACallback: training '%s' with %d params (lr=%.1e), %d hooks",
                self.task_id,
                self._trainer.num_trainable_params,
                self.lr,
                len(self._fwd_hooks),
            )

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ):
            if self._trainer is None:
                return

            # Step the loadings optimizer. Gradients accumulated during the
            # Trainer's backward pass via our differentiable hooks. Our
            # loadings are standalone tensors (not model.parameters()), so
            # their gradients survive the Trainer's model.zero_grad().
            self._trainer.optimizer.step()
            self._trainer.optimizer.zero_grad()
            self._trainer._step_count += 1

            step = state.global_step
            if step > 0 and step % self.log_every == 0:
                total_norm = 0.0
                for p in self._trainer.params.values():
                    total_norm += p.data.norm().item() ** 2
                total_norm = total_norm ** 0.5

                metrics = {
                    "vlora/loadings_norm": total_norm,
                    "vlora/trainable_params": self._trainer.num_trainable_params,
                    "vlora/step": self._trainer.step_count,
                }
                state.log_history.append(
                    {"step": step, **metrics}
                )
                logger.debug(
                    "VLoRACallback step %d: loadings_norm=%.4f",
                    step,
                    total_norm,
                )

        def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: Any,
        ):
            # Remove hooks
            for h in self._fwd_hooks:
                h.remove()
            self._fwd_hooks.clear()

            if self._trainer is not None and self.save_on_end:
                self._trainer.write_back()
                logger.info(
                    "VLoRACallback: wrote back loadings for '%s' after %d steps",
                    self.task_id,
                    self._trainer.step_count,
                )

        @property
        def trainer(self) -> SubspaceTrainer | None:
            """Access the underlying SubspaceTrainer (available after on_train_begin)."""
            return self._trainer

else:
    # Stub class when transformers is not installed
    class VLoRACallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_transformers()
