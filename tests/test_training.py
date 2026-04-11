"""Tests for vlora.training — subspace-constrained training."""

import pytest
import torch

from vlora.io import LoRAWeights
from vlora.subspace import SharedSubspace
from vlora.training import SubspaceTrainer, orthogonal_init


def _make_adapters(n=3, layers=None, rank=4, dim=64):
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]
    shared_a = {l: torch.randn(3, rank * dim) for l in layers}
    shared_b = {l: torch.randn(3, dim * rank) for l in layers}
    adapters = []
    for i in range(n):
        lora_a = {l: (torch.randn(3) @ shared_a[l]).reshape(rank, dim) for l in layers}
        lora_b = {l: (torch.randn(3) @ shared_b[l]).reshape(dim, rank) for l in layers}
        adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))
    return adapters, layers


class TestOrthogonalInit:
    def test_creates_task(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        proj = orthogonal_init(sub, "new_task")
        assert "new_task" in sub.tasks
        assert proj.task_id == "new_task"

    def test_loadings_shape(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        proj = orthogonal_init(sub, "new_task")
        for l in layers:
            assert proj.loadings_a[l].shape == (2,)
            assert proj.loadings_b[l].shape == (2,)

    def test_b_side_starts_zero(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        proj = orthogonal_init(sub, "new_task")
        for l in layers:
            assert (proj.loadings_b[l] == 0).all()

    def test_a_side_is_small(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        proj = orthogonal_init(sub, "new_task", scale=0.01)
        for l in layers:
            assert proj.loadings_a[l].abs().max() < 0.1

    def test_can_reconstruct_after_init(self):
        adapters, layers = _make_adapters(3, rank=4, dim=64)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        orthogonal_init(sub, "new_task")
        recon = sub.reconstruct("new_task")
        for l in layers:
            assert recon.lora_a[l].shape == (4, 64)
            assert recon.lora_b[l].shape == (64, 4)


class TestSubspaceTrainer:
    def test_init_requires_existing_task(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        with pytest.raises(KeyError, match="not in subspace"):
            SubspaceTrainer(sub, "nonexistent")

    def test_num_trainable_params(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        trainer = SubspaceTrainer(sub, "task_0")
        # 2 layers × 2 sides × 2 components = 8
        assert trainer.num_trainable_params == len(layers) * 2 * 2

    def test_step_decreases_loss(self):
        adapters, layers = _make_adapters(3, rank=4, dim=16)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        trainer = SubspaceTrainer(sub, "task_0", lr=0.1)

        # Target: task_1's loadings
        target_proj = sub.tasks["task_1"]

        losses = []
        for _ in range(20):
            loss = torch.tensor(0.0)
            for l in layers:
                diff_a = trainer.params[f"{l}.loadings_a"] - target_proj.loadings_a[l].detach()
                diff_b = trainer.params[f"{l}.loadings_b"] - target_proj.loadings_b[l].detach()
                loss = loss + (diff_a ** 2).sum() + (diff_b ** 2).sum()
            losses.append(trainer.step(loss))

        assert losses[-1] < losses[0]

    def test_step_count(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        trainer = SubspaceTrainer(sub, "task_0")
        assert trainer.step_count == 0
        # Dummy loss
        loss = sum(p.sum() for p in trainer.params.values())
        trainer.step(loss)
        assert trainer.step_count == 1

    def test_write_back(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        trainer = SubspaceTrainer(sub, "task_0", lr=0.5)

        # Take a step to change params
        loss = sum((p ** 2).sum() for p in trainer.params.values())
        trainer.step(loss)

        # Before write_back, subspace still has original values
        old_val = sub.tasks["task_0"].loadings_a[layers[0]].clone()
        trainer.write_back()
        new_val = sub.tasks["task_0"].loadings_a[layers[0]]

        # Values should have changed
        assert not torch.allclose(old_val, new_val)

    def test_with_expand(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        trainer = SubspaceTrainer(sub, "task_0", num_expand=2)
        # Should have more params due to expanded basis
        assert trainer.num_trainable_params > len(layers) * 2 * 2
