"""Tests for vlora.router — task routing and adapter blending."""

import torch

from vlora.io import LoRAWeights
from vlora.router import TaskRouter
from vlora.subspace import SharedSubspace


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


class TestTaskRouter:
    def test_output_shape(self):
        router = TaskRouter(input_dim=32, num_tasks=3)
        x = torch.randn(4, 32)
        weights = router(x)
        assert weights.shape == (4, 3)

    def test_weights_sum_to_one(self):
        router = TaskRouter(input_dim=32, num_tasks=3)
        x = torch.randn(4, 32)
        weights = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_weights_nonnegative(self):
        router = TaskRouter(input_dim=32, num_tasks=3)
        x = torch.randn(4, 32)
        weights = router(x)
        assert (weights >= 0).all()

    def test_temperature_makes_softer(self):
        torch.manual_seed(42)
        router_cold = TaskRouter(input_dim=32, num_tasks=3, temperature=0.1)
        torch.manual_seed(42)
        router_warm = TaskRouter(input_dim=32, num_tasks=3, temperature=10.0)

        x = torch.randn(4, 32)
        cold_weights = router_cold(x)
        warm_weights = router_warm(x)

        # Higher temperature → more uniform → higher entropy
        cold_entropy = -(cold_weights * cold_weights.log()).sum(dim=-1).mean()
        warm_entropy = -(warm_weights * warm_weights.log()).sum(dim=-1).mean()
        assert warm_entropy > cold_entropy

    def test_num_params(self):
        router = TaskRouter(input_dim=32, num_tasks=3, hidden_dim=16)
        # Layer 1: 32*16 + 16 = 528
        # Layer 2: 16*3 + 3 = 51
        assert router.num_params == 32 * 16 + 16 + 16 * 3 + 3


class TestBlendLoadings:
    def test_blended_task_id(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        router = TaskRouter.from_subspace(sub, input_dim=32)

        x = torch.randn(4, 32)
        blended = router.blend_loadings(x, sub)
        assert blended.task_id == "__routed__"

    def test_blended_loadings_shape(self):
        adapters, layers = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        router = TaskRouter.from_subspace(sub, input_dim=32)

        x = torch.randn(4, 32)
        blended = router.blend_loadings(x, sub)
        for l in layers:
            assert blended.loadings_a[l].shape == (2,)
            assert blended.loadings_b[l].shape == (2,)

    def test_blended_can_reconstruct(self):
        adapters, layers = _make_adapters(3, rank=4, dim=64)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        router = TaskRouter.from_subspace(sub, input_dim=32)

        x = torch.randn(4, 32)
        blended = router.blend_loadings(x, sub)

        # Register the blended task and reconstruct
        sub.tasks["__routed__"] = blended
        recon = sub.reconstruct("__routed__")
        for l in layers:
            assert recon.lora_a[l].shape == (4, 64)


class TestFromSubspace:
    def test_creates_matching_router(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        router = TaskRouter.from_subspace(sub, input_dim=32)
        assert router.num_tasks == 3
        assert router.task_ids == sorted(sub.tasks.keys())

    def test_init_from_loadings_sets_bias(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)
        router = TaskRouter.from_subspace(sub, input_dim=32, init_from_loadings=True)
        bias = router.net[-1].bias.data
        # Bias should be set (not all zeros as default init)
        assert not (bias == 0).all()

    def test_no_init_from_loadings(self):
        adapters, _ = _make_adapters(3)
        sub = SharedSubspace.from_adapters(adapters, num_components=2)

        torch.manual_seed(42)
        router = TaskRouter.from_subspace(sub, input_dim=32, init_from_loadings=False)
        # With init_from_loadings=False, bias comes from default PyTorch init
        # Just check it runs without error
        x = torch.randn(2, 32)
        weights = router(x)
        assert weights.shape == (2, 3)
