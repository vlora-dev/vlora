"""Tests for vlora.integrations.huggingface — HF Trainer callback."""

import pytest
import torch
import torch.nn as nn

from vlora.io import LoRAWeights
from vlora.subspace import SharedSubspace
from vlora.training import orthogonal_init

LAYERS = ["layer.0.q_proj", "layer.0.v_proj"]
DIM = 32
RANK = 4


def _make_subspace():
    """Create a small subspace for testing."""
    shared_a = {l: torch.randn(3, RANK * DIM) for l in LAYERS}
    shared_b = {l: torch.randn(3, DIM * RANK) for l in LAYERS}

    adapters = []
    for i in range(3):
        lora_a = {l: (torch.randn(3) @ shared_a[l]).reshape(RANK, DIM) for l in LAYERS}
        lora_b = {l: (torch.randn(3) @ shared_b[l]).reshape(DIM, RANK) for l in LAYERS}
        adapters.append(LoRAWeights(layer_names=LAYERS, lora_a=lora_a, lora_b=lora_b, rank=RANK))

    return SharedSubspace.from_adapters(adapters, num_components=2)


class _TinyModel(nn.Module):
    """Minimal model with named Linear layers matching the subspace."""

    def __init__(self):
        super().__init__()
        # Build nested structure so named_modules() produces "layer.0.q_proj" etc.
        layer_0 = nn.Module()
        layer_0.add_module("q_proj", nn.Linear(DIM, DIM, bias=False))
        layer_0.add_module("v_proj", nn.Linear(DIM, DIM, bias=False))
        layer = nn.Module()
        layer.add_module("0", layer_0)
        self.add_module("layer", layer)

    def forward(self, x):
        x = self.layer._modules["0"].q_proj(x)
        x = self.layer._modules["0"].v_proj(x)
        return x


class TestVLoRACallbackImport:
    def test_import_without_transformers(self):
        """VLoRACallback should be importable even without transformers."""
        from vlora.integrations.huggingface import VLoRACallback
        assert VLoRACallback is not None

    def test_stub_raises_without_transformers(self):
        """If transformers not installed, instantiation raises ImportError."""
        try:
            import transformers  # noqa: F401
            pytest.skip("transformers is installed")
        except ImportError:
            from vlora.integrations.huggingface import VLoRACallback
            with pytest.raises(ImportError, match="transformers"):
                VLoRACallback(None, "test")


def _can_use_training_args():
    """Check if TrainingArguments can be instantiated (needs accelerate)."""
    try:
        from transformers import TrainingArguments
        TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        return True
    except (ImportError, Exception):
        return False


class TestVLoRACallbackWithTransformers:
    @pytest.fixture(autouse=True)
    def skip_without_full_hf(self):
        if not _can_use_training_args():
            pytest.skip("transformers + accelerate not installed")

    def test_callback_creates_trainer_on_begin(self):
        from transformers import TrainerControl, TrainerState, TrainingArguments

        from vlora.integrations.huggingface import VLoRACallback

        sub = _make_subspace()
        orthogonal_init(sub, "test_task")

        callback = VLoRACallback(sub, "test_task", lr=1e-3)
        assert callback.trainer is None

        args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        state = TrainerState()
        control = TrainerControl()
        callback.on_train_begin(args, state, control)

        assert callback.trainer is not None
        assert callback.trainer.num_trainable_params > 0

    def test_callback_write_back_on_end(self):
        from transformers import TrainerControl, TrainerState, TrainingArguments

        from vlora.integrations.huggingface import VLoRACallback

        sub = _make_subspace()
        orthogonal_init(sub, "test_task")

        callback = VLoRACallback(sub, "test_task", lr=1e-3, save_on_end=True)
        args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        state = TrainerState()
        control = TrainerControl()

        callback.on_train_begin(args, state, control)
        callback.on_train_end(args, state, control)

        assert "test_task" in sub.tasks

    def test_callback_logs_metrics(self):
        from transformers import TrainerControl, TrainerState, TrainingArguments

        from vlora.integrations.huggingface import VLoRACallback

        sub = _make_subspace()
        orthogonal_init(sub, "test_task")

        callback = VLoRACallback(sub, "test_task", lr=1e-3, log_every=1)
        args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        state = TrainerState()
        state.global_step = 1
        control = TrainerControl()

        callback.on_train_begin(args, state, control)
        callback.on_step_end(args, state, control)

        vlora_logs = [l for l in state.log_history if "vlora/loadings_norm" in l]
        assert len(vlora_logs) == 1
        assert vlora_logs[0]["vlora/loadings_norm"] >= 0

    def test_callback_actually_trains_loadings(self):
        """Verify loadings change after forward+backward+on_step_end.

        This is the critical integration test: the old callback was a
        no-op that never stepped its optimizer. The new callback registers
        differentiable hooks so the Trainer's backward pass produces
        gradients on loadings, and on_step_end steps the optimizer.
        """
        from transformers import TrainerControl, TrainerState, TrainingArguments

        from vlora.integrations.huggingface import VLoRACallback

        sub = _make_subspace()
        orthogonal_init(sub, "test_task")

        # Snapshot loadings before training
        initial_loadings = {
            l: sub.tasks["test_task"].loadings_a[l].clone()
            for l in LAYERS
        }

        callback = VLoRACallback(sub, "test_task", lr=1e-2, log_every=999)
        args = TrainingArguments(output_dir="/tmp/test", use_cpu=True)
        state = TrainerState()
        state.global_step = 1
        control = TrainerControl()

        # on_train_begin registers differentiable hooks on the model
        model = _TinyModel()
        callback.on_train_begin(args, state, control, model=model)

        # Simulate a training step: forward → loss → backward
        x = torch.randn(2, DIM)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # on_step_end should step the loadings optimizer
        callback.on_step_end(args, state, control)

        # Write back and check loadings changed
        callback.trainer.write_back()
        changed = False
        for l in LAYERS:
            if not torch.equal(sub.tasks["test_task"].loadings_a[l], initial_loadings[l]):
                changed = True
                break

        assert changed, "Loadings did not change after training step — optimizer may not be stepping"
