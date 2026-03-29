"""Load and parse PEFT LoRA adapters from disk or HuggingFace Hub.

Reads safetensors + adapter_config.json without requiring the PEFT
library at runtime.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

logger = logging.getLogger("vlora")
from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass
class LoRAWeights:
    """Parsed LoRA adapter weights grouped by layer."""

    layer_names: list[str]
    lora_a: dict[str, Tensor]  # layer_name -> (rank, in_features)
    lora_b: dict[str, Tensor]  # layer_name -> (out_features, rank)
    rank: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"LoRAWeights(rank={self.rank}, layers={len(self.layer_names)})"


# Pattern to extract layer name + side from PEFT state dict keys.
# Handles single-adapter format:
#   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
#   model.layers.0.self_attn.q_proj.lora_A.weight
# And multi-adapter format (PEFT >= 0.6):
#   base_model.model.model.layers.0.self_attn.q_proj.lora_A.adapter_name.weight
_LORA_KEY_RE = re.compile(
    r"(?:base_model\.model\.)?(.+)\.(lora_[AB])\.(?:\w+\.)?weight"
)


def parse_state_dict(
    state_dict: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor], list[str]]:
    """Parse a PEFT state dict into grouped A and B weight dicts.

    Returns:
        lora_a: {layer_name: tensor}
        lora_b: {layer_name: tensor}
        layer_names: sorted unique layer names
    """
    lora_a: dict[str, Tensor] = {}
    lora_b: dict[str, Tensor] = {}

    for key, tensor in state_dict.items():
        match = _LORA_KEY_RE.search(key)
        if match is None:
            continue
        layer_name = match.group(1)
        side = match.group(2)
        if side == "lora_A":
            lora_a[layer_name] = tensor
        else:
            lora_b[layer_name] = tensor

    layer_names = sorted(set(lora_a.keys()) & set(lora_b.keys()))
    # Keep only layers where both A and B exist
    lora_a = {k: lora_a[k] for k in layer_names}
    lora_b = {k: lora_b[k] for k in layer_names}

    return lora_a, lora_b, layer_names


def load_adapter(path: str | Path) -> LoRAWeights:
    """Load a PEFT LoRA adapter from a local directory.

    Expects adapter_model.safetensors and adapter_config.json in the
    given directory.
    """
    path = Path(path)

    # Load safetensors weights
    safetensors_path = path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {path}")
    state_dict = load_file(str(safetensors_path))

    # Load config for metadata
    config_path = path / "adapter_config.json"
    metadata: dict = {}
    rank = 0
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        rank = config.get("r", 0)
        metadata = config

    lora_a, lora_b, layer_names = parse_state_dict(state_dict)
    logger.debug("Loaded adapter from %s: %d layers", path, len(layer_names))

    # Infer rank from weight shapes if not in config
    if rank == 0 and layer_names:
        rank = lora_a[layer_names[0]].shape[0]

    return LoRAWeights(
        layer_names=layer_names,
        lora_a=lora_a,
        lora_b=lora_b,
        rank=rank,
        metadata=metadata,
    )


def load_adapter_from_hub(repo_id: str, revision: str | None = None) -> LoRAWeights:
    """Load a PEFT LoRA adapter from HuggingFace Hub.

    Requires the `huggingface-hub` package (install with `pip install vlora-dev[hub]`).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required to load from Hub. "
            "Install with: pip install vlora-dev[hub]"
        )

    local_dir = snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
    )
    return load_adapter(local_dir)


def stack_lora_weights(
    adapters: list[LoRAWeights],
    side: Literal["A", "B"],
) -> dict[str, Tensor]:
    """Stack LoRA weight matrices from multiple adapters per layer.

    For side="A": each adapter's A matrix is (rank, in_features).
    We flatten each to a row vector and stack N adapters into (N, rank*in_features).

    This produces the "factor data matrix" the paper feeds into SVD.

    Returns:
        {layer_name: (N, flattened_dim)} stacked matrix.
    """
    if not adapters:
        raise ValueError("Need at least one adapter to stack")

    # Use intersection of all adapters' layer names
    layer_set = set(adapters[0].layer_names)
    for adapter in adapters[1:]:
        layer_set &= set(adapter.layer_names)
    layer_names = sorted(layer_set)

    stacked: dict[str, Tensor] = {}
    for layer in layer_names:
        matrices = []
        for adapter in adapters:
            w = adapter.lora_a[layer] if side == "A" else adapter.lora_b[layer]
            matrices.append(w.flatten())
        stacked[layer] = torch.stack(matrices)

    return stacked


def save_adapter(weights: LoRAWeights, path: str | Path) -> None:
    """Save LoRA weights back to PEFT-compatible format."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Rebuild state dict with PEFT key format
    state_dict = {}
    for layer_name in weights.layer_names:
        state_dict[f"base_model.model.{layer_name}.lora_A.weight"] = weights.lora_a[layer_name]
        state_dict[f"base_model.model.{layer_name}.lora_B.weight"] = weights.lora_b[layer_name]

    save_file(state_dict, str(path / "adapter_model.safetensors"))

    # Save config — include defaults needed by vLLM/TGI
    config = dict(weights.metadata) if weights.metadata else {}
    config.setdefault("r", weights.rank)
    config.setdefault("lora_alpha", weights.rank)  # alpha=rank → scaling=1.0
    config.setdefault("peft_type", "LORA")
    # Don't default task_type — it depends on the model architecture
    # (CAUSAL_LM, SEQ_2_SEQ_LM, TOKEN_CLS, etc.). Let PEFT infer it.
    config.setdefault("bias", "none")
    config.setdefault("lora_dropout", 0.0)
    with open(path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
