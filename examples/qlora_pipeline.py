"""QLoRA + vLoRA: End-to-end pipeline for efficient multi-adapter serving.

This example shows the full workflow:
1. Load a QLoRA-quantized base model (4-bit NF4)
2. Load multiple LoRA adapters (produced by QLoRA fine-tuning)
3. Build a shared subspace with NF4 quantization
4. Serve with instant task switching via VLoRAModel

Requirements:
    pip install vlora-dev[hub] transformers bitsandbytes accelerate

The pipeline combines two orthogonal compression techniques:
- QLoRA: compresses the base model (FP16 -> NF4, ~4x savings)
- vLoRA: compresses the adapter space (N adapters -> shared subspace, ~122x)
Together they enable serving hundreds of task-specific adapters on a single GPU.
"""

from __future__ import annotations

import torch

# ── Step 0: Configuration ──────────────────────────────────────────────
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for demo
ADAPTER_REPOS = [
    # Replace with your QLoRA adapter repos from HuggingFace Hub
    # "username/adapter-task-a",
    # "username/adapter-task-b",
]
NUM_COMPONENTS = 4  # Subspace dimension
USE_NF4_STORAGE = True  # Save subspace in packed NF4 format


def main():
    # ── Step 1: Load QLoRA base model ──────────────────────────────────
    # In production, load with 4-bit quantization:
    #
    #   from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    #   bnb_config = BitsAndBytesConfig(
    #       load_in_4bit=True,
    #       bnb_4bit_quant_type="nf4",
    #       bnb_4bit_compute_dtype=torch.bfloat16,
    #   )
    #   base_model = AutoModelForCausalLM.from_pretrained(
    #       BASE_MODEL, quantization_config=bnb_config
    #   )
    #
    # For this demo, we simulate with synthetic data:
    print("=== QLoRA + vLoRA Pipeline Demo ===\n")

    # ── Step 2: Load adapters ──────────────────────────────────────────
    from vlora import LoRAWeights, SharedSubspace, VLoRAModel

    print("Creating synthetic adapters (replace with load_adapter_from_hub)...")
    layers = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.v_proj",
    ]
    rank = 8
    dim = 512
    n_adapters = 10

    # Create correlated adapters (simulates real LoRA adapters sharing structure)
    torch.manual_seed(42)
    shared_basis = {l: torch.randn(5, rank * dim) for l in layers}
    adapters = []
    task_ids = []
    for i in range(n_adapters):
        lora_a = {l: (torch.randn(5) @ shared_basis[l]).reshape(rank, dim) for l in layers}
        lora_b = {l: torch.randn(dim, rank) * 0.01 for l in layers}
        adapters.append(LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank))
        task_ids.append(f"task_{i}")
    print(f"  Loaded {n_adapters} adapters, rank={rank}, {len(layers)} layers\n")

    # ── Step 3: Build shared subspace ──────────────────────────────────
    print("Building shared subspace...")
    subspace = SharedSubspace.from_adapters(
        adapters,
        task_ids=task_ids,
        num_components=NUM_COMPONENTS,
    )

    stats = subspace.compression_stats()
    print(f"  Components: {subspace.num_components}")
    print(f"  Compression: {stats['compression_ratio']:.1f}x")
    print(f"  Original params:   {stats['total_params_original']:,}")
    print(f"  Compressed params: {stats['total_params_compressed']:,}\n")

    # ── Step 4: Apply NF4 quantization to subspace ─────────────────────
    print("Quantizing subspace with NF4...")
    subspace.quantize(method="nf4", quantize_loadings=True)
    print("  Done (components + loadings quantized)\n")

    # ── Step 5: Save with packed NF4 storage ───────────────────────────
    import tempfile
    from pathlib import Path

    save_dir = Path(tempfile.mkdtemp()) / "subspace"

    if USE_NF4_STORAGE:
        print("Saving with NF4-packed format...")
        subspace.save_quantized(save_dir)
    else:
        print("Saving with float32 format...")
        subspace.save(save_dir)

    # Compare file sizes
    total_bytes = sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
    print(f"  Saved to: {save_dir}")
    print(f"  Total size: {total_bytes / 1024:.1f} KB\n")

    # ── Step 6: Load and serve ─────────────────────────────────────────
    print("Loading subspace (auto-detects format)...")
    loaded = SharedSubspace.load(save_dir)
    print(f"  {loaded!r}\n")

    # Full-stack compression stats (with hypothetical QLoRA base model)
    full_stats = loaded.full_stack_compression(
        base_model_params=1_100_000_000,  # TinyLlama 1.1B
        base_model_bits=16,
        quantized_bits=4,
    )
    if "total_compression_ratio" in full_stats:
        print("Full-stack compression (QLoRA base + vLoRA adapters):")
        print(f"  Base model: {full_stats['base_model']['compression_ratio']:.1f}x (FP16->NF4)")
        print(f"  Adapters:   {stats['compression_ratio']:.1f}x ({n_adapters} adapters)")
        print(f"  Total:      {full_stats['total_original_bytes']/1e9:.1f} GB -> "
              f"{full_stats['total_compressed_bytes']/1e9:.2f} GB")
        print(f"  Combined:   {full_stats['total_compression_ratio']:.1f}x\n")

    # In production with a real base model:
    #
    #   model = VLoRAModel(base_model, loaded, compute_dtype=torch.bfloat16)
    #   print(f"QLoRA info: {model.qlora_info}")
    #
    #   # Instant task switching
    #   model.set_task("task_0")
    #   output = model(input_ids)
    #
    #   model.set_task("task_5")  # microseconds to switch
    #   output = model(input_ids)

    # Demonstrate reconstruction
    print("Reconstructing adapters from subspace...")
    for tid in ["task_0", "task_5", "task_9"]:
        recon = loaded.reconstruct(tid)
        print(f"  {tid}: {recon!r}")

    print("\nDone!")


if __name__ == "__main__":
    main()
