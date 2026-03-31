<p align="center">
  <img src="logo.png" alt="vLoRA" width="400">
</p>

<p align="center">
  <strong>Various LoRA adapters. One shared basis.</strong>
</p>

Your adapters share more structure than you think. vLoRA finds the common basis and stores each adapter as a tiny coefficient vector — up to 122× compression at scale. Based on the [Share paper](https://arxiv.org/abs/2602.06043).

## Install

```bash
pip install vlora-dev
```

Or from source:
```bash
git clone https://github.com/vlora-dev/vlora.git
cd vlora
pip install -e ".[dev]"
```

## Quickstart

```python
from vlora import SharedSubspace, load_adapter

# Step 1: Build shared subspace from existing adapters
adapters = [load_adapter(f"adapters/task_{i}") for i in range(5)]
subspace = SharedSubspace.from_adapters(adapters, num_components=16)

# Step 2: Project a new adapter (only stores small loadings vector)
new_adapter = load_adapter("adapters/new_task")
projection = subspace.project(new_adapter, task_id="new_task")
subspace.add_task(projection)

# Step 3: Absorb — recompute basis to include new adapter
subspace.absorb(load_adapter("adapters/another_task"), new_task_id="another")

# Reconstruct any task back to full LoRA weights
weights = subspace.reconstruct("new_task")

# Save / load
subspace.save("shared_subspace/")
subspace = SharedSubspace.load("shared_subspace/")
```

## CLI

vlora ships with 9 commands for common workflows:

```bash
# Build a shared subspace from adapter directories
vlora compress adapters/task_0 adapters/task_1 adapters/task_2 -o shared_subspace/

# Inspect a subspace (--json for machine-readable output)
vlora info shared_subspace/

# Export a task back to PEFT format (vLLM/TGI compatible)
vlora export shared_subspace/ task_0 -o exported_adapter/ \
  --alpha 32 --base-model meta-llama/Llama-3-8B --target-modules q_proj,v_proj

# Add a new adapter to an existing subspace
vlora add shared_subspace/ adapters/new_task --task-id new_task --incremental

# Analyze adapter similarity and clustering
vlora analyze adapters/task_0 adapters/task_1 adapters/task_2

# Merge adapters using task arithmetic, TIES, or DARE
vlora merge adapters/task_0 adapters/task_1 adapters/task_2 \
  -o merged/ --method ties --density 0.5

# Health check a subspace (NaN, orthonormality, loadings consistency)
vlora validate shared_subspace/

# Compare two tasks within a subspace
vlora diff shared_subspace/ task_0 task_1

# Benchmark subspace operations
vlora benchmark shared_subspace/
```

## Multi-Task Inference

Wrap any PyTorch model with `VLoRAModel` for on-the-fly adapter switching:

```python
from vlora import VLoRAModel, SharedSubspace

subspace = SharedSubspace.load("shared_subspace/")
model = VLoRAModel(base_model, subspace, lora_alpha=32)  # or scaling=alpha/rank

# Switch adapters instantly — reconstructed from compressed loadings
model.set_task("task_0")
output = model(input_ids)

model.set_task("task_1")  # cached if same task
output = model(input_ids)

print(model.available_tasks)  # ["task_0", "task_1", ...]
```

## QLoRA Support

vLoRA has first-class support for [QLoRA](https://arxiv.org/abs/2305.14314) workflows. QLoRA compresses the **base model** (FP16 → 4-bit NF4), while vLoRA compresses the **adapter space** — these are orthogonal and stack multiplicatively.

### NF4 Quantization

Quantize subspace components using the same NF4 data type from QLoRA — 16 quantile levels optimized for normally-distributed weights:

```python
# NF4 quantization (better than symmetric int4 for normal-ish weights)
subspace.quantize(method="nf4")

# With double quantization (quantize the per-block scales too)
subspace.quantize(method="nf4", double_quant=True)

# Also quantize loadings (effective when loadings are approximately normal)
subspace.quantize(method="nf4", quantize_loadings=True)
```

### Packed NF4 Storage

Save subspace in packed 4-bit format for ~7× disk savings:

```python
# Save: packs components as uint8 (two 4-bit values per byte)
subspace.save_quantized("shared_subspace/")

# Load: auto-detects format, dequantizes on the fly
subspace = SharedSubspace.load("shared_subspace/")
```

### QLoRA Base Model

`VLoRAModel` works with quantized base models loaded via bitsandbytes:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from vlora import VLoRAModel, SharedSubspace

# Load 4-bit base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained("model-name", quantization_config=bnb_config)

# Wrap with vLoRA — compute_dtype ensures LoRA math runs in BF16
subspace = SharedSubspace.load("shared_subspace/")
model = VLoRAModel(base_model, subspace, compute_dtype=torch.bfloat16)

print(model.qlora_info)  # {'quantized': True, 'method': 'nf4', ...}
model.set_task("task_0")
output = model(input_ids)
```

### Full-Stack Compression

Report combined savings across base model quantization and adapter compression:

```python
stats = subspace.full_stack_compression(
    base_model_params=7_000_000_000,  # 7B model
    base_model_bits=16,               # original FP16
    quantized_bits=4,                 # QLoRA NF4
)
# → {'total_compression_ratio': 4.0, 'total_original_bytes': 14.0 GB, ...}
```

See [`examples/qlora_pipeline.py`](examples/qlora_pipeline.py) for a complete end-to-end example.

## Training in the Subspace

Train only the loadings vector (k params per layer) instead of full LoRA matrices — 100×+ parameter reduction:

```python
from vlora import SharedSubspace, orthogonal_init, SubspaceTrainer

subspace = SharedSubspace.load("shared_subspace/")
orthogonal_init(subspace, "new_task")  # initialize near-zero

trainer = SubspaceTrainer(subspace, "new_task", lr=1e-3)
print(f"Trainable params: {trainer.num_trainable_params}")  # e.g. 192 vs 200K

for batch in dataloader:
    loss = compute_loss(model, batch)
    trainer.step(loss)

trainer.write_back()  # persist learned loadings
subspace.save("updated_subspace/")
```

## Task Router

Automatically blend adapters per input using a lightweight router:

```python
from vlora import TaskRouter, SharedSubspace

subspace = SharedSubspace.load("shared_subspace/")
router = TaskRouter.from_subspace(subspace, input_dim=4096)

# Router produces soft blend weights over tasks
x = get_input_embedding(batch)  # (B, 4096)
blended = router.blend_loadings(x, subspace)
subspace.tasks["__routed__"] = blended
recon = subspace.reconstruct("__routed__")
```

## Adapter Analysis

Analyze relationships between adapters before compression:

```python
from vlora import load_adapter, compute_similarity_matrix, find_clusters, adapter_diff

adapters = [load_adapter(f"adapters/task_{i}") for i in range(10)]

# Pairwise cosine similarity
sim_matrix = compute_similarity_matrix(adapters)

# Find redundant adapter groups
clusters = find_clusters(sim_matrix, threshold=0.9)

# Per-layer comparison of two adapters
diff = adapter_diff(adapters[0], adapters[1])
```

## Adapter Merging

Merge multiple adapters into one using state-of-the-art techniques:

```python
from vlora import load_adapter, task_arithmetic, ties_merge, dare_merge

adapters = [load_adapter(f"adapters/task_{i}") for i in range(3)]

# Simple weighted average
merged = task_arithmetic(adapters, weights=[0.5, 0.3, 0.2])

# TIES: trim small values, elect sign by majority, average (reduces interference)
merged = ties_merge(adapters, density=0.5)

# DARE: randomly drop & rescale before averaging (sparsification regularizer)
merged = dare_merge(adapters, drop_rate=0.5, seed=42)
```

## Advanced Compression

```python
# Adaptive k: different components per layer based on explained variance
subspace = SharedSubspace.from_adapters(adapters, adaptive_k=True, variance_threshold=0.9)

# Quantize components — symmetric (int8/int4) or NF4
subspace.quantize(bits=8)                        # symmetric int8
subspace.quantize(method="nf4")                  # NF4 4-bit (better for normal weights)
subspace.quantize(method="nf4", double_quant=True)  # + quantize the scales

# Check compression stats
stats = subspace.compression_stats()
print(f"Compression ratio: {stats['compression_ratio']:.1f}×")
print(f"Compressed: {stats['total_params_compressed']:,} params")
print(f"Original:   {stats['total_params_original']:,} params")
```

## Incremental Updates

Scale to thousands of adapters without loading them all at once:

```python
# Streaming: load adapters one at a time from disk
subspace = SharedSubspace.from_adapters_streaming(
    adapter_paths, num_components=8
)

# Incremental absorb: fast O(1) update without full SVD recompute
subspace.absorb_incremental(new_adapter, "new_task")

# Move to GPU / change precision
subspace.to(device="cuda", dtype=torch.float16)
```

## The 3-Step Algorithm

| Step | Method | What happens |
|------|--------|-------------|
| **1. Initialize** | `SharedSubspace.from_adapters()` | SVD on stacked weight matrices → shared basis |
| **2. Project** | `subspace.project()` | New adapter → small loadings vector |
| **3. Absorb** | `subspace.absorb()` | Incorporate new adapter, recompute basis |

## API Reference

### Core

- **`SharedSubspace`** — Central state container. Holds per-layer basis and per-task loadings.
  - `.from_adapters(adapters, ...)` — Build from existing adapters
  - `.from_adapters_streaming(paths, ...)` — Build one adapter at a time from disk
  - `.project(adapter, task_id)` → `TaskProjection`
  - `.add_task(projection)` — Register a projected task
  - `.reconstruct(task_id)` → `LoRAWeights`
  - `.absorb(adapter, task_id)` — Incorporate + recompute (full SVD)
  - `.absorb_incremental(adapter, task_id)` — Fast incremental update
  - `.get_trainable_params(task_id)` — For training integration
  - `.quantize(bits=8, method="symmetric")` — Quantize components (int8/int4/NF4)
  - `.compression_stats()` — Compression ratio and parameter counts
  - `.full_stack_compression(base_model_params)` — Combined base + adapter stats
  - `.to(device, dtype)` — Move tensors to device/dtype
  - `.save(path)` / `.save_quantized(path)` / `.load(path)` — Serialization (NF4-packed auto-detected)

### Model Integration

- **`VLoRAModel(base_model, subspace, lora_alpha=None, compute_dtype=None)`** — Inference wrapper with forward hooks
  - `.qlora_info` — Base model quantization metadata
  - `.set_task(task_id)` — Switch adapter (cached)
  - `.clear_task()` — Remove adapter
  - `.available_tasks` — List task IDs
  - `.reconstruct_state_dict(task_id)` — Get delta weight dict
  - `.compile()` — torch.compile the base model for faster inference

### Training

- **`orthogonal_init(subspace, task_id)`** — Initialize new task with small loadings
- **`SubspaceTrainer(subspace, task_id)`** — Optimizer wrapper for loadings-only training
  - `.step(loss)` — Backprop + update
  - `.write_back()` — Persist to subspace

### Router

- **`TaskRouter(input_dim, num_tasks)`** — Lightweight adapter routing MLP
  - `.from_subspace(subspace, input_dim)` — Auto-create from subspace
  - `.blend_loadings(x, subspace)` — Per-input adapter blending

### Merging

- **`task_arithmetic(adapters, weights=None)`** — Weighted average merge
- **`ties_merge(adapters, density=0.5, weights=None)`** — Trim + elect sign + merge
- **`dare_merge(adapters, drop_rate=0.5, weights=None, seed=None)`** — Drop and rescale merge

### Analysis

- **`compute_similarity_matrix(adapters)`** — Pairwise cosine similarity
- **`find_clusters(sim_matrix, threshold)`** — Greedy clustering
- **`adapter_diff(a, b)`** — Per-layer L2 distance + cosine similarity
- **`subspace_coverage(subspace, adapter)`** — How well subspace represents an adapter
- **`find_outliers(adapters, threshold)`** — Detect statistical outlier adapters

### I/O

- **`load_adapter(path)`** — Load PEFT adapter from disk (safetensors)
- **`load_adapter_from_hub(repo_id)`** — Load from HuggingFace Hub
- **`save_adapter(weights, path)`** — Save back to PEFT format

### Pipeline (convenience)

- **`init_subspace(paths, ...)`** — Load + build in one call
- **`absorb_task(subspace, path, task_id)`** — Load + absorb
- **`extract_adapter(subspace, task_id, path)`** — Reconstruct + save

### Math ops

- `compute_svd`, `project_onto_subspace`, `reconstruct_from_subspace`
- `gram_schmidt`, `explained_variance_ratio`, `select_num_components`
- `incremental_svd_update`
- `nf4_quantize_dequantize`, `nf4_pack`, `nf4_unpack` — NF4 quantization (QLoRA)

## Benchmarks — Real-World Adapters

Tested with 8 [Lots-of-LoRAs](https://huggingface.co/Lots-of-LoRAs) adapters (Mistral-7B, rank 16, 96 layers each):

**Variance explained** — the B matrices share structure much more strongly:

| k | Variance (A) | Variance (B) |
|---|-------------|-------------|
| 1 | 0.19 | 0.43 |
| 2 | 0.37 | 0.73 |
| 4 | 0.69 | 0.95 |
| 6 | 1.00 | 1.00 |

**Reconstruction error** (relative L2 norm):

| k | Mean Error | Max Error |
|---|-----------|-----------|
| 1 | 0.826 | 0.938 |
| 4 | 0.387 | 0.846 |
| 6 | 0.000002 | 0.000003 |

**Compression at scale** — shared basis is a one-time cost; each new adapter adds only k loadings per layer:

| N adapters | Full (MB) | vLoRA (MB) | Ratio |
|-----------|----------|-----------|-------|
| 8 | 288 | 288 | 1.0× |
| 100 | 3,600 | 289 | 12.5× |
| 1,000 | 36,000 | 293 | 122.8× |

Run the benchmark yourself:
```bash
pip install vlora-dev[hub]
python examples/real_adapters.py
```

## HuggingFace Trainer Integration

Train in the subspace directly with HuggingFace Trainer:

```python
from vlora import SharedSubspace, orthogonal_init
from vlora.integrations.huggingface import VLoRACallback

subspace = SharedSubspace.load("shared_subspace/")
orthogonal_init(subspace, "new_task")

callback = VLoRACallback(subspace, "new_task", lr=1e-3)
trainer = Trainer(model=base_model, args=args, callbacks=[callback])
trainer.train()
subspace.save("updated_subspace/")
```

## Documentation

- [Quickstart notebook](examples/quickstart.ipynb) — try vlora in Google Colab
- [Migration from PEFT](docs/migration_from_peft.md) — integrate into existing workflow
- [vLLM guide](docs/guide_vllm.md) — serve with vLLM
- [TGI guide](docs/guide_tgi.md) — serve with TGI
- [Ollama guide](docs/guide_ollama.md) — local inference via GGUF

## Dependencies

- `torch >= 2.0`
- `safetensors >= 0.4`
- `click >= 8.0`
- `huggingface-hub >= 0.20` *(optional, `pip install vlora-dev[hub]`)*
- `transformers >= 4.38` *(optional, `pip install vlora-dev[hf]`)*

## Citation

```bibtex
@article{share2025,
  title={Share: Shared Low-Rank Subspaces for Efficient LoRA Adapter Management},
  year={2025},
  eprint={2602.06043},
  archivePrefix={arXiv},
}
```

## License

Apache 2.0
