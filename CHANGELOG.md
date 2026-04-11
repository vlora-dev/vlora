# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-04-11

### Added
- **`VLoRAModel.merge()` / `unmerge()`** — bake LoRA deltas into base model weights for hook-free inference. `is_merged` property tracks state. Errors on quantized base models.
- **Quality warning in `project()`** — optional `warn_threshold` parameter warns when subspace coverage is low, suggesting `absorb()` to expand the basis.
- **PEP 561 `py.typed` marker** — enables type checker support for downstream users.
- **CI hardening** — ruff lint + format check, mypy type checking, pytest-cov with 80% coverage floor, Python 3.12 added to test matrix.
- **Edge case and numerical precision tests** — rank-1 adapters, single-layer adapters, project→reconstruct roundtrip bounds, save→load serialization verification.

### Changed
- **API surface reduced** — low-level ops (`compute_svd`, `project_onto_subspace`, `gram_schmidt`, NF4 functions, `incremental_svd_update`, etc.) removed from top-level `vlora` namespace. Still accessible via `from vlora.ops import ...`.
- **Development status** — classifier updated from "Alpha" to "Production/Stable".

### Migration from 0.3.0
Replace `from vlora import compute_svd` with `from vlora.ops import compute_svd` (and similarly for `project_onto_subspace`, `reconstruct_from_subspace`, `gram_schmidt`, `explained_variance_ratio`, `select_num_components`, `incremental_svd_update`, `NF4_QUANT_TABLE`, `nf4_quantize_dequantize`, `nf4_pack`, `nf4_unpack`). No other breaking changes.

## [0.3.0] - 2026-03-30

### Added
- **NF4 quantization** — 4-bit NormalFloat quantization from QLoRA (Dettmers et al., 2023). `subspace.quantize(method="nf4")` uses 16 quantile levels optimized for normally-distributed weights, with per-block absmax scaling. Lower error than symmetric int4.
- **Double quantization** — quantize per-block NF4 scales to FP8 via `double_quant=True`, reducing scale overhead from 0.5 to ~0.127 bits/param.
- **NF4 packed storage** — `subspace.save_quantized()` packs components as uint8 (two 4-bit indices per byte) for ~7x disk savings. `SharedSubspace.load()` auto-detects format.
- **QLoRA-aware VLoRAModel** — `compute_dtype` parameter for mixed-precision LoRA computation with quantized base models; `qlora_info` property for base model introspection.
- **`full_stack_compression()`** — report combined base model quantization + adapter compression savings.
- **`quantize_loadings` parameter** — optionally quantize per-task loadings (not just components).
- **`nf4_pack` / `nf4_unpack`** — low-level ops for 4-bit packing to uint8.
- **Layer shapes stored in metadata** — `reconstruct()` uses stored shapes instead of deriving from `numel() // rank`, supporting per-layer rank configs.
- **`__repr__` on core objects** — `SharedSubspace`, `TaskProjection`, `LoRAWeights` now print useful info.
- **`adaptive_k` preserved through `absorb()`** — subspaces built with `adaptive_k=True` retain that setting after absorption.
- QLoRA + vLoRA pipeline example (`examples/qlora_pipeline.py`).

### Fixed
- **`absorb_incremental` re-projection bug** — existing tasks were having loadings padded/truncated instead of properly re-projected when the basis rotated. Now reconstructs from old basis and projects onto updated basis.
- **`VLoRACallback` was a no-op** — the HF Trainer callback created an optimizer but never stepped it. Now registers differentiable forward hooks so the Trainer's backward pass produces gradients on loadings, and steps the optimizer in `on_step_end`.
- **TIES merge normalization** — `n / contributor_count` over-scaled output when elements were trimmed. Fixed to `1 / contributor_count`.
- **`__version__` mismatch** — `__init__.py` said 0.1.0 while `pyproject.toml` said 0.2.1.
- **`check_tensor_health` never called** — imported but unused; now wired up after SVD in `from_adapters`.
- **Task ID collision** — `absorb()` and `absorb_incremental()` now warn when overwriting an existing task ID.
- **Filesystem-unsafe task IDs** — `save()` now sanitizes task IDs for filenames (handles `/`, `:`, spaces) with a mapping in metadata for lossless round-trip.
- **`from_adapters_streaming` missing validation** — now checks `len(task_ids) == len(adapter_paths)`.

### Changed
- **`gram_schmidt` uses QR factorization** — replaced O(k^2 * D) inner loop with `torch.linalg.qr` for better performance and numerical stability.
- **VLoRAModel caches module handles** — `_apply_hooks` no longer scans all `named_modules()` on every task switch.
- **VLoRAModel inference hooks wrapped in `torch.no_grad()`** — prevents unnecessary autograd tracking.
- **NF4 quantization uses `torch.bucketize`** — replaced O(N*16) distance broadcast with binary search, reducing memory from O(N*16) to O(N).
- **`_LORA_KEY_RE` handles multi-adapter PEFT format** — supports `base_model.model.{layer}.lora_A.{adapter_name}.weight`.
- **`save_adapter` no longer hardcodes `CAUSAL_LM`** — task type left for PEFT to infer.
- Repo URL updated to `github.com/vlora-dev/vlora`.

## [0.2.1] - 2026-02-10

Initial public release on PyPI as `vlora-dev`.

### Added
- `SharedSubspace` — 3-step algorithm: from_adapters, project, absorb
- `VLoRAModel` — inference wrapper with forward hooks
- `SubspaceTrainer` — loadings-only training
- `TaskRouter` — per-input adapter routing
- `task_arithmetic`, `ties_merge`, `dare_merge` — adapter merging
- Analysis tools: similarity matrix, clustering, outlier detection
- CLI with 9 commands
- HuggingFace Trainer integration via `VLoRACallback`
- Streaming and incremental subspace construction
