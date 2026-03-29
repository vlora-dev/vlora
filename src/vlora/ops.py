"""Pure math operations for shared low-rank subspace computation.

All functions are stateless and operate on raw tensors — no adapter
or subspace concepts leak in here.
"""

from __future__ import annotations

import torch
from torch import Tensor

# NF4 quantile levels from QLoRA (Dettmers et al., 2023).
# 16 values chosen so each bin captures equal probability mass
# under a standard normal distribution, minimizing expected
# quantization error for normally-distributed weights.
NF4_QUANT_TABLE = torch.tensor([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
     0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
])


def compute_svd(
    data_matrix: Tensor,
    num_components: int | None = None,
    center: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """SVD on data matrix, returning top-k components.

    Args:
        data_matrix: (N, D) matrix where rows are observations.
        num_components: Number of singular vectors to keep. If None, keep all.
        center: Whether to mean-center rows before SVD.

    Returns:
        components: (k, D) right singular vectors (shared basis).
        singular_values: (k,) corresponding singular values.
        mean: (D,) row mean (zeros if center=False).
    """
    if center:
        mean = data_matrix.mean(dim=0)
        centered = data_matrix - mean
    else:
        mean = torch.zeros(data_matrix.shape[1], dtype=data_matrix.dtype, device=data_matrix.device)
        centered = data_matrix

    # full_matrices=False gives the economy SVD — U is (N, min(N,D))
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    if torch.isnan(S).any():
        raise ValueError(
            "SVD produced NaN singular values. Check input data for "
            "NaN/Inf or try using float64 precision."
        )

    k = num_components if num_components is not None else len(S)
    k = min(k, len(S))

    return Vh[:k], S[:k], mean


def project_onto_subspace(weights: Tensor, components: Tensor) -> Tensor:
    """Project weight vectors onto the subspace basis.

    Args:
        weights: (D,) or (N, D) weight vectors to project.
        components: (k, D) orthonormal basis vectors.

    Returns:
        loadings: (k,) or (N, k) projection coefficients.
    """
    # loadings = weights @ V^T  (since components = V^T, i.e. rows are basis vectors)
    return weights @ components.T


def reconstruct_from_subspace(components: Tensor, loadings: Tensor) -> Tensor:
    """Reconstruct weight vectors from subspace loadings.

    Args:
        components: (k, D) orthonormal basis vectors.
        loadings: (k,) or (N, k) projection coefficients.

    Returns:
        reconstructed: (D,) or (N, D) reconstructed weight vectors.
    """
    return loadings @ components


def gram_schmidt(basis: Tensor, new_vectors: Tensor) -> Tensor:
    """Orthogonalize new_vectors against an existing orthonormal basis.

    Uses QR factorization on the combined matrix for better numerical
    stability and O(k*D) performance vs the naive O(k²*D) approach.
    Appends only directions with non-trivial norm (threshold: 1e-6).

    Args:
        basis: (k, D) existing orthonormal basis.
        new_vectors: (m, D) candidate vectors.

    Returns:
        expanded_basis: (k + n, D) where n <= m new orthogonal directions
            were found.
    """
    k = basis.shape[0]
    combined = torch.cat([basis, new_vectors], dim=0)  # (k+m, D)
    # QR on the transpose: (D, k+m) → Q is (D, min(D, k+m))
    Q, R = torch.linalg.qr(combined.T, mode="reduced")
    # Q columns are the orthonormal basis, rows of Q.T are the vectors
    all_vectors = Q.T  # (min(D, k+m), D)
    # Keep original basis vectors plus any new ones with non-trivial norm.
    # R's diagonal tells us which vectors were linearly independent.
    diag = R.diag().abs()
    # Always keep the first k (original basis), filter the rest
    keep = list(range(k))
    for i in range(k, len(diag)):
        if diag[i] > 1e-6:
            keep.append(i)
    return all_vectors[keep]


def incremental_svd_update(
    components: Tensor,
    singular_values: Tensor,
    mean: Tensor,
    n_seen: int,
    new_data: Tensor,
    max_components: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Incrementally update SVD with new data points.

    Uses the projection-residual approach: project new data onto existing
    basis, compute residual, and if significant, expand the basis with
    the residual direction via QR decomposition.

    Args:
        components: (k, D) current orthonormal basis.
        singular_values: (k,) current singular values.
        mean: (D,) current data mean.
        n_seen: Number of data points seen so far.
        new_data: (m, D) new data points to incorporate.
        max_components: Cap on number of components. If None, allows growth.

    Returns:
        updated_components: (k', D) updated basis.
        updated_singular_values: (k',) updated singular values.
        updated_mean: (D,) updated mean.
        new_n_seen: Updated count.
    """
    m = new_data.shape[0]
    n_total = n_seen + m

    # Update mean incrementally
    new_mean = (mean * n_seen + new_data.sum(dim=0)) / n_total

    # Center new data with updated mean
    centered_new = new_data - new_mean

    # Also adjust for mean shift on existing data:
    # The old centered data had mean=0 relative to old_mean.
    # Relative to new_mean, old data is shifted by (old_mean - new_mean).
    mean_shift = mean - new_mean

    # Project new centered data onto existing basis
    projections = centered_new @ components.T  # (m, k)
    residuals = centered_new - projections @ components  # (m, D)

    # Find new orthogonal directions from residuals via QR
    residual_norms = residuals.norm(dim=1)
    significant = residual_norms > 1e-6
    new_directions = []

    if significant.any():
        sig_residuals = residuals[significant]
        # Orthogonalize residuals against existing basis and each other
        expanded = gram_schmidt(components, sig_residuals)
        new_directions_tensor = expanded[components.shape[0]:]
        if new_directions_tensor.shape[0] > 0:
            new_directions.append(new_directions_tensor)

    # Build augmented system
    if new_directions:
        extra = torch.cat(new_directions, dim=0)
        all_components = torch.cat([components, extra], dim=0)
        # Approximate new singular values for expanded directions
        extra_projections = centered_new @ extra.T  # (m, n_extra)
        extra_svals = extra_projections.norm(dim=0)
        all_svals = torch.cat([singular_values, extra_svals])
    else:
        all_components = components
        # Update singular values to account for new data contribution
        new_contributions = projections.norm(dim=0)
        all_svals = torch.sqrt(singular_values ** 2 + new_contributions ** 2)

    # Account for mean shift effect on singular values
    shift_proj = mean_shift @ all_components.T
    shift_contribution = shift_proj * (n_seen ** 0.5)
    all_svals = torch.sqrt(all_svals ** 2 + shift_contribution ** 2)

    # Sort by singular value magnitude (descending)
    order = all_svals.argsort(descending=True)
    all_components = all_components[order]
    all_svals = all_svals[order]

    # Cap components if needed
    if max_components is not None and all_components.shape[0] > max_components:
        all_components = all_components[:max_components]
        all_svals = all_svals[:max_components]

    return all_components, all_svals, new_mean, n_total


def explained_variance_ratio(singular_values: Tensor) -> Tensor:
    """Compute cumulative explained variance ratio from singular values.

    Args:
        singular_values: (k,) singular values in descending order.

    Returns:
        cumulative_ratio: (k,) cumulative fraction of total variance explained.
    """
    variance = singular_values ** 2
    total = variance.sum()
    if total == 0:
        return torch.zeros_like(variance)
    return torch.cumsum(variance, dim=0) / total


def select_num_components(singular_values: Tensor, threshold: float = 0.6) -> int:
    """Select number of components to explain at least `threshold` variance.

    Args:
        singular_values: (k,) singular values in descending order.
        threshold: Minimum cumulative variance ratio (default 0.6 per paper).

    Returns:
        Number of components needed.
    """
    cumulative = explained_variance_ratio(singular_values)
    # Find first index where cumulative ratio >= threshold
    above = (cumulative >= threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return len(singular_values)
    return above[0].item() + 1


def nf4_quantize_dequantize(
    tensor: Tensor,
    block_size: int = 64,
    double_quant: bool = False,
    double_quant_block_size: int = 256,
) -> Tensor:
    """Simulate NF4 quantization by snapping values to NF4 levels.

    Applies per-block absmax normalization, maps each value to the
    nearest NF4 quantile level, and dequantizes back to float.
    The returned tensor is the same dtype as input but only contains
    values representable in NF4 format.

    Uses ``torch.bucketize`` (binary search) for O(N log 16) lookup
    instead of broadcasting all 16 distances, keeping memory O(N).

    Based on QLoRA (Dettmers et al., 2023, arXiv:2305.14314).

    Args:
        tensor: Tensor to quantize.
        block_size: Number of elements per quantization block.
            Smaller blocks = finer granularity = less error. Default 64.
        double_quant: If True, apply double quantization — quantize the
            per-block absmax scales to FP8, reducing scale overhead from
            0.5 bits/param to ~0.127 bits/param. Introduces a small
            additional precision loss.
        double_quant_block_size: Number of block-scales per second-level
            group for double quantization. Default 256.

    Returns:
        Dequantized tensor with same shape and dtype as input.
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    flat = tensor.detach().float().flatten()
    numel = flat.numel()

    # Pad to multiple of block_size
    remainder = numel % block_size
    if remainder:
        pad = block_size - remainder
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])

    # Reshape into blocks: (num_blocks, block_size)
    blocks = flat.reshape(-1, block_size)

    # Per-block absmax normalization → values in [-1, 1]
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)

    if double_quant:
        # Quantize the absmax scales to FP8 (simulated via round-trip
        # to reduced precision). Group scales into second-level blocks.
        absmax_flat = absmax.flatten()
        n_scales = absmax_flat.numel()
        dq_bs = double_quant_block_size
        pad_scales = (dq_bs - n_scales % dq_bs) % dq_bs
        if pad_scales:
            absmax_flat = torch.cat([absmax_flat, torch.ones(pad_scales, device=absmax_flat.device)])
        scale_blocks = absmax_flat.reshape(-1, dq_bs)
        # Second-level absmax + FP8 simulation (127 levels for E4M3)
        scale_absmax = scale_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        qmax_fp8 = 127.0
        scale_normalized = scale_blocks / scale_absmax
        scale_quantized = (scale_normalized * qmax_fp8).round().clamp(-qmax_fp8, qmax_fp8) / qmax_fp8
        absmax_dq = (scale_quantized * scale_absmax).flatten()[:n_scales]
        absmax = absmax_dq.reshape(-1, 1)

    normalized = blocks / absmax

    # Snap to nearest NF4 level via binary search (memory-efficient).
    table = NF4_QUANT_TABLE.to(device=normalized.device, dtype=normalized.dtype)
    midpoints = (table[:-1] + table[1:]) / 2  # 15 boundaries
    indices = torch.bucketize(normalized, midpoints)
    quantized_normalized = table[indices]

    # Dequantize: scale back by absmax
    dequantized = quantized_normalized * absmax

    # Remove padding and restore shape
    return dequantized.flatten()[:numel].reshape(original_shape).to(original_dtype)
