"""vlora CLI — command-line interface for adapter management."""

from __future__ import annotations

import json as json_mod
import logging
import time
from pathlib import Path

import click

from vlora.io import load_adapter, save_adapter
from vlora.ops import explained_variance_ratio
from vlora.subspace import SharedSubspace


@click.group()
@click.version_option(package_name="vlora")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool):
    """vLoRA — Shared low-rank subspaces for LoRA adapter management."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s: %(message)s",
    )


@cli.command()
@click.argument("subspace_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def info(subspace_path: str, as_json: bool):
    """Show subspace stats: tasks, layers, compression ratios."""
    sub = SharedSubspace.load(subspace_path)

    stats = sub.compression_stats()

    # Variance explained (first layer)
    first_layer = sub.layer_names[0]
    var_a = explained_variance_ratio(sub.singular_values_a[first_layer])
    var_b = explained_variance_ratio(sub.singular_values_b[first_layer])
    k = sub.num_components
    var_a_val = var_a[k - 1].item() if k <= len(var_a) else None
    var_b_val = var_b[k - 1].item() if k <= len(var_b) else None

    if as_json:
        output = {
            "path": subspace_path,
            "num_components": sub.num_components,
            "rank": sub.rank,
            "num_layers": len(sub.layer_names),
            "num_tasks": len(sub.tasks),
            "task_ids": sorted(sub.tasks.keys()),
            "variance_explained_a": var_a_val,
            "variance_explained_b": var_b_val,
            **stats,
        }
        click.echo(json_mod.dumps(output, indent=2, default=str))
        return

    click.echo(f"\n  Subspace: {subspace_path}")
    click.echo(f"  Components (k): {sub.num_components}")
    click.echo(f"  LoRA rank: {sub.rank}")
    click.echo(f"  Layers: {len(sub.layer_names)}")
    click.echo(f"  Tasks: {len(sub.tasks)}")

    if sub.tasks:
        click.echo("\n  Task IDs:")
        for tid in sorted(sub.tasks.keys()):
            click.echo(f"    - {tid}")

    click.echo(f"\n  Variance explained (first layer, k={k}):")
    if var_a_val is not None:
        click.echo(f"    A: {var_a_val:.1%}")
    if var_b_val is not None:
        click.echo(f"    B: {var_b_val:.1%}")

    # Compression estimate
    n = len(sub.tasks)
    if n > 0:
        ratio = stats["compression_ratio"]
        click.echo(f"\n  Compression ratio: {ratio:.1f}x ({n} adapters)")

    click.echo()


@cli.command()
@click.argument("adapter_dirs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory for shared subspace.")
@click.option("-k", "--num-components", type=int, default=None, help="Number of basis components (auto if not set).")
@click.option("--variance-threshold", type=float, default=0.6, help="Variance threshold for auto k selection.")
@click.option("--adaptive-k", is_flag=True, help="Use per-layer adaptive k selection.")
def compress(adapter_dirs: tuple[str, ...], output: str, num_components: int | None, variance_threshold: float, adaptive_k: bool):
    """Build shared subspace from adapter directories."""
    click.echo(f"\n  Loading {len(adapter_dirs)} adapters...")

    adapters = []
    task_ids = []
    for d in adapter_dirs:
        path = Path(d)
        adapters.append(load_adapter(path))
        task_ids.append(path.name)
        click.echo(f"    Loaded: {path.name}")

    click.echo("  Building subspace...")
    sub = SharedSubspace.from_adapters(
        adapters,
        task_ids=task_ids,
        num_components=num_components,
        variance_threshold=variance_threshold,
        adaptive_k=adaptive_k,
    )

    sub.save(output)
    click.echo(f"  Saved to: {output}")
    click.echo(f"  Components: {sub.num_components}, Layers: {len(sub.layer_names)}, Tasks: {len(sub.tasks)}")
    click.echo()


@cli.command("export")
@click.argument("subspace_path", type=click.Path(exists=True))
@click.argument("task_id")
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory for PEFT adapter.")
@click.option("--alpha", type=float, default=None, help="LoRA alpha for adapter_config.json (default: same as rank).")
@click.option("--base-model", type=str, default=None, help="Base model name for adapter_config.json.")
@click.option("--target-modules", type=str, default=None, help="Comma-separated target modules for adapter_config.json.")
def export_cmd(subspace_path: str, task_id: str, output: str, alpha: float | None, base_model: str | None, target_modules: str | None):
    """Reconstruct a task adapter to PEFT format."""
    sub = SharedSubspace.load(subspace_path)

    if task_id not in sub.tasks:
        available = ", ".join(sorted(sub.tasks.keys()))
        raise click.ClickException(f"Unknown task '{task_id}'. Available: {available}")

    click.echo(f"\n  Reconstructing '{task_id}'...")
    weights = sub.reconstruct(task_id)

    # Enrich metadata for serving compatibility
    if alpha is not None:
        weights.metadata["lora_alpha"] = alpha
    if base_model is not None:
        weights.metadata["base_model_name_or_path"] = base_model
    if target_modules is not None:
        weights.metadata["target_modules"] = [m.strip() for m in target_modules.split(",")]

    save_adapter(weights, output)
    click.echo(f"  Exported to: {output}")
    click.echo()


@cli.command()
@click.argument("subspace_path", type=click.Path(exists=True))
@click.argument("adapter_dir", type=click.Path(exists=True))
@click.option("--task-id", required=True, help="ID for the new task.")
@click.option("--incremental", is_flag=True, help="Use fast incremental absorb (approximate).")
def add(subspace_path: str, adapter_dir: str, task_id: str, incremental: bool):
    """Absorb a new adapter into an existing subspace."""
    sub = SharedSubspace.load(subspace_path)

    click.echo(f"\n  Loading adapter from {adapter_dir}...")
    adapter = load_adapter(adapter_dir)

    method = "incremental" if incremental else "full SVD"
    click.echo(f"  Absorbing as '{task_id}' ({method})...")
    if incremental:
        sub.absorb_incremental(adapter, task_id)
    else:
        sub.absorb(adapter, task_id)

    sub.save(subspace_path)
    click.echo(f"  Subspace updated. Tasks: {len(sub.tasks)}")
    click.echo()


@cli.command()
@click.argument("adapter_dirs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--threshold", type=float, default=0.9, help="Similarity threshold for clustering.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def analyze(adapter_dirs: tuple[str, ...], threshold: float, as_json: bool):
    """Analyze adapter similarity and find redundant clusters."""
    from vlora.analysis import compute_similarity_matrix, find_clusters

    adapters = []
    names = []
    for d in adapter_dirs:
        path = Path(d)
        adapters.append(load_adapter(path))
        names.append(path.name)

    if len(adapters) < 2:
        raise click.ClickException("Need at least 2 adapters for analysis.")

    sim = compute_similarity_matrix(adapters)
    clusters = find_clusters(sim, threshold=threshold)

    if as_json:
        sim_dict = {}
        for i, name_i in enumerate(names):
            sim_dict[name_i] = {names[j]: sim[i, j].item() for j in range(len(names))}
        output = {
            "similarity_matrix": sim_dict,
            "clusters": [[names[i] for i in c] for c in clusters],
            "threshold": threshold,
            "redundant_count": sum(len(c) - 1 for c in clusters if len(c) > 1),
        }
        click.echo(json_mod.dumps(output, indent=2))
        return

    click.echo(f"\n  Loading {len(adapter_dirs)} adapters...")
    for n in names:
        click.echo(f"    Loaded: {n}")

    click.echo("\n  Pairwise Cosine Similarity:")
    header = "  " + " " * 20 + "  ".join(f"{n[:8]:>8}" for n in names)
    click.echo(header)
    for i, name in enumerate(names):
        row = f"  {name[:20]:<20}"
        for j in range(len(names)):
            val = sim[i, j].item()
            row += f"  {val:8.3f}"
        click.echo(row)

    clusters = find_clusters(sim, threshold=threshold)
    click.echo(f"\n  Clusters (threshold={threshold}):")
    for ci, cluster in enumerate(clusters):
        members = ", ".join(names[i] for i in cluster)
        click.echo(f"    Cluster {ci + 1}: {members}")

    redundant = sum(len(c) - 1 for c in clusters if len(c) > 1)
    if redundant > 0:
        click.echo(f"\n  Potentially redundant adapters: {redundant}")
    else:
        click.echo(f"\n  No redundant adapters found at threshold={threshold}")

    click.echo()


@cli.command()
@click.argument("subspace_path", type=click.Path(exists=True))
def validate(subspace_path: str):
    """Run health checks on a subspace."""
    import torch

    sub = SharedSubspace.load(subspace_path)
    issues: dict[str, list[str]] = {"errors": [], "warnings": []}

    click.echo(f"\n  Validating: {subspace_path}")
    click.echo(f"  Tasks: {len(sub.tasks)}, Layers: {len(sub.layer_names)}, k={sub.num_components}")

    # Check for NaN/Inf in components and means
    for layer in sub.layer_names:
        for name, tensor in [
            (f"{layer}.components_a", sub.components_a[layer]),
            (f"{layer}.components_b", sub.components_b[layer]),
            (f"{layer}.means_a", sub.means_a[layer]),
            (f"{layer}.means_b", sub.means_b[layer]),
        ]:
            if torch.isnan(tensor).any():
                issues["errors"].append(f"NaN in {name}")
            if torch.isinf(tensor).any():
                issues["errors"].append(f"Inf in {name}")

    # Check component orthonormality
    for layer in sub.layer_names:
        for side, comps in [("A", sub.components_a[layer]), ("B", sub.components_b[layer])]:
            if comps.shape[0] > 0:
                gram = comps @ comps.T
                eye = torch.eye(comps.shape[0])
                err = (gram - eye).abs().max().item()
                if err > 0.01:
                    issues["warnings"].append(
                        f"{layer}.{side} components not orthonormal (max error: {err:.4f})"
                    )

    # Check task loadings consistency
    for tid, proj in sub.tasks.items():
        for layer in sub.layer_names:
            k_a = sub.components_a[layer].shape[0]
            k_b = sub.components_b[layer].shape[0]
            if proj.loadings_a[layer].shape[0] != k_a:
                issues["errors"].append(
                    f"Task '{tid}' loadings_a mismatch at {layer}: "
                    f"expected {k_a}, got {proj.loadings_a[layer].shape[0]}"
                )
            if proj.loadings_b[layer].shape[0] != k_b:
                issues["errors"].append(
                    f"Task '{tid}' loadings_b mismatch at {layer}: "
                    f"expected {k_b}, got {proj.loadings_b[layer].shape[0]}"
                )

    # Report
    if issues["errors"]:
        click.echo(f"\n  ERRORS ({len(issues['errors'])}):")
        for err in issues["errors"]:
            click.echo(f"    [ERROR] {err}")
    if issues["warnings"]:
        click.echo(f"\n  WARNINGS ({len(issues['warnings'])}):")
        for warn in issues["warnings"]:
            click.echo(f"    [WARN]  {warn}")
    if not issues["errors"] and not issues["warnings"]:
        click.echo("\n  All checks passed.")

    click.echo()


@cli.command()
@click.argument("subspace_path", type=click.Path(exists=True))
@click.argument("task_a")
@click.argument("task_b")
def diff(subspace_path: str, task_a: str, task_b: str):
    """Compare two tasks within a subspace."""
    import torch

    sub = SharedSubspace.load(subspace_path)

    for tid in [task_a, task_b]:
        if tid not in sub.tasks:
            available = ", ".join(sorted(sub.tasks.keys()))
            raise click.ClickException(f"Unknown task '{tid}'. Available: {available}")

    click.echo(f"\n  Comparing '{task_a}' vs '{task_b}'")

    recon_a = sub.reconstruct(task_a)
    recon_b = sub.reconstruct(task_b)

    click.echo(f"\n  {'Layer':<30} {'L2 Dist (A)':>12} {'L2 Dist (B)':>12} {'Cosine (A)':>12} {'Cosine (B)':>12}")
    click.echo(f"  {'─' * 78}")

    for layer in sub.layer_names:
        a_a = recon_a.lora_a[layer].flatten()
        a_b = recon_a.lora_b[layer].flatten()
        b_a = recon_b.lora_a[layer].flatten()
        b_b = recon_b.lora_b[layer].flatten()

        l2_a = (a_a - b_a).norm().item()
        l2_b = (a_b - b_b).norm().item()
        cos_a = torch.nn.functional.cosine_similarity(a_a.unsqueeze(0), b_a.unsqueeze(0)).item()
        cos_b = torch.nn.functional.cosine_similarity(a_b.unsqueeze(0), b_b.unsqueeze(0)).item()

        name = layer[:30]
        click.echo(f"  {name:<30} {l2_a:>12.4f} {l2_b:>12.4f} {cos_a:>12.4f} {cos_b:>12.4f}")

    click.echo()


@cli.command()
@click.argument("subspace_path", type=click.Path(exists=True))
def benchmark(subspace_path: str):
    """Benchmark subspace operations: reconstruct, project, absorb."""

    sub = SharedSubspace.load(subspace_path)
    task_ids = sorted(sub.tasks.keys())

    if not task_ids:
        raise click.ClickException("Subspace has no tasks to benchmark.")

    click.echo(f"\n  Benchmarking: {subspace_path}")
    click.echo(f"  Tasks: {len(task_ids)}, Layers: {len(sub.layer_names)}, k={sub.num_components}")

    # Benchmark reconstruct
    tid = task_ids[0]
    times = []
    for _ in range(10):
        start = time.perf_counter()
        sub.reconstruct(tid)
        times.append(time.perf_counter() - start)
    avg_recon = sum(times) / len(times)
    click.echo(f"\n  reconstruct('{tid}'): {avg_recon * 1000:.2f} ms (avg of 10)")

    # Benchmark project
    recon = sub.reconstruct(tid)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        sub.project(recon, "bench_proj")
        times.append(time.perf_counter() - start)
    avg_proj = sum(times) / len(times)
    click.echo(f"  project():            {avg_proj * 1000:.2f} ms (avg of 10)")

    # Benchmark compression stats
    start = time.perf_counter()
    stats = sub.compression_stats()
    stats_time = time.perf_counter() - start
    click.echo(f"  compression_stats():  {stats_time * 1000:.2f} ms")
    click.echo(f"\n  Compression ratio: {stats['compression_ratio']:.1f}x")

    click.echo()


@cli.command()
@click.argument("adapter_dirs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory for merged adapter.")
@click.option("--method", type=click.Choice(["average", "ties", "dare"]), default="average", help="Merge method.")
@click.option("--weights", type=str, default=None, help="Comma-separated per-adapter weights (e.g. '0.7,0.3').")
@click.option("--density", type=float, default=0.5, help="TIES density: fraction of values to keep.")
@click.option("--drop-rate", type=float, default=0.5, help="DARE drop rate: probability of dropping each element.")
@click.option("--seed", type=int, default=None, help="Random seed for DARE reproducibility.")
def merge(adapter_dirs: tuple[str, ...], output: str, method: str, weights: str | None, density: float, drop_rate: float, seed: int | None):
    """Merge multiple adapters into one using task arithmetic, TIES, or DARE."""
    from vlora.merge import MERGE_METHODS

    click.echo(f"\n  Loading {len(adapter_dirs)} adapters...")
    adapters = []
    for d in adapter_dirs:
        path = Path(d)
        adapters.append(load_adapter(path))
        click.echo(f"    Loaded: {path.name}")

    if len(adapters) < 2:
        raise click.ClickException("Need at least 2 adapters to merge.")

    parsed_weights = None
    if weights is not None:
        parsed_weights = [float(w.strip()) for w in weights.split(",")]

    click.echo(f"  Merging with method={method}...")

    fn = MERGE_METHODS[method]
    if method == "ties":
        merged = fn(adapters, density=density, weights=parsed_weights)  # type: ignore[operator]
    elif method == "dare":
        merged = fn(adapters, drop_rate=drop_rate, weights=parsed_weights, seed=seed)  # type: ignore[operator]
    else:
        merged = fn(adapters, weights=parsed_weights)  # type: ignore[operator]

    save_adapter(merged, output)
    click.echo(f"  Merged adapter saved to: {output}")
    click.echo(f"  Layers: {len(merged.layer_names)}, Rank: {merged.rank}")
    click.echo()
