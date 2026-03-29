"""Tests for vlora.cli — command-line interface."""

import json
import tempfile
from pathlib import Path

import torch
import pytest
from click.testing import CliRunner

from vlora.cli import cli
from vlora.io import LoRAWeights, save_adapter
from vlora.subspace import SharedSubspace


def _make_adapter_dir(tmp_path: Path, name: str, layers=None, rank=4, dim=64) -> Path:
    """Create a synthetic adapter on disk in PEFT format."""
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

    lora_a = {l: torch.randn(rank, dim) for l in layers}
    lora_b = {l: torch.randn(dim, rank) for l in layers}
    adapter = LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank)

    adapter_dir = tmp_path / name
    save_adapter(adapter, adapter_dir)
    return adapter_dir


def _make_correlated_adapter_dirs(tmp_path: Path, n=3, layers=None, rank=4, dim=64) -> list[Path]:
    """Create n correlated adapter dirs (share structure for good SVD)."""
    if layers is None:
        layers = ["layer.0.q_proj", "layer.0.v_proj"]

    shared_a = {l: torch.randn(3, rank * dim) for l in layers}
    shared_b = {l: torch.randn(3, dim * rank) for l in layers}

    dirs = []
    for i in range(n):
        lora_a = {}
        lora_b = {}
        for l in layers:
            coeffs_a = torch.randn(3)
            coeffs_b = torch.randn(3)
            lora_a[l] = (coeffs_a @ shared_a[l] + torch.randn(rank * dim) * 0.01).reshape(rank, dim)
            lora_b[l] = (coeffs_b @ shared_b[l] + torch.randn(dim * rank) * 0.01).reshape(dim, rank)
        adapter = LoRAWeights(layer_names=layers, lora_a=lora_a, lora_b=lora_b, rank=rank)
        d = tmp_path / f"adapter_{i}"
        save_adapter(adapter, d)
        dirs.append(d)

    return dirs


class TestCliHelp:
    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "vLoRA" in result.output

    def test_has_all_commands(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        for cmd in ["info", "compress", "export", "add", "analyze", "validate", "diff", "benchmark", "merge"]:
            assert cmd in result.output


class TestCompress:
    def test_compress_creates_subspace(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        output = tmp_path / "subspace"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(output), "-k", "2"]
        result = runner.invoke(cli, ["compress"] + args)

        assert result.exit_code == 0, result.output
        assert (output / "subspace.safetensors").exists()
        assert (output / "subspace_meta.json").exists()


class TestInfo:
    def test_info_displays_stats(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        output = tmp_path / "subspace"

        runner = CliRunner()
        # First compress
        args = [str(d) for d in dirs] + ["-o", str(output), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        # Then info
        result = runner.invoke(cli, ["info", str(output)])
        assert result.exit_code == 0, result.output
        assert "Components (k): 2" in result.output
        assert "Tasks: 3" in result.output
        assert "Layers:" in result.output


class TestExport:
    def test_export_creates_adapter(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        sub_path = tmp_path / "subspace"
        export_path = tmp_path / "exported"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(sub_path), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        result = runner.invoke(cli, ["export", str(sub_path), "adapter_0", "-o", str(export_path)])
        assert result.exit_code == 0, result.output
        assert (export_path / "adapter_model.safetensors").exists()

    def test_export_unknown_task_fails(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        sub_path = tmp_path / "subspace"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(sub_path), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        result = runner.invoke(cli, ["export", str(sub_path), "nonexistent", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "Unknown task" in result.output


class TestAdd:
    def test_add_absorbs_new_adapter(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        sub_path = tmp_path / "subspace"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(sub_path), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        new_adapter = _make_adapter_dir(tmp_path, "new_adapter")
        result = runner.invoke(cli, ["add", str(sub_path), str(new_adapter), "--task-id", "new"])
        assert result.exit_code == 0, result.output
        assert "Tasks: 4" in result.output


class TestMerge:
    def test_merge_average(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        output = tmp_path / "merged"

        runner = CliRunner()
        result = runner.invoke(cli, ["merge"] + [str(d) for d in dirs] + ["-o", str(output)])
        assert result.exit_code == 0, result.output
        assert (output / "adapter_model.safetensors").exists()
        assert "Merged adapter saved to" in result.output

    def test_merge_ties(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        output = tmp_path / "merged_ties"

        runner = CliRunner()
        result = runner.invoke(cli, ["merge"] + [str(d) for d in dirs] + [
            "-o", str(output), "--method", "ties", "--density", "0.7"
        ])
        assert result.exit_code == 0, result.output

    def test_merge_dare(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        output = tmp_path / "merged_dare"

        runner = CliRunner()
        result = runner.invoke(cli, ["merge"] + [str(d) for d in dirs] + [
            "-o", str(output), "--method", "dare", "--drop-rate", "0.3", "--seed", "42"
        ])
        assert result.exit_code == 0, result.output


class TestValidate:
    def test_validate_healthy_subspace(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        sub_path = tmp_path / "subspace"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(sub_path), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        result = runner.invoke(cli, ["validate", str(sub_path)])
        assert result.exit_code == 0, result.output
        assert "All checks passed" in result.output


class TestDiff:
    def test_diff_two_tasks(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        sub_path = tmp_path / "subspace"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(sub_path), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        result = runner.invoke(cli, ["diff", str(sub_path), "adapter_0", "adapter_1"])
        assert result.exit_code == 0, result.output
        assert "L2 Dist" in result.output
        assert "Cosine" in result.output

    def test_diff_unknown_task_fails(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        sub_path = tmp_path / "subspace"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(sub_path), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        result = runner.invoke(cli, ["diff", str(sub_path), "adapter_0", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown task" in result.output


class TestBenchmark:
    def test_benchmark_runs(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)
        sub_path = tmp_path / "subspace"

        runner = CliRunner()
        args = [str(d) for d in dirs] + ["-o", str(sub_path), "-k", "2"]
        runner.invoke(cli, ["compress"] + args)

        result = runner.invoke(cli, ["benchmark", str(sub_path)])
        assert result.exit_code == 0, result.output
        assert "reconstruct" in result.output
        assert "project" in result.output
        assert "Compression ratio" in result.output


class TestAnalyze:
    def test_analyze_shows_similarity(self, tmp_path):
        dirs = _make_correlated_adapter_dirs(tmp_path, n=3)

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze"] + [str(d) for d in dirs])
        assert result.exit_code == 0, result.output
        assert "Similarity" in result.output
        assert "Cluster" in result.output
