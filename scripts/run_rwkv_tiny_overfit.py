#!/usr/bin/env python3
"""Run a head-only 0.4B RWKV packed-trainer diagnostic on 32/128 synthetic decisions."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.types import NormalizedEpisode, NormalizedMessage
from src.training.episode_collator import EpisodeEncodingConfig, PackedEpisodeCollator
from src.training.parameter_groups import (
    build_fp32_master_adamw,
    build_rwkv7_optimizer_plan,
    build_torch_adamw,
)
from src.training.sft_trainer import PackedAgentSFTTrainer, move_packed_batch
from src.training.token_budget_sampler import TokenBudgetPackSampler
from src.training.tokenizer import RWKVByteTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config)
    packing = config["packing"]
    if packing["max_tokens"] % packing["alignment"]:
        raise ValueError("packing.max_tokens must be divisible by alignment")
    return config


def verify_inputs(config: dict[str, Any], worktree: Path, checkpoint: Path) -> Path:
    expected = config["base_model"]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected["upstream_revision"]:
        raise ValueError(f"RWKV worktree revision mismatch: {revision}")
    if checkpoint.name != expected["checkpoint_file"]:
        raise ValueError("checkpoint filename does not match the fixed diagnostic config")
    if sha256_file(checkpoint) != expected["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")
    vocabulary = worktree / "RWKV-v7/rwkv_vocab_v20230424.txt"
    if sha256_file(vocabulary) != expected["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")
    return vocabulary


def build_fixture(count: int, revision: str) -> list[NormalizedEpisode]:
    return [
        NormalizedEpisode(
            source_dataset="fixture/rwkv-tiny-agent-overfit",
            source_revision=revision,
            source_license="synthetic",
            source_record_id=f"decision-{index:04d}",
            task_id=f"task-{index % 8}",
            task_text="Return the acknowledged status.",
            messages=(
                NormalizedMessage(role="user", content=f"request {index % 8}"),
                NormalizedMessage(role="assistant", content="acknowledged"),
            ),
            success=True,
        )
        for index in range(count)
    ]


def import_patched_model(worktree: Path):
    train_root = worktree / "RWKV-v7/train_temp"
    os.chdir(train_root)
    sys.path.insert(0, str(train_root / "src"))
    return importlib.import_module("model")


def make_network(model_module, checkpoint: Path, config: dict[str, Any]):
    model = config["base_model"]
    args = SimpleNamespace(
        n_layer=model["layers"],
        n_embd=model["embedding_dim"],
        vocab_size=65536,
        ctx_len=model["context_length"],
        head_size=model["head_size"],
        dim_att=model["embedding_dim"],
        dim_ffn=model["embedding_dim"] * 7 // 2,
        grad_cp=0,
        my_testing="x070",
        decay_lora_rank=model["decay_lora_rank"],
        aaa_lora_rank=model["aaa_lora_rank"],
        mv_lora_rank=model["mv_lora_rank"],
        gate_lora_rank=model["gate_lora_rank"],
    )
    network = model_module.RWKV(args)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(state_dict, strict=True)
    del state_dict
    network = network.to(device="cuda", dtype=torch.bfloat16).eval()
    head_only = config["optimization"]["mode"] == "head_only_diagnostic"
    for parameter in network.parameters():
        parameter.requires_grad_(not head_only)
    if head_only:
        network.head.weight.requires_grad_(True)
    return network


def make_optimizer(network, config: dict[str, Any]):
    optimization = config["optimization"]
    if optimization["mode"] == "head_only_diagnostic":
        optimizer = torch.optim.AdamW(
            [network.head.weight],
            lr=optimization["learning_rate"],
            betas=(optimization["beta1"], optimization["beta2"]),
            eps=optimization["epsilon"],
            weight_decay=optimization["weight_decay"],
            fused=optimization["fused_adamw"],
        )
        return optimizer, {
            "covered_parameter_tensors": 1,
            "trainable_parameter_tensors": 1,
            "trainable_numel": network.head.weight.numel(),
            "optimizer_groups": ["head_only_diagnostic"],
            "optimizer_state": optimization["optimizer_state"],
        }
    plan = build_rwkv7_optimizer_plan(
        network.named_parameters(),
        n_layer=config["base_model"]["layers"],
        weight_decay=optimization["weight_decay"],
        base_prefix="",
        latent_prefix=None,
        value_prefix=None,
    )
    builder = (
        build_fp32_master_adamw
        if optimization["optimizer_state"] == "fp32_master"
        else build_torch_adamw
    )
    optimizer = builder(
        plan,
        learning_rate=optimization["learning_rate"],
        beta1=optimization["beta1"],
        beta2=optimization["beta2"],
        epsilon=optimization["epsilon"],
        fused=optimization["fused_adamw"],
    )
    return optimizer, {
        "covered_parameter_tensors": plan.covered_parameter_count,
        "trainable_parameter_tensors": sum(len(group.parameters) for group in plan.groups),
        "trainable_numel": plan.trainable_numel,
        "optimizer_groups": [group.name for group in plan.groups],
        "optimizer_state": optimization["optimizer_state"],
    }


def row_batches(collator, decisions, sampler):
    for row in sampler:
        cpu_batch = collator.collate_tokenized([decisions[index] for index in row])
        yield move_packed_batch(cpu_batch, "cuda")


def evaluate(trainer, collator, decisions, sampler) -> float:
    trainer.network.eval()
    numerator = 0.0
    denominator = 0.0
    with torch.no_grad():
        for batch in row_batches(collator, decisions, sampler):
            result = trainer.compute_loss(batch)
            numerator += float(result.total) * result.weight_sum
            denominator += result.weight_sum
    if denominator <= 0:
        raise ValueError("evaluation has no effective loss weight")
    return numerator / denominator


def run_size(model_module, config, checkpoint, tokenizer, count) -> dict[str, Any]:
    packing = config["packing"]
    optimization = config["optimization"]
    torch.manual_seed(optimization["seed"] + count)
    network = make_network(model_module, checkpoint, config)
    collator = PackedEpisodeCollator(
        tokenizer,
        encoding=EpisodeEncodingConfig(include_tool_schema=False),
        align_to=packing["alignment"],
        max_pack_tokens=packing["max_tokens"],
    )
    episodes = build_fixture(count, config["fixture"]["source_revision"])
    decisions = [collator.encode(episode) for episode in episodes]
    lengths = [len(decision.to_causal_sequence().input_ids) for decision in decisions]
    sampler = TokenBudgetPackSampler(
        lengths,
        [decision.sample_id for decision in decisions],
        max_tokens=packing["max_tokens"],
        alignment=packing["alignment"],
        seed=packing["sampler_seed"],
        bucket_size=packing["bucket_size"],
    )
    optimizer, optimizer_info = make_optimizer(network, config)
    trainer = PackedAgentSFTTrainer(
        network,
        optimizer,
        head_chunk_tokens=optimization["head_chunk_tokens"],
        gradient_clip_norm=optimization["gradient_clip_norm"],
    )
    initial_plan = sampler.plan()
    initial_loss = evaluate(trainer, collator, decisions, sampler)
    epoch_losses = []
    maximum_gradient_norm = 0.0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(optimization["epochs"]):
        sampler.set_epoch(epoch)
        for batch in row_batches(collator, decisions, sampler):
            metrics = trainer.train_step(batch)
            maximum_gradient_norm = max(maximum_gradient_norm, metrics.gradient_norm)
        epoch_losses.append(evaluate(trainer, collator, decisions, sampler))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
    final_loss = epoch_losses[-1]
    ratio = final_loss / initial_loss
    prior_losses = [initial_loss, *epoch_losses[:-1]]
    maximum_epoch_increase_ratio = max(
        current / max(previous, 1e-12)
        for previous, current in zip(prior_losses, epoch_losses, strict=True)
    )
    expected_sequences = count * optimization["epochs"]
    if hasattr(optimizer, "optimizer_state_dtypes"):
        state_dtypes = optimizer.optimizer_state_dtypes
    else:
        state_dtypes = {
            value.dtype
            for state in optimizer.state.values()
            for key, value in state.items()
            if key != "step" and hasattr(value, "dtype")
        }
    optimizer_info["observed_optimizer_state_dtypes"] = sorted(str(item) for item in state_dtypes)
    result = {
        "sample_count": count,
        "pack_rows": len(initial_plan.all_rows),
        "real_tokens": sum(row.real_tokens for row in initial_plan.all_rows),
        "aligned_tokens": sum(row.aligned_tokens for row in initial_plan.all_rows),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_to_initial_loss_ratio": ratio,
        "maximum_epoch_to_previous_loss_ratio": maximum_epoch_increase_ratio,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
        "epoch_losses": epoch_losses,
        "optimizer_steps": trainer.progress.optimizer_steps,
        "trained_sequence_presentations": trainer.progress.sequences,
        "expected_sequence_presentations": expected_sequences,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": peak_gib,
        **optimizer_info,
        "passed": (
            ratio <= config["acceptance"]["maximum_final_to_initial_loss_ratio"]
            and maximum_epoch_increase_ratio
            <= config["acceptance"]["maximum_epoch_to_previous_loss_ratio"]
            and trainer.progress.sequences == expected_sequences
        ),
    }
    del trainer, optimizer, network
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/rwkv_tiny_overfit.yaml")
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "schemas/rwkv_tiny_overfit.schema.json",
    )
    parser.add_argument("--rwkv-worktree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real RWKV diagnostic")
    config = load_config(args.config.resolve(), args.schema.resolve())
    worktree = args.rwkv_worktree.resolve()
    checkpoint = args.checkpoint.resolve()
    vocabulary = verify_inputs(config, worktree, checkpoint)
    os.environ.setdefault("RWKV_JIT_ON", "0")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    model_module = import_patched_model(worktree)
    tokenizer = RWKVByteTokenizer(vocabulary)
    results = [
        run_size(model_module, config, checkpoint, tokenizer, count)
        for count in config["fixture"]["sample_counts"]
    ]
    mode = config["optimization"]["mode"]
    payload = {
        "schema_version": 1,
        "mode": config["optimization"]["mode"],
        "checkpoint_sha256": config["base_model"]["checkpoint_sha256"],
        "upstream_revision": config["base_model"]["upstream_revision"],
        "device": torch.cuda.get_device_name(0),
        "results": results,
        "status": "passed" if all(result["passed"] for result in results) else "failed",
        "limitation": (
            "head-only synthetic diagnostic; not full-parameter Agent SFT"
            if mode == "head_only_diagnostic"
            else "full-parameter synthetic diagnostic; not real Agent-data SFT"
        ),
    }
    artifact_root = Path(config["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path = artifact_root / f"rwkv_0_4b_{args.config.stem}_v1.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"artifact={output_path}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
