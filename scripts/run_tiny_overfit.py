#!/usr/bin/env python3
"""Run deterministic 32/128-sample packed SFT pipeline overfit smokes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.types import NormalizedEpisode, NormalizedMessage
from src.training.episode_collator import EpisodeEncodingConfig, PackedEpisodeCollator
from src.training.sft_trainer import PackedAgentSFTTrainer
from src.training.token_budget_sampler import TokenBudgetPackSampler


class ByteLevelFixtureTokenizer:
    def encode_bytes(self, source: bytes) -> list[int]:
        return [value + 1 for value in source]

    def token_bytes(self, token_id: int) -> bytes:
        if not 1 <= token_id <= 256:
            raise ValueError("fixture byte token ID must be in [1, 256]")
        return bytes([token_id - 1])


class TinyFeatureModel(torch.nn.Module):
    """Small backend for testing trainer mechanics, not RWKV architecture quality."""

    def __init__(self, vocabulary_size: int, embedding_width: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_width)
        self.projection = torch.nn.Linear(embedding_width, embedding_width)
        self.head = torch.nn.Linear(embedding_width, vocabulary_size, bias=False)

    def _forward_features(self, input_ids, sequence_start_mask):
        if sequence_start_mask.dtype != torch.uint8 or sequence_start_mask.shape != input_ids.shape:
            raise ValueError("tiny backend requires the RWKV packed start-mask contract")
        if not torch.all(sequence_start_mask[:, 0] != 0):
            raise ValueError("every packed row must start with a reset")
        return torch.tanh(self.projection(self.embedding(input_ids)))


def load_config(config_path: Path, schema_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config)
    if config["packing"]["max_tokens"] % config["packing"]["alignment"]:
        raise ValueError("packing.max_tokens must be divisible by alignment")
    return config


def build_fixture(count: int, revision: str) -> list[NormalizedEpisode]:
    return [
        NormalizedEpisode(
            source_dataset="fixture/tiny-agent-overfit",
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


def row_batches(collator, decisions, sampler):
    for row in sampler:
        yield collator.collate_tokenized([decisions[index] for index in row])


def evaluate(trainer, batches) -> float:
    trainer.network.eval()
    numerator = 0.0
    denominator = 0.0
    with torch.no_grad():
        for batch in batches:
            result = trainer.compute_loss(batch)
            numerator += float(result.total) * result.weight_sum
            denominator += result.weight_sum
    if denominator <= 0:
        raise ValueError("evaluation has no effective loss weight")
    return numerator / denominator


def run_size(config: dict[str, Any], count: int) -> dict[str, Any]:
    packing = config["packing"]
    optimization = config["optimization"]
    torch.manual_seed(optimization["seed"] + count)
    collator = PackedEpisodeCollator(
        ByteLevelFixtureTokenizer(),
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
    model = TinyFeatureModel(
        config["model"]["vocabulary_size"],
        config["model"]["embedding_width"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization["learning_rate"],
        betas=(optimization["beta1"], optimization["beta2"]),
        eps=optimization["epsilon"],
        weight_decay=optimization["weight_decay"],
    )
    trainer = PackedAgentSFTTrainer(
        model,
        optimizer,
        head_chunk_tokens=optimization["head_chunk_tokens"],
        gradient_clip_norm=optimization["gradient_clip_norm"],
    )
    initial_plan = sampler.plan()
    initial_loss = evaluate(trainer, row_batches(collator, decisions, sampler))
    epoch_losses = []
    for epoch in range(optimization["epochs"]):
        sampler.set_epoch(epoch)
        for batch in row_batches(collator, decisions, sampler):
            trainer.train_step(batch)
        epoch_losses.append(evaluate(trainer, row_batches(collator, decisions, sampler)))
    final_loss = epoch_losses[-1]
    ratio = final_loss / initial_loss
    expected_sequences = count * optimization["epochs"]
    passed = (
        ratio <= config["acceptance"]["maximum_final_to_initial_loss_ratio"]
        and trainer.progress.sequences == expected_sequences
    )
    return {
        "sample_count": count,
        "pack_rows": len(initial_plan.all_rows),
        "real_tokens": sum(row.real_tokens for row in initial_plan.all_rows),
        "aligned_tokens": sum(row.aligned_tokens for row in initial_plan.all_rows),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_to_initial_loss_ratio": ratio,
        "epoch_losses": epoch_losses,
        "optimizer_steps": trainer.progress.optimizer_steps,
        "trained_sequence_presentations": trainer.progress.sequences,
        "expected_sequence_presentations": expected_sequences,
        "effective_loss_tokens": trainer.progress.effective_loss_tokens,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/tiny_overfit.yaml")
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "schemas/tiny_overfit.schema.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve(), args.schema.resolve())
    results = [run_size(config, count) for count in config["fixture"]["sample_counts"]]
    payload = {
        "schema_version": 1,
        "backend": config["model"]["backend"],
        "seed": config["optimization"]["seed"],
        "results": results,
        "status": "passed" if all(result["passed"] for result in results) else "failed",
        "limitation": "synthetic trainer-mechanics smoke; not a real RWKV checkpoint quality result",
    }
    artifact_root = Path(config["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_path = artifact_root / f"tiny_overfit_v1_seed{payload['seed']}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"artifact={output_path}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
