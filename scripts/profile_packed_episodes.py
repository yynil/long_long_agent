#!/usr/bin/env python3
"""Profile real normalized episodes through the RWKV tokenizer and packed collator."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_adapter
from src.training import EpisodeEncodingConfig, RWKVByteTokenizer
from src.training.episode_collator import (
    decision_message_indices,
    tokenize_decision,
    tokenize_episode,
)
from src.training.packing import CausalSequence, pack_sequences
from src.training.token_budget_sampler import TokenBudgetPackSampler


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected mapping in {path}")
    return value


def selected_files(root: Path, patterns: list[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.parquet")
        if any(
            fnmatch.fnmatchcase(path.relative_to(root).as_posix(), pattern) for pattern in patterns
        )
    )


def sampled_episodes(source: dict[str, Any], storage: dict[str, Any], limit: int):
    data_root = Path(storage["local_root"])
    raw_root = data_root / storage["raw_dir"] / source["source_id"] / source["revision"]
    adapter = get_adapter(source["adapter"])
    count = 0
    for path in selected_files(raw_root, source["allow_patterns"]):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=min(limit, 16)):
            for row in batch.to_pylist():
                row["__source_file__"] = path.relative_to(raw_root).as_posix()
                yield adapter(row, source["revision"], source["declared_license"])
                count += 1
                if count >= limit:
                    return


def percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * probability)))
    return ordered[index]


def profile_source(
    source: dict[str, Any],
    storage: dict[str, Any],
    tokenizer: RWKVByteTokenizer,
    config: EpisodeEncodingConfig,
    limit: int,
    max_pack_tokens: int,
    alignment: int,
    decisions_per_episode: int,
) -> tuple[dict[str, Any], list[CausalSequence]]:
    episode_lengths: list[int] = []
    episode_supervised = 0
    episode_action = 0
    episode_reasoning = 0
    episode_mixed = 0
    oversized_episodes = 0
    decision_lengths: list[int] = []
    decision_supervised = 0
    decision_action = 0
    decision_reasoning = 0
    decision_mixed = 0
    decision_windows: list[CausalSequence] = []
    decision_failures = 0
    truncated_decisions = 0
    dropped_messages = 0
    for episode in sampled_episodes(source, storage, limit):
        encoded = tokenize_episode(episode, tokenizer, config)
        sequence = encoded.to_causal_sequence()
        length = len(sequence.input_ids)
        episode_lengths.append(length)
        counts = encoded.loss_token_counts()
        episode_supervised += sum(counts.values())
        episode_action += counts.get("assistant_action", 0)
        episode_reasoning += counts.get("assistant_reasoning", 0)
        episode_mixed += encoded.mixed_boundary_token_count
        oversized_episodes += length > max_pack_tokens

        indices = decision_message_indices(episode)
        if len(indices) > decisions_per_episode:
            positions = {
                round(step * (len(indices) - 1) / (decisions_per_episode - 1))
                for step in range(decisions_per_episode)
            }
            indices = tuple(indices[position] for position in sorted(positions))
        for message_index in indices:
            try:
                decision = tokenize_decision(
                    episode,
                    message_index,
                    tokenizer,
                    max_tokens=max_pack_tokens,
                    config=config,
                )
            except ValueError:
                decision_failures += 1
                continue
            decision_sequence = decision.to_causal_sequence()
            decision_windows.append(decision_sequence)
            decision_lengths.append(len(decision_sequence.input_ids))
            decision_counts = decision.episode.loss_token_counts()
            decision_supervised += sum(decision_counts.values())
            decision_action += decision_counts.get("assistant_action", 0)
            decision_reasoning += decision_counts.get("assistant_reasoning", 0)
            decision_mixed += decision.episode.mixed_boundary_token_count
            if decision.dropped_message_count:
                truncated_decisions += 1
                dropped_messages += decision.dropped_message_count
    if not episode_lengths:
        raise RuntimeError(f"no episodes sampled for {source['source_id']}")
    decision_length_report = None
    if decision_lengths:
        decision_length_report = {
            "min": min(decision_lengths),
            "median": statistics.median(decision_lengths),
            "p95": percentile(decision_lengths, 0.95),
            "max": max(decision_lengths),
            "sum": sum(decision_lengths),
        }
    return (
        {
            "sampled_episodes": len(episode_lengths),
            "whole_episode": {
                "causal_tokens": {
                    "min": min(episode_lengths),
                    "median": statistics.median(episode_lengths),
                    "p95": percentile(episode_lengths, 0.95),
                    "max": max(episode_lengths),
                    "sum": sum(episode_lengths),
                },
                "supervised_target_tokens": episode_supervised,
                "assistant_action_tokens": episode_action,
                "assistant_reasoning_tokens": episode_reasoning,
                "mixed_boundary_tokens_masked": episode_mixed,
                "over_max_pack_tokens": oversized_episodes,
            },
            "decision_windows": {
                "sampled": len(decision_lengths),
                "untrainable_within_limit": decision_failures,
                "causal_tokens": decision_length_report,
                "supervised_target_tokens": decision_supervised,
                "assistant_action_tokens": decision_action,
                "assistant_reasoning_tokens": decision_reasoning,
                "mixed_boundary_tokens_masked": decision_mixed,
                "windows_with_history_dropped": truncated_decisions,
                "dropped_messages": dropped_messages,
            },
        },
        decision_windows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-source", type=int, default=8)
    parser.add_argument("--decisions-per-episode", type=int, default=3)
    parser.add_argument("--max-pack-tokens", type=int, default=16384)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--action-weight", type=float, default=2.0)
    parser.add_argument("--sampler-seed", type=int, default=20260904)
    parser.add_argument("--bucket-size", type=int, default=2048)
    parser.add_argument("--sources-config", type=Path, default=REPO_ROOT / "configs/sources.yaml")
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=REPO_ROOT / "external/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_source <= 0 or args.decisions_per_episode < 2:
        raise SystemExit("--per-source must be positive and --decisions-per-episode must be >= 2")
    if args.alignment <= 0 or args.max_pack_tokens % args.alignment:
        raise SystemExit("--max-pack-tokens must be a positive multiple of --alignment")
    sources_config = load_yaml(args.sources_config)
    storage_config = load_yaml(REPO_ROOT / sources_config["storage_config"])
    storage = storage_config["storage"]
    tokenizer = RWKVByteTokenizer(args.vocabulary)
    encoding = EpisodeEncodingConfig(action_weight=args.action_weight)

    reports = {}
    decision_sequences: list[CausalSequence] = []
    for source in sources_config["sources"]:
        report, source_sequences = profile_source(
            source,
            storage,
            tokenizer,
            encoding,
            args.per_source,
            args.max_pack_tokens,
            args.alignment,
            args.decisions_per_episode,
        )
        reports[source["source_id"]] = report
        decision_sequences.extend(source_sequences)
    sampler = TokenBudgetPackSampler(
        [len(sequence.input_ids) for sequence in decision_sequences],
        [sequence.sample_id for sequence in decision_sequences],
        max_tokens=args.max_pack_tokens,
        alignment=args.alignment,
        seed=args.sampler_seed,
        bucket_size=args.bucket_size,
    )
    rows = sampler.plan().rank_rows
    packs = [
        pack_sequences(
            (decision_sequences[index] for index in row.indices),
            align_to=args.alignment,
        ).utilization()
        for row in rows
    ]
    real_tokens = sum(int(pack["real_tokens"]) for pack in packs)
    aligned_tokens = sum(int(pack["aligned_tokens"]) for pack in packs)
    padded_tokens = sum(int(pack["padded_baseline_tokens"]) for pack in packs)
    distributed_preview = []
    for rank in range(3):
        rank_sampler = TokenBudgetPackSampler(
            [len(sequence.input_ids) for sequence in decision_sequences],
            [sequence.sample_id for sequence in decision_sequences],
            max_tokens=args.max_pack_tokens,
            alignment=args.alignment,
            seed=args.sampler_seed,
            bucket_size=args.bucket_size,
            world_size=3,
            rank=rank,
        )
        rank_plan = rank_sampler.plan()
        distributed_preview.append(
            {
                "rank": rank,
                "rows": len(rank_plan.rank_rows),
                "real_tokens": sum(row.real_tokens for row in rank_plan.rank_rows),
                "aligned_tokens": sum(row.aligned_tokens for row in rank_plan.rank_rows),
                "global_dropped_rows": len(rank_plan.dropped_rows),
                "global_dropped_samples": rank_plan.dropped_sample_count,
            }
        )
    result = {
        "schema_version": 1,
        "tokenizer": {
            "path": str(args.vocabulary.relative_to(REPO_ROOT)),
            "sha256": hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(),
            "defined_token_count": tokenizer.defined_token_count,
            "maximum_token_id": tokenizer.maximum_token_id,
            "model_vocabulary_size": 65536,
        },
        "settings": {
            "sample_order": "first_rows_by_sorted_selected_file",
            "per_source": args.per_source,
            "decisions_per_episode": args.decisions_per_episode,
            "max_pack_tokens": args.max_pack_tokens,
            "alignment": args.alignment,
            "action_weight": args.action_weight,
            "sampler_algorithm": "seeded_bucket_best_fit_decreasing",
            "sampler_seed": args.sampler_seed,
            "bucket_size": args.bucket_size,
        },
        "sources": reports,
        "packing": {
            "pack_rows": len(packs),
            "packed_decision_windows": len(decision_sequences),
            "real_tokens": real_tokens,
            "aligned_tokens": aligned_tokens,
            "alignment_tokens": aligned_tokens - real_tokens,
            "packed_storage_efficiency": real_tokens / aligned_tokens,
            "padded_baseline_tokens": padded_tokens,
            "padding_tokens_avoided": padded_tokens - aligned_tokens,
            "distributed_preview_world_size_3": distributed_preview,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
