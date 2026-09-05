"""Load only admitted decisions; prepare diverse tiny-overfit inputs without training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import jsonschema
import pyarrow.parquet as pq

from src.data.a0_release import verify_release
from src.data.blob_store import BlobStore
from src.data.canonical import stable_id
from src.data.governance import sha256_file
from src.data.types import NormalizedEpisode, NormalizedMessage
from src.data.util import canonical_json

from .episode_collator import tokenize_decision
from .token_budget_sampler import TokenBudgetPackSampler


def restore_episode(row: dict, blob: dict) -> NormalizedEpisode:
    if blob.get("blob_schema_version") != 1:
        raise ValueError("unknown canonical blob version")
    for key in ("source_record_id", "source_dataset", "source_revision"):
        if row[key] != blob[key]:
            raise ValueError("episode/blob source identity mismatch")
    if stable_id(row["source_dataset"], row["source_record_id"]) != row["episode_id"]:
        raise ValueError("canonical episode identity mismatch")
    messages = tuple(NormalizedMessage(**message) for message in blob["normalized"]["messages"])
    return NormalizedEpisode(
        source_dataset=row["source_dataset"],
        source_revision=row["source_revision"],
        source_license=row["source_license"],
        source_record_id=row["source_record_id"],
        task_id=row["task_id"],
        task_text=row["task_text"],
        messages=messages,
        harness=row["harness"],
        tools=json.loads(row["tools_schema"]),
        repo=row["repo"],
        base_commit=row["base_commit"],
        teacher_model=row["teacher_model"],
        success=row["success"],
        terminal_reward=row["terminal_reward"],
        failure_type=row["failure_type"],
        metadata=blob["normalized"]["metadata"],
        raw_record=blob["raw_record"],
    )


def prepare_overfit_inputs(release: Path, tokenizer, *, count: int, max_tokens: int, seed: int):
    if type(count) is not int or count <= 0 or count % 4:
        raise ValueError("count must be positive and divisible by the four teacher/outcome strata")
    if type(max_tokens) is not int or max_tokens <= 0 or max_tokens % 16:
        raise ValueError("max_tokens must be positive and 16-aligned")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    manifest = verify_release(release)
    admission = json.loads((release / "admission.json").read_text())
    train_ids = set((release / "splits/train.txt").read_text().splitlines())
    entries = {e["episode_id"]: e for e in admission["accepted"]}
    rows = {
        r["episode_id"]: r
        for r in pq.read_table(release / "episodes.parquet").to_pylist()
        if r["episode_id"] in train_ids
    }
    if set(rows) != train_ids:
        raise ValueError("missing train episode")
    groups = defaultdict(list)
    for row in rows.values():
        if row["success"] is None:
            raise ValueError("unknown outcome in admitted A0")
        groups[(row["teacher_model"], row["success"])].append(row["episode_id"])
    if len(groups) != 4 or {key[0] for key in groups} != {"MiniMax-M2.5", "Qwen3.5-122B"}:
        raise ValueError("A0 teacher/outcome strata changed")
    for ids in groups.values():
        ids.sort(key=lambda value: hashlib.sha256(f"{seed}/{value}".encode()).hexdigest())
    iterators = {key: iter(ids) for key, ids in groups.items()}
    store = BlobStore(Path(manifest["blob_root"]))
    selected, metadata, rejected = [], [], Counter()
    for _ in range(count // 4):
        for key in sorted(groups):
            while True:
                episode_id = next(iterators[key], None)
                if episode_id is None:
                    raise ValueError(
                        "insufficient eligible real decisions for balanced tiny overfit"
                    )
                row, entry = rows[episode_id], entries[episode_id]
                if entry["split"] != "train":
                    raise ValueError("split list disagrees with admission")
                indices = entry["message_indices"]
                choice = int(
                    hashlib.sha256(f"{seed}/decision/{episode_id}".encode()).hexdigest(), 16
                ) % len(indices)
                message_index = indices[choice]
                episode = restore_episode(row, store.read_json(row["raw_trace_ref"]))
                try:
                    window = tokenize_decision(
                        episode, message_index, tokenizer, max_tokens=max_tokens
                    )
                except ValueError as error:
                    if "protected context" not in str(error):
                        raise
                    rejected["protected_context_overflow"] += 1
                    continue
                target = episode.messages[message_index]
                target_digest = hashlib.sha256(
                    canonical_json(
                        {
                            "content": target.content,
                            "reasoning": target.reasoning,
                            "action": target.action,
                        }
                    ).encode()
                ).hexdigest()
                selected.append(window.episode)
                metadata.append(
                    {
                        "episode_id": episode_id,
                        "task_id_sha256": hashlib.sha256(episode.task_id.encode()).hexdigest(),
                        "message_index": message_index,
                        "teacher": key[0],
                        "success": key[1],
                        "sample_id": window.episode.sample_id,
                        "input_tokens": len(window.episode.token_ids) - 1,
                        "dropped_messages": window.dropped_message_count,
                        "supervised_regions": window.episode.loss_token_counts(),
                        "target_sha256": target_digest,
                        "tokens_sha256": hashlib.sha256(
                            canonical_json(window.episode.token_ids).encode()
                        ).hexdigest(),
                        "weights_sha256": hashlib.sha256(
                            canonical_json(window.episode.token_loss_weights).encode()
                        ).hexdigest(),
                    }
                )
                break
    return selected, metadata, dict(rejected)


def write_overfit_input_plan(
    release: Path, tokenizer, output: Path, *, max_tokens=8192, seed=20260905
):
    if output.exists():
        raise FileExistsError("refusing to overwrite tiny-overfit input plan")
    samples, rows, rejected = prepare_overfit_inputs(
        release, tokenizer, count=128, max_tokens=max_tokens, seed=seed
    )
    profiles = {}
    for count in (32, 128):
        subset = samples[:count]
        lengths = [len(sample.token_ids) - 1 for sample in subset]
        sampler = TokenBudgetPackSampler(
            lengths, [s.sample_id for s in subset], max_tokens=max_tokens, seed=seed
        )
        plan = sampler.plan()
        real = sum(r.real_tokens for r in plan.all_rows)
        aligned = sum(r.aligned_tokens for r in plan.all_rows)
        profiles[str(count)] = {
            "unique_episodes": len({r["episode_id"] for r in rows[:count]}),
            "unique_targets": len({r["target_sha256"] for r in rows[:count]}),
            "effective_input_tokens": real,
            "pack_rows": len(plan.all_rows),
            "tail_alignment_tokens": aligned - real,
            "loss_tokens": sum(sum(s.loss_token_counts().values()) for s in subset),
            "teacher_outcome_strata": dict(
                Counter(f"{r['teacher']}/{r['success']}" for r in rows[:count])
            ),
        }
    result = {
        "schema_version": 1,
        "purpose": "input_preparation_only",
        "training_executed": False,
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "tokenizer_sha256": sha256_file(tokenizer.vocabulary_path),
        "seed": seed,
        "max_tokens": max_tokens,
        "selection": "train_only_one_hashed_admitted_decision_per_unique_episode_equal_teacher_outcome_v1",
        "rejected_before_128": rejected,
        "profiles": profiles,
        "samples": rows,
    }
    schema = Path(__file__).resolve().parents[2] / "schemas/overfit_input_plan.schema.json"
    jsonschema.validate(result, json.loads(schema.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
