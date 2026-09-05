"""Immutable, split-isolated SFT tokens from every admitted successful A0 decision."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data.a0_release import verify_release
from src.data.blob_store import BlobStore
from src.data.governance import sha256_file
from src.data.util import canonical_json
from src.model.runtime import ROOT

from .a0_dataset import restore_episode
from .episode_collator import TokenizedEpisode, tokenize_decision
from .preflight import write_json_once
from .tokenizer import RWKVByteTokenizer

SPLITS = ("train", "dev", "test")
TOKEN_SCHEMA = pa.schema(
    [
        ("episode_id", pa.string()),
        ("decision_id", pa.string()),
        ("sample_id", pa.string()),
        ("split", pa.string()),
        ("group_sha256", pa.string()),
        ("teacher", pa.string()),
        ("source_success", pa.bool_()),
        ("message_index", pa.int32()),
        ("token_ids", pa.list_(pa.int32())),
        ("token_loss_weights", pa.list_(pa.float32())),
        ("token_regions", pa.list_(pa.string())),
        ("mixed_boundary_count", pa.int32()),
        ("dropped_messages", pa.int32()),
        ("record_sha256", pa.string()),
    ],
    metadata={b"schema_version": b"1", b"purpose": b"a0_success_sft_inputs"},
)
REGION_WEIGHTS = {
    "tool_schema": {0.0},
    "assistant_prefix": {0.0},
    "system": {0.0},
    "user": {0.0},
    "tool_response": {0.0},
    "mixed_boundary": {0.0},
    "document_start": {0.0},
    "assistant_reasoning": {0.0, 1.0},
    "assistant_final": {0.0, 1.0},
    "assistant_action": {0.0, 2.0},
    "assistant_separator": {0.0, 1.0, 2.0},
    "assistant_turn_end": {0.0, 1.0},
    "document_end": {0.0, 1.0},
}


def record_digest(row: dict) -> str:
    return hashlib.sha256(
        canonical_json({k: v for k, v in row.items() if k != "record_sha256"}).encode()
    ).hexdigest()


def validate_record(row: dict, *, split: str, max_tokens: int) -> None:
    if set(row) != set(TOKEN_SCHEMA.names) or split not in SPLITS or row["split"] != split:
        raise ValueError("unknown token fields or split mismatch")
    if row["source_success"] is not True:
        raise ValueError("unverified failure is not a positive SFT target")
    if any(
        not isinstance(row[k], str) or not re.fullmatch("[a-f0-9]{64}", row[k])
        for k in ("episode_id", "decision_id", "sample_id", "group_sha256", "record_sha256")
    ):
        raise ValueError("invalid token record identity")
    if not isinstance(row["teacher"], str) or not row["teacher"]:
        raise ValueError("invalid teacher provenance")
    tokens, weights, regions = row["token_ids"], row["token_loss_weights"], row["token_regions"]
    if (
        not 2 <= len(tokens) <= max_tokens + 1
        or len(tokens) != len(weights)
        or len(tokens) != len(regions)
    ):
        raise ValueError("invalid token lengths")
    if (
        tokens[0] != 0
        or tokens[-1] != 0
        or weights[0] != 0
        or any(type(t) is not int or not 0 <= t < 65536 for t in tokens)
    ):
        raise ValueError("invalid token IDs or causal start")
    if (
        any(
            type(w) not in (float, int)
            or not math.isfinite(w)
            or r not in REGION_WEIGHTS
            or w not in REGION_WEIGHTS[r]
            for w, r in zip(weights, regions, strict=True)
        )
        or sum(weights[1:]) <= 0
    ):
        raise ValueError("invalid assistant-only mask or no loss")
    if row["mixed_boundary_count"] != regions.count("mixed_boundary") or any(
        type(row[k]) is not int or row[k] < 0
        for k in ("message_index", "dropped_messages", "mixed_boundary_count")
    ):
        raise ValueError("invalid token provenance counters")
    if record_digest(row) != row["record_sha256"]:
        raise ValueError("token record content hash mismatch")


def to_tokenized(row: dict, *, split: str, max_tokens: int) -> TokenizedEpisode:
    validate_record(row, split=split, max_tokens=max_tokens)
    return TokenizedEpisode(
        row["sample_id"],
        "",
        tuple(row["token_ids"]),
        tuple(row["token_loss_weights"]),
        tuple(row["token_regions"]),
        row["mixed_boundary_count"],
    )


def encode_record(
    episode_row: dict,
    episode,
    decision: dict,
    message_index: int,
    tokenizer,
    *,
    split: str,
    max_tokens: int,
) -> dict:
    if episode_row["success"] is not True:
        raise ValueError("failure source cannot enter positive SFT data")
    window = tokenize_decision(episode, message_index, tokenizer, max_tokens=max_tokens)
    value = window.episode
    row = {
        "episode_id": episode_row["episode_id"],
        "decision_id": decision["decision_id"],
        "sample_id": value.sample_id,
        "split": split,
        "group_sha256": hashlib.sha256(episode_row["split_group"].encode()).hexdigest(),
        "teacher": episode_row["teacher_model"],
        "source_success": True,
        "message_index": message_index,
        "token_ids": list(value.token_ids),
        "token_loss_weights": list(value.token_loss_weights),
        "token_regions": list(value.token_regions),
        "mixed_boundary_count": value.mixed_boundary_token_count,
        "dropped_messages": window.dropped_message_count,
    }
    row["record_sha256"] = record_digest(row)
    validate_record(row, split=split, max_tokens=max_tokens)
    return row


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    jsonschema.validate(
        config, json.loads((ROOT / "schemas/a0_sft_inputs.schema.json").read_text())
    )
    return config


def distribution(values: list[int]) -> dict:
    return {
        "count": len(values),
        "sum": sum(values),
        "p50": float(np.quantile(values, 0.5)) if values else 0,
        "p95": float(np.quantile(values, 0.95)) if values else 0,
        "max": max(values, default=0),
    }


def source_index(release: Path) -> dict[str, dict]:
    """Canonical identities/outcomes, independent of the derived token files."""
    episodes = {r["episode_id"]: r for r in pq.read_table(release / "episodes.parquet").to_pylist()}
    admission = {
        r["episode_id"]: r for r in json.loads((release / "admission.json").read_text())["accepted"]
    }
    owners = {}
    for split in SPLITS:
        for eid in (release / f"splits/{split}.txt").read_text().splitlines():
            if eid in owners:
                raise ValueError("duplicate canonical split membership")
            owners[eid] = split
    if set(owners) != set(episodes) or set(admission) != set(episodes):
        raise ValueError("canonical episode admission coverage mismatch")
    result = {}
    for d in pq.read_table(release / "decisions.parquet").to_pylist():
        eid = d["episode_id"]
        e, a = episodes[eid], admission[eid]
        index = int(d["prefix_messages_ref"].split("#normalized.messages=0:")[1])
        if index not in a["message_indices"] or a["split"] != owners[eid]:
            raise ValueError("canonical admission decision or split mismatch")
        if d["decision_id"] in result:
            raise ValueError("duplicate canonical decision identity")
        result[d["decision_id"]] = {
            "episode_id": eid,
            "split": owners[eid],
            "source_success": e["success"] is True,
            "teacher": e["teacher_model"],
            "message_index": index,
            "group_sha256": hashlib.sha256(e["split_group"].encode()).hexdigest(),
        }
    return result


def verify_source_row(row: dict, sources: dict[str, dict], *, excluded: bool = False) -> None:
    source = sources.get(row["decision_id"])
    if source is None:
        raise ValueError("unknown canonical decision")
    keys = ("episode_id", "split") if excluded else tuple(source)
    if any(row[k] != source[k] for k in keys):
        raise ValueError("SFT record does not match canonical source")
    if excluded:
        expected = (
            "protected_context_overflow_8k"
            if source["source_success"]
            else "source_failure_requires_verified_recovery"
        )
        if row["reason"] != expected:
            raise ValueError("exclusion reason does not match source outcome")


def build_inputs(config_path: Path) -> dict:
    config = load_config(config_path)
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    release, output = (
        data_root / "releases" / config["release_id"],
        data_root / config["output_directory"],
    )
    if not output.resolve().is_relative_to(data_root / "artifacts/sft_inputs"):
        raise ValueError("SFT inputs outside artifact storage")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit data protocol before building SFT inputs")
    if sha256_file(release / "manifest.json") != config["release_manifest_sha256"]:
        raise ValueError("source release changed")
    release_manifest = verify_release(release)
    if (
        release_manifest["episode_count"] != config["expected_episodes"]
        or release_manifest["decision_count"] != config["expected_decisions"]
    ):
        raise ValueError("source release coverage changed")
    if sha256_file(ROOT / "configs/training_data.yaml") != config["training_config_sha256"]:
        raise ValueError("SFT training serialization or mask config changed")
    training = yaml.safe_load((ROOT / "configs/training_data.yaml").read_text())
    vocabulary = ROOT / training["tokenizer"]["vocabulary"]
    if sha256_file(vocabulary) != config["tokenizer_sha256"]:
        raise ValueError("SFT tokenizer changed")
    tokenizer, blobs = RWKVByteTokenizer(vocabulary), BlobStore(Path(release_manifest["blob_root"]))
    entries = {
        r["episode_id"]: r for r in json.loads((release / "admission.json").read_text())["accepted"]
    }
    episodes = sorted(
        pq.read_table(release / "episodes.parquet").to_pylist(), key=lambda r: r["episode_id"]
    )
    decisions = {}
    for d in pq.read_table(release / "decisions.parquet").to_pylist():
        marker = "#normalized.messages=0:"
        index = int(d["prefix_messages_ref"].split(marker)[1])
        key = (d["episode_id"], index)
        if key in decisions:
            raise ValueError("duplicate admitted decision locator")
        decisions[key] = d
    split_ids = {
        split: set((release / f"splits/{split}.txt").read_text().splitlines()) for split in SPLITS
    }
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "status": "failed",
        "scanned_episodes": 0,
        "scanned_decisions": 0,
        "accepted": {s: 0 for s in SPLITS},
        "rejected": {s: {} for s in SPLITS},
        "files": {},
        "config_sha256": sha256_file(config_path),
        "source_release_sha256": config["release_manifest_sha256"],
        "tokenizer_sha256": config["tokenizer_sha256"],
        "training_config_sha256": sha256_file(ROOT / "configs/training_data.yaml"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "max_tokens": config["max_tokens"],
    }
    write_json_once(
        output / "intent.json",
        {k: v for k, v in report.items() if k not in ("accepted", "rejected", "files")},
    )
    writers = {}
    seen_ids, input_lengths, loss_lengths = set(), {s: [] for s in SPLITS}, {s: [] for s in SPLITS}
    rejected = {s: Counter() for s in SPLITS}
    try:
        writers = {
            s: pq.ParquetWriter(
                output / f"{s}.parquet",
                TOKEN_SCHEMA,
                compression="zstd",
                use_compliant_nested_type=False,
            )
            for s in SPLITS
        }
        with (output / "excluded.jsonl").open("x") as exclusions:
            for episode_row in episodes:
                eid = episode_row["episode_id"]
                entry = entries[eid]
                split = entry["split"]
                if split not in SPLITS or eid not in split_ids[split]:
                    raise ValueError("admission split mismatch")
                selected = entry["message_indices"]
                if set(selected) != {i for e, i in decisions if e == eid}:
                    raise ValueError("admitted decision coverage changed")
                report["scanned_episodes"] += 1
                episode = (
                    restore_episode(episode_row, blobs.read_json(episode_row["raw_trace_ref"]))
                    if episode_row["success"] is True
                    else None
                )
                encoded = []
                for index in selected:
                    d = decisions[(eid, index)]
                    if d["decision_id"] in seen_ids:
                        raise ValueError("duplicate canonical decision")
                    seen_ids.add(d["decision_id"])
                    report["scanned_decisions"] += 1
                    reason = (
                        "source_failure_requires_verified_recovery" if episode is None else None
                    )
                    if reason is None:
                        try:
                            row = encode_record(
                                episode_row,
                                episode,
                                d,
                                index,
                                tokenizer,
                                split=split,
                                max_tokens=config["max_tokens"],
                            )
                        except ValueError as error:
                            if "protected context" not in str(error):
                                raise
                            reason = "protected_context_overflow_8k"
                    if reason:
                        rejected[split][reason] += 1
                        exclusions.write(
                            json.dumps(
                                {
                                    "episode_id": eid,
                                    "decision_id": d["decision_id"],
                                    "split": split,
                                    "reason": reason,
                                }
                            )
                            + "\n"
                        )
                    else:
                        encoded.append(row)
                        report["accepted"][split] += 1
                        input_lengths[split].append(len(row["token_ids"]) - 1)
                        loss_lengths[split].append(
                            sum(w > 0 for w in row["token_loss_weights"][1:])
                        )
                if encoded:
                    writers[split].write_table(pa.Table.from_pylist(encoded, schema=TOKEN_SCHEMA))
                if report["scanned_episodes"] % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "scanned_episodes": report["scanned_episodes"],
                                "scanned_decisions": report["scanned_decisions"],
                                "accepted": report["accepted"],
                            }
                        ),
                        flush=True,
                    )
        for writer in writers.values():
            writer.close()
        writers.clear()
        if (
            len(seen_ids) != config["expected_decisions"]
            or sum(report["accepted"].values()) + sum(sum(v.values()) for v in rejected.values())
            != config["expected_decisions"]
        ):
            raise ValueError("SFT accounting does not cover full A0")
        report["rejected"] = {s: dict(v) for s, v in rejected.items()}
        report["input_tokens"] = {s: distribution(v) for s, v in input_lengths.items()}
        report["loss_tokens"] = {s: distribution(v) for s, v in loss_lengths.items()}
        report["files"] = {
            name: sha256_file(output / name)
            for name in ("train.parquet", "dev.parquet", "test.parquet", "excluded.jsonl")
        }
        report["status"] = "built"
        validate_manifest(report)
        write_json_once(output / "manifest.json", report)
        # The independent verify command is required for admission to a trainer.
    except Exception as error:  # noqa: BLE001 -- type-only failures; no source contents
        report["status"] = "failed"
        report["exception_type"] = type(error).__name__
        report["rejected"] = {s: dict(v) for s, v in rejected.items()}
    finally:
        for writer in writers.values():
            writer.close()
        write_json_once(output / "result.json", report)
    return report


def validate_manifest(manifest: dict) -> None:
    jsonschema.validate(
        manifest, json.loads((ROOT / "schemas/a0_sft_inputs_manifest.schema.json").read_text())
    )


def verify_inputs(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    config_path = ROOT / "configs/a0_sft_inputs.yaml"
    config = load_config(config_path)
    if (
        sha256_file(config_path) != manifest["config_sha256"]
        or config["release_manifest_sha256"] != manifest["source_release_sha256"]
        or config["tokenizer_sha256"] != manifest["tokenizer_sha256"]
        or config["training_config_sha256"] != manifest["training_config_sha256"]
        or sha256_file(ROOT / "configs/training_data.yaml") != manifest["training_config_sha256"]
    ):
        raise ValueError("SFT configuration provenance mismatch")
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    release = data_root / "releases" / config["release_id"]
    if sha256_file(release / "manifest.json") != manifest["source_release_sha256"]:
        raise ValueError("canonical source release identity mismatch")
    source_manifest = verify_release(release)
    sources = source_index(release)
    if (
        source_manifest["episode_count"] != manifest["scanned_episodes"]
        or source_manifest["decision_count"] != manifest["scanned_decisions"]
        or len(sources) != manifest["scanned_decisions"]
    ):
        raise ValueError("canonical source count mismatch")
    if set(manifest["files"]) != {"train.parquet", "dev.parquet", "test.parquet", "excluded.jsonl"}:
        raise ValueError("SFT file inventory changed")
    for name, digest in manifest["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("SFT artifact file hash mismatch")
    sample_ids, decision_ids, owners = set(), set(), {}
    counts, lengths, losses = {}, {}, {}
    for split in SPLITS:
        path = root / f"{split}.parquet"
        if not pq.read_schema(path).equals(TOKEN_SCHEMA, check_metadata=True):
            raise ValueError("SFT token schema changed")
        counts[split], lengths[split], losses[split] = 0, [], []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                validate_record(row, split=split, max_tokens=manifest["max_tokens"])
                verify_source_row(row, sources)
                if row["sample_id"] in sample_ids or row["decision_id"] in decision_ids:
                    raise ValueError("duplicate SFT sample or decision")
                sample_ids.add(row["sample_id"])
                decision_ids.add(row["decision_id"])
                for owner in (row["episode_id"], row["group_sha256"]):
                    if owners.setdefault(owner, split) != split:
                        raise ValueError("cross-split SFT group or episode")
                counts[split] += 1
                lengths[split].append(len(row["token_ids"]) - 1)
                losses[split].append(sum(w > 0 for w in row["token_loss_weights"][1:]))
        if (
            distribution(lengths[split]) != manifest["input_tokens"][split]
            or distribution(losses[split]) != manifest["loss_tokens"][split]
        ):
            raise ValueError("SFT token statistics differ")
    rejected = {s: Counter() for s in SPLITS}
    with (root / "excluded.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if (
                set(row) != {"episode_id", "decision_id", "split", "reason"}
                or row["split"] not in SPLITS
                or row["decision_id"] in decision_ids
                or row["reason"]
                not in {
                    "source_failure_requires_verified_recovery",
                    "protected_context_overflow_8k",
                }
            ):
                raise ValueError("invalid exclusion or duplicate decision accounting")
            verify_source_row(row, sources, excluded=True)
            decision_ids.add(row["decision_id"])
            rejected[row["split"]][row["reason"]] += 1
    if (
        counts != manifest["accepted"]
        or {s: dict(v) for s, v in rejected.items()} != manifest["rejected"]
        or decision_ids != set(sources)
    ):
        raise ValueError("incomplete SFT admission accounting")
    return {
        "schema_version": 1,
        "status": "passed",
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "accepted": counts,
        "excluded": {s: sum(v.values()) for s, v in rejected.items()},
        "total_decisions_accounted": len(decision_ids),
        "canonical_source_join": "all_decisions_exact",
        "cross_split_groups": 0,
        "duplicate_sample_ids": 0,
    }


def load_split(root: Path, split: str, *, expected_manifest_sha256: str) -> list[TokenizedEpisode]:
    if split not in SPLITS or sha256_file(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError("SFT split or manifest identity mismatch")
    verification = json.loads((root / "verification.json").read_text())
    if (
        verification["status"] != "passed"
        or verification["manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("SFT inputs require independent verification")
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    path = root / f"{split}.parquet"
    if sha256_file(path) != manifest["files"][path.name] or not pq.read_schema(path).equals(
        TOKEN_SCHEMA, check_metadata=True
    ):
        raise ValueError("SFT split hash or schema mismatch")
    result = [
        to_tokenized(row, split=split, max_tokens=manifest["max_tokens"])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16)
        for row in batch.to_pylist()
    ]
    if len(result) != manifest["accepted"][split] or len({r.sample_id for r in result}) != len(
        result
    ):
        raise ValueError("SFT split count or uniqueness mismatch")
    return result
