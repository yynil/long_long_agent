import json
from copy import deepcopy
from dataclasses import replace

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.data.blob_store import BlobStore
from src.data.canonical import canonicalize_episode
from src.data.governance import sha256_file
from src.data.types import NormalizedEpisode, NormalizedMessage
from src.model.runtime import ROOT
from src.training.sft_inputs import (
    SPLITS,
    TOKEN_SCHEMA,
    distribution,
    encode_record,
    load_config,
    load_split,
    record_digest,
    source_index,
    to_tokenized,
    validate_manifest,
    validate_record,
    verify_source_row,
)


class ByteTokenizer:
    def encode_bytes(self, source):
        return [b + 1 for b in source]

    def token_bytes(self, token):
        return bytes([token - 1])


@pytest.fixture
def encoded(tmp_path):
    episode = NormalizedEpisode(
        "fixture/agent",
        "a" * 40,
        "synthetic",
        "one",
        "task-one",
        "Inspect.",
        (
            NormalizedMessage("user", "Inspect."),
            NormalizedMessage(
                "assistant",
                reasoning="Read files.",
                action={"name": "bash", "arguments": {"command": "ls"}},
            ),
            NormalizedMessage("tool", "README.md"),
            NormalizedMessage("assistant", "Verified."),
        ),
        teacher_model="fixture-teacher",
        repo="fixture/project",
        base_commit="b" * 40,
        success=True,
    )
    erow, decisions = canonicalize_episode(episode, BlobStore(tmp_path / "blobs"))
    row = encode_record(
        erow, episode, decisions[1], 3, ByteTokenizer(), split="train", max_tokens=8192
    )
    return episode, erow, decisions, row


def test_full_decision_encoding_roundtrip_and_history_masks(encoded, tmp_path):
    _, _, _, row = encoded
    path = tmp_path / "tokens.parquet"
    pq.write_table(
        pa.Table.from_pylist([row], schema=TOKEN_SCHEMA), path, use_compliant_nested_type=False
    )
    assert pq.read_schema(path).equals(TOKEN_SCHEMA, check_metadata=True)
    restored = pq.read_table(path).to_pylist()[0]
    assert restored == row
    sample = to_tokenized(restored, split="train", max_tokens=8192)
    assert len(sample.token_ids) > 10
    assert all(
        w == 0
        for r, w in zip(sample.token_regions, sample.token_loss_weights)
        if r in {"user", "tool_response", "assistant_action", "assistant_reasoning"}
    )
    assert any(w == 1 for w in sample.token_loss_weights)


@pytest.mark.parametrize(
    "mutation", ["unknown", "split", "failure", "hash", "mask", "nan", "counter", "null_id"]
)
def test_token_records_fail_closed(encoded, mutation):
    row = deepcopy(encoded[3])
    if mutation == "unknown":
        row["surprise"] = True
    elif mutation == "split":
        row["split"] = "dev"
    elif mutation == "failure":
        row["source_success"] = False
    elif mutation == "hash":
        row["token_ids"][1] += 1
    elif mutation == "mask":
        row["token_loss_weights"][row["token_regions"].index("user")] = 1
    elif mutation == "nan":
        row["token_loss_weights"][2] = float("nan")
    elif mutation == "counter":
        row["mixed_boundary_count"] += 1
    else:
        row["episode_id"] = None
    with pytest.raises(ValueError):
        validate_record(row, split="train", max_tokens=8192)


def test_positive_selection_and_protected_overflow(encoded):
    episode, erow, decisions, _ = encoded
    with pytest.raises(ValueError, match="failure source"):
        encode_record(
            dict(erow, success=False),
            episode,
            decisions[0],
            1,
            ByteTokenizer(),
            split="train",
            max_tokens=8192,
        )
    with pytest.raises(ValueError, match="protected context"):
        encode_record(
            erow,
            replace(episode, task_text="Inspect."),
            decisions[0],
            1,
            ByteTokenizer(),
            split="train",
            max_tokens=16,
        )


def test_canonical_source_join_and_wrong_outcome_rejected(encoded, tmp_path):
    _, erow, decisions, row = encoded
    pq.write_table(pa.Table.from_pylist([erow]), tmp_path / "episodes.parquet")
    pq.write_table(pa.Table.from_pylist(decisions), tmp_path / "decisions.parquet")
    (tmp_path / "admission.json").write_text(
        json.dumps(
            {
                "accepted": [
                    {"episode_id": erow["episode_id"], "split": "train", "message_indices": [1, 3]}
                ]
            }
        )
    )
    (tmp_path / "splits").mkdir()
    for split in SPLITS:
        (tmp_path / f"splits/{split}.txt").write_text(
            erow["episode_id"] + "\n" if split == "train" else ""
        )
    sources = source_index(tmp_path)
    verify_source_row(row, sources)
    for key, value in (
        ("split", "dev"),
        ("teacher", "wrong"),
        ("episode_id", "c" * 64),
        ("message_index", 1),
    ):
        with pytest.raises(ValueError, match="canonical source"):
            verify_source_row(dict(row, **{key: value}), sources)
    excluded = {k: row[k] for k in ("episode_id", "decision_id", "split")}
    verify_source_row(
        dict(excluded, reason="protected_context_overflow_8k"), sources, excluded=True
    )
    with pytest.raises(ValueError, match="source outcome"):
        verify_source_row(
            dict(excluded, reason="source_failure_requires_verified_recovery"),
            sources,
            excluded=True,
        )
    with pytest.raises(ValueError, match="unknown canonical"):
        verify_source_row(dict(row, decision_id="0" * 64), sources)


def test_config_and_manifest_closed_and_loader_requires_verification(encoded, tmp_path):
    config = load_config(ROOT / "configs/a0_sft_inputs.yaml")
    assert config["training_config_sha256"] == sha256_file(ROOT / "configs/training_data.yaml")
    config["unknown"] = 1
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(jsonschema.ValidationError):
        load_config(config_path)
    row = encoded[3]
    pq.write_table(
        pa.Table.from_pylist([row], schema=TOKEN_SCHEMA),
        tmp_path / "train.parquet",
        use_compliant_nested_type=False,
    )
    manifest = {
        "schema_version": 1,
        "purpose": "a0_success_sft_inputs",
        "status": "built",
        "scanned_episodes": 1000,
        "scanned_decisions": 10000,
        "max_tokens": 8192,
        "accepted": {s: int(s == "train") for s in SPLITS},
        "rejected": {s: {} for s in SPLITS},
        "files": {
            name: "a" * 64
            for name in ("train.parquet", "dev.parquet", "test.parquet", "excluded.jsonl")
        },
        "code_commit": "a" * 40,
        "input_tokens": {s: distribution([]) for s in SPLITS},
        "loss_tokens": {s: distribution([]) for s in SPLITS},
        **{
            k: "a" * 64
            for k in (
                "config_sha256",
                "source_release_sha256",
                "tokenizer_sha256",
                "training_config_sha256",
            )
        },
    }
    manifest["files"]["train.parquet"] = sha256_file(tmp_path / "train.parquet")
    validate_manifest(manifest)
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(dict(manifest, extra=1))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    digest = sha256_file(tmp_path / "manifest.json")
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "train", expected_manifest_sha256=digest)
    (tmp_path / "verification.json").write_text(
        json.dumps({"status": "failed", "manifest_sha256": digest})
    )
    with pytest.raises(ValueError, match="independent verification"):
        load_split(tmp_path, "train", expected_manifest_sha256=digest)
    (tmp_path / "verification.json").write_text(
        json.dumps({"status": "passed", "manifest_sha256": digest})
    )
    assert (
        load_split(tmp_path, "train", expected_manifest_sha256=digest)[0].sample_id
        == row["sample_id"]
    )
    assert record_digest(row) == row["record_sha256"]
    (tmp_path / "train.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash or schema"):
        load_split(tmp_path, "train", expected_manifest_sha256=digest)
