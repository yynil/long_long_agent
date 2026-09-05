from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data.governance import (
    GROUP_IDENTITY_VERSION,
    SPLIT_ALGORITHM,
    canonical_group_identity,
    digest_bucket,
    group_digest,
    load_heldout_policy,
    sha256_file,
)


def test_group_identity_joins_repo_spelling_and_swe_task_variants() -> None:
    expected = "repo:python-attrs/attrs"
    assert canonical_group_identity("python-attrs/attrs", "ignored") == expected
    assert canonical_group_identity("python-attrs__attrs", "ignored") == expected
    assert canonical_group_identity(None, "python-attrs__attrs-770") == expected
    assert (
        canonical_group_identity("https://github.com/python-attrs/attrs.git", "ignored") == expected
    )
    assert canonical_group_identity(None, "swesmith-31327_copy0002") == "task:swesmith-31327"


def test_group_digest_and_bucket_are_deterministic_and_salted() -> None:
    identity = "repo:owner/project"
    first = group_digest(identity, "salt-a")
    assert first == group_digest(identity, "salt-a")
    assert first != group_digest(identity, "salt-b")
    assert 0 <= digest_bucket(first, 10_000) < 10_000
    with pytest.raises(ValueError):
        canonical_group_identity(None, "unknown")


def _digest_in_bucket(bucket: int, salt: str) -> str:
    for index in range(1000):
        digest = group_digest(f"task:fixture-{bucket}-{index}", salt)
        if digest_bucket(digest, 10) == bucket:
            return digest
    raise AssertionError("could not construct fixture digest")


def write_policy(root: Path) -> Path:
    salt = "fixture-salt"
    dev = _digest_in_bucket(0, salt)
    test = _digest_in_bucket(1, salt)
    (root / "dev_group_hashes.txt").write_text(f"{dev}\n")
    (root / "test_group_hashes.txt").write_text(f"{test}\n")
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "created_at": "2026-09-04T00:00:00+00:00",
        "group_identity_version": GROUP_IDENTITY_VERSION,
        "split_algorithm": SPLIT_ALGORITHM,
        "salt": salt,
        "bucket_count": 10,
        "ranges": {"dev": [0, 0], "test": [1, 1], "train": [2, 9]},
        "sources": [
            {
                "source_id": "fixture",
                "revision": "a" * 40,
                "episode_count": 2,
                "group_count": 2,
            }
        ],
        "lists": {
            "dev": {
                "path": "dev_group_hashes.txt",
                "sha256": sha256_file(root / "dev_group_hashes.txt"),
                "count": 1,
            },
            "test": {
                "path": "test_group_hashes.txt",
                "sha256": sha256_file(root / "test_group_hashes.txt"),
                "count": 1,
            },
        },
        "statistics": {"global_group_count": 2, "cross_source_group_count": 0},
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def test_frozen_policy_loads_lists_verifies_sources_and_covers_buckets(tmp_path) -> None:
    path = write_policy(tmp_path)
    policy = load_heldout_policy(path)
    policy.verify_source("fixture", "a" * 40)
    assert len(policy.dev_group_hashes) == 1
    assert len(policy.test_group_hashes) == 1
    with pytest.raises(ValueError, match="does not freeze"):
        policy.verify_source("fixture", "b" * 40)

    (tmp_path / "dev_group_hashes.txt").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_heldout_policy(path)


def test_project_heldout_manifest_verifies_every_pinned_source() -> None:
    policy = load_heldout_policy("data/heldout/manifest.yaml")
    sources = yaml.safe_load(Path("configs/sources.yaml").read_text())["sources"]
    for source in sources:
        policy.verify_source(source["source_id"], source["revision"])
    assert len(policy.dev_group_hashes) == 2268
    assert len(policy.test_group_hashes) == 2194
