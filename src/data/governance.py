"""Deterministic split-group and frozen held-out policy contracts."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

HELDOUT_SCHEMA_VERSION = 1
GROUP_IDENTITY_VERSION = "repo_else_task_v1"
SPLIT_ALGORITHM = "sha256_salted_bucket_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COPY_SUFFIX = re.compile(r"_copy\d+$", re.IGNORECASE)
SWE_TASK = re.compile(r"^(?P<owner>[^/]+)__(?P<repo>.+)-\d+$")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def normalize_repo(repo: str) -> str:
    value = _normalized(repo).replace("\\", "/")
    for prefix in ("https://github.com/", "http://github.com/", "github.com/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    if "/" not in value and "__" in value:
        value = value.replace("__", "/", 1)
    value = re.sub(r"/+", "/", value).strip("/")
    value = value.removesuffix(".git")
    if not value or value in {"none", "null", "unknown"}:
        raise ValueError("repo identity is empty or unknown")
    return value


def canonical_group_identity(repo: str | None, task_id: str) -> str:
    if repo is not None and _normalized(str(repo)) not in {"", "none", "null", "unknown"}:
        return f"repo:{normalize_repo(str(repo))}"
    task = COPY_SUFFIX.sub("", _normalized(str(task_id)))
    match = SWE_TASK.fullmatch(task)
    if match:
        inferred_repo = f"{match.group('owner')}/{match.group('repo')}"
        return f"repo:{normalize_repo(inferred_repo)}"
    if not task or task in {"none", "null", "unknown"}:
        raise ValueError("task identity is empty or unknown")
    return f"task:{task}"


def group_digest(identity: str, salt: str) -> str:
    if not identity.startswith(("repo:", "task:")):
        raise ValueError("group identity must use a versioned namespace")
    if not salt:
        raise ValueError("held-out salt must not be empty")
    payload = f"{SPLIT_ALGORITHM}\0{GROUP_IDENTITY_VERSION}\0{salt}\0{identity}".encode()
    return hashlib.sha256(payload).hexdigest()


def digest_bucket(digest: str, bucket_count: int) -> int:
    if not HEX64.fullmatch(digest):
        raise ValueError("group digest must be lowercase SHA-256")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    return int(digest[:16], 16) % bucket_count


@dataclass(frozen=True)
class SplitRange:
    start: int
    end: int

    def contains(self, bucket: int) -> bool:
        return self.start <= bucket <= self.end


@dataclass(frozen=True)
class HeldoutPolicy:
    salt: str
    bucket_count: int
    train: SplitRange
    dev: SplitRange
    test: SplitRange
    source_revisions: dict[str, str]
    dev_group_hashes: frozenset[str]
    test_group_hashes: frozenset[str]
    manifest_sha256: str

    def split_for_identity(self, identity: str) -> str:
        digest = group_digest(identity, self.salt)
        bucket = digest_bucket(digest, self.bucket_count)
        matches = [
            name
            for name, split_range in (("train", self.train), ("dev", self.dev), ("test", self.test))
            if split_range.contains(bucket)
        ]
        if len(matches) != 1:
            raise ValueError(f"split ranges do not cover bucket {bucket} exactly once")
        return matches[0]

    def split_for_episode(self, repo: str | None, task_id: str) -> str:
        return self.split_for_identity(canonical_group_identity(repo, task_id))

    def verify_source(self, source_id: str, revision: str) -> None:
        if self.source_revisions.get(source_id) != revision:
            raise ValueError(f"held-out policy does not freeze {source_id}@{revision}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(mapping: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(expected - actual)} unknown={sorted(actual - expected)}"
        )


def _load_hash_list(root: Path, descriptor: dict[str, Any], split: str) -> frozenset[str]:
    _exact_keys(descriptor, {"path", "sha256", "count"}, f"{split} list descriptor")
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("held-out list path must remain inside the manifest directory")
    path = root / relative
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError(f"{split} held-out list SHA-256 mismatch")
    values = path.read_text().splitlines()
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError(f"{split} held-out hashes must be sorted and unique")
    if len(values) != descriptor["count"] or any(not HEX64.fullmatch(value) for value in values):
        raise ValueError(f"{split} held-out list count or digest format mismatch")
    return frozenset(values)


def load_heldout_policy(path: str | Path) -> HeldoutPolicy:
    manifest_path = Path(path).resolve()
    payload = yaml.safe_load(manifest_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("held-out manifest must be a mapping")
    _exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "created_at",
            "group_identity_version",
            "split_algorithm",
            "salt",
            "bucket_count",
            "ranges",
            "sources",
            "lists",
            "statistics",
        },
        "held-out manifest",
    )
    if payload["schema_version"] != HELDOUT_SCHEMA_VERSION or payload["status"] != "frozen":
        raise ValueError("held-out manifest must be frozen schema version 1")
    if payload["group_identity_version"] != GROUP_IDENTITY_VERSION:
        raise ValueError("unknown held-out group identity version")
    if payload["split_algorithm"] != SPLIT_ALGORITHM:
        raise ValueError("unknown held-out split algorithm")
    ranges = payload["ranges"]
    _exact_keys(ranges, {"train", "dev", "test"}, "split ranges")

    def parse_range(name: str) -> SplitRange:
        values = ranges[name]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"{name} range must be [start, end]")
        return SplitRange(int(values[0]), int(values[1]))

    source_revisions = {}
    for source in payload["sources"]:
        _exact_keys(source, {"source_id", "revision", "episode_count", "group_count"}, "source")
        if source["source_id"] in source_revisions:
            raise ValueError("held-out source IDs must be unique")
        source_revisions[source["source_id"]] = source["revision"]
    lists = payload["lists"]
    _exact_keys(lists, {"dev", "test"}, "held-out lists")
    dev = _load_hash_list(manifest_path.parent, lists["dev"], "dev")
    test = _load_hash_list(manifest_path.parent, lists["test"], "test")
    if dev & test:
        raise ValueError("dev and test held-out group hashes overlap")
    policy = HeldoutPolicy(
        salt=str(payload["salt"]),
        bucket_count=int(payload["bucket_count"]),
        train=parse_range("train"),
        dev=parse_range("dev"),
        test=parse_range("test"),
        source_revisions=source_revisions,
        dev_group_hashes=dev,
        test_group_hashes=test,
        manifest_sha256=sha256_file(manifest_path),
    )
    for bucket in range(policy.bucket_count):
        coverage = sum(
            split_range.contains(bucket) for split_range in (policy.train, policy.dev, policy.test)
        )
        if coverage != 1:
            raise ValueError(f"split ranges cover bucket {bucket} {coverage} times")
    for digest in dev:
        if not policy.dev.contains(digest_bucket(digest, policy.bucket_count)):
            raise ValueError("dev held-out list contains a digest outside the dev range")
    for digest in test:
        if not policy.test.contains(digest_bucket(digest, policy.bucket_count)):
            raise ValueError("test held-out list contains a digest outside the test range")
    return policy
