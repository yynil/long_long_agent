#!/usr/bin/env python3
"""Freeze deterministic train/dev/test group boundaries from pinned identity columns."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.governance import (
    GROUP_IDENTITY_VERSION,
    SPLIT_ALGORITHM,
    canonical_group_identity,
    digest_bucket,
    group_digest,
    load_heldout_policy,
    sha256_file,
)
from src.data.util import parse_json_maybe

SALT = "long-long-agent-heldout-v1-20260904"
BUCKET_COUNT = 10_000
RANGES = {"train": [1000, 9999], "dev": [500, 999], "test": [0, 499]}
IDENTITY_COLUMNS = {
    "openthoughts_agent": ("task",),
    "open_swe_traces": ("repo", "instance_id"),
    "orchard_swe": ("metadata",),
    "nebius_swe_agent": ("instance_id",),
    "nebius_openhands": ("repo", "instance_id"),
}


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping in {path}")
    return payload


def selected_files(raw_dir: Path, patterns: list[str]) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.rglob("*.parquet")
        if any(
            fnmatch.fnmatchcase(path.relative_to(raw_dir).as_posix(), pattern)
            for pattern in patterns
        )
    )


def identity_from_row(adapter: str, row: dict[str, Any]) -> str:
    if adapter == "openthoughts_agent":
        return canonical_group_identity(None, str(row.get("task") or "unknown"))
    if adapter in {"open_swe_traces", "nebius_openhands"}:
        return canonical_group_identity(row.get("repo"), str(row.get("instance_id") or "unknown"))
    if adapter == "nebius_swe_agent":
        return canonical_group_identity(None, str(row.get("instance_id") or "unknown"))
    if adapter == "orchard_swe":
        metadata = parse_json_maybe(row.get("metadata"), default={})
        if not isinstance(metadata, dict):
            raise ValueError("Orchard metadata is not an object")
        return canonical_group_identity(
            metadata.get("repo"),
            str(metadata.get("instance_id") or "unknown"),
        )
    raise ValueError(f"unsupported held-out identity adapter: {adapter}")


def split_for_digest(digest: str) -> str:
    bucket = digest_bucket(digest, BUCKET_COUNT)
    matches = [name for name, (start, end) in RANGES.items() if start <= bucket <= end]
    if len(matches) != 1:
        raise AssertionError(f"bucket {bucket} has {len(matches)} split assignments")
    return matches[0]


def scan_sources(sources_config: dict[str, Any], storage_config: dict[str, Any]):
    storage = storage_config["storage"]
    raw_root = Path(storage["local_root"]) / storage["raw_dir"]
    identity_sources: dict[str, int] = {}
    source_summaries = []
    episode_split_counts = {"train": 0, "dev": 0, "test": 0}
    invalid_identity_count = 0
    for source_index, source in enumerate(sources_config["sources"]):
        adapter = source["adapter"]
        columns = IDENTITY_COLUMNS[adapter]
        raw_dir = raw_root / source["source_id"] / source["revision"]
        if not (raw_dir / "download_manifest.json").is_file():
            raise ValueError(f"download manifest missing for {source['source_id']}")
        files = selected_files(raw_dir, source["allow_patterns"])
        if not files:
            raise ValueError(f"no selected Parquet files for {source['source_id']}")
        source_identities: set[str] = set()
        episode_count = 0
        for path in files:
            parquet = pq.ParquetFile(path)
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"identity columns missing in {path.name}: {sorted(missing)}")
            for batch in parquet.iter_batches(columns=list(columns), batch_size=4096):
                for row in batch.to_pylist():
                    episode_count += 1
                    try:
                        identity = identity_from_row(adapter, row)
                    except (TypeError, ValueError):
                        invalid_identity_count += 1
                        continue
                    digest = group_digest(identity, SALT)
                    split = split_for_digest(digest)
                    episode_split_counts[split] += 1
                    source_identities.add(identity)
                    identity_sources[identity] = identity_sources.get(identity, 0) | (
                        1 << source_index
                    )
        source_summaries.append(
            {
                "source_id": source["source_id"],
                "revision": source["revision"],
                "episode_count": episode_count,
                "group_count": len(source_identities),
            }
        )
    return identity_sources, source_summaries, episode_split_counts, invalid_identity_count


def write_hashes(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-config", type=Path, default=REPO_ROOT / "configs/sources.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/heldout")
    parser.add_argument(
        "--schema",
        type=Path,
        default=REPO_ROOT / "schemas/heldout_manifest.schema.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources_path = args.sources_config.resolve()
    sources_config = load_yaml(sources_path)
    storage_config = load_yaml((REPO_ROOT / sources_config["storage_config"]).resolve())
    identities, sources, episode_splits, invalid = scan_sources(sources_config, storage_config)
    group_hashes = {"train": [], "dev": [], "test": []}
    for identity in identities:
        digest = group_digest(identity, SALT)
        group_hashes[split_for_digest(digest)].append(digest)
    for values in group_hashes.values():
        values.sort()
    cross_source = sum(mask.bit_count() > 1 for mask in identities.values())
    summary = {
        "episode_count": sum(item["episode_count"] for item in sources),
        "global_group_count": len(identities),
        "cross_source_group_count": cross_source,
        "invalid_identity_count": invalid,
        "episode_split_counts": episode_splits,
        "group_split_counts": {name: len(values) for name, values in group_hashes.items()},
    }
    print(json.dumps(summary, sort_keys=True))
    if invalid:
        raise SystemExit("identity scan found invalid rows; refusing to freeze held-out policy")
    if args.dry_run:
        return 0

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen held-out directory: {output}")
    temporary = output.parent / f".{output.name}.tmp"
    if temporary.exists():
        raise SystemExit(f"temporary held-out directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    dev_path = temporary / "dev_group_hashes.txt"
    test_path = temporary / "test_group_hashes.txt"
    write_hashes(dev_path, group_hashes["dev"])
    write_hashes(test_path, group_hashes["test"])
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "created_at": datetime.now(UTC).isoformat(),
        "group_identity_version": GROUP_IDENTITY_VERSION,
        "split_algorithm": SPLIT_ALGORITHM,
        "salt": SALT,
        "bucket_count": BUCKET_COUNT,
        "ranges": RANGES,
        "sources": sources,
        "lists": {
            "dev": {
                "path": dev_path.name,
                "sha256": sha256_file(dev_path),
                "count": len(group_hashes["dev"]),
            },
            "test": {
                "path": test_path.name,
                "sha256": sha256_file(test_path),
                "count": len(group_hashes["test"]),
            },
        },
        "statistics": summary,
    }
    schema = json.loads(args.schema.resolve().read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)
    manifest_path = temporary / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    load_heldout_policy(manifest_path)
    temporary.replace(output)
    print(f"frozen held-out policy -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
