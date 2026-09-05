"""Pinned source iteration shared by audit and release builders."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from .governance import canonical_group_identity, sha256_file
from .util import parse_json_maybe


def selected_files(root: Path, patterns: list[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.parquet")
        if any(
            fnmatch.fnmatchcase(path.relative_to(root).as_posix(), pattern) for pattern in patterns
        )
    )


def load_sources(config: Path) -> tuple[dict[str, Any], Path]:
    payload = yaml.safe_load(config.read_text())
    from jsonschema import validate

    repo = Path(__file__).resolve().parents[2]
    validate(payload, json.loads((repo / "schemas/source_registry.schema.json").read_text()))
    storage = yaml.safe_load((repo / payload["storage_config"]).read_text())["storage"]
    return payload, Path(storage["local_root"])


def verified_inventory(source: dict[str, Any], data_root: Path) -> tuple[Path, list[dict]]:
    root = data_root / "raw" / source["source_id"] / source["revision"]
    manifest = json.loads((root / "download_manifest.json").read_text())
    if manifest["revision"] != source["revision"] or manifest["repo_id"] != source["repo_id"]:
        raise ValueError("source manifest identity mismatch")
    expected = {
        item["path"]: item for item in manifest["files"] if item["path"].endswith(".parquet")
    }
    paths = selected_files(root, source["allow_patterns"])
    if {p.relative_to(root).as_posix() for p in paths} != set(expected):
        raise ValueError("source inventory mismatch")
    result = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        if digest != expected[relative].get("lfs_sha256"):
            raise ValueError("source content hash mismatch")
        result.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    return root, result


def iter_rows(root: Path, files: list[dict], *, columns: list[str] | None = None):
    for item in files:
        parquet = pq.ParquetFile(root / item["path"])
        ordinal = 0
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                row["__source_file__"] = item["path"]
                yield item["path"], ordinal, row
                ordinal += 1


def task_identity_and_text(adapter: str, row: dict) -> tuple[str, str]:
    if adapter == "orchard_swe":
        meta = parse_json_maybe(row["metadata"])
        identity = canonical_group_identity(meta.get("repo"), meta["instance_id"])
    elif adapter == "openthoughts_agent":
        identity = canonical_group_identity(None, row["task"])
    else:
        identity = canonical_group_identity(row.get("repo"), row["instance_id"])
    messages = row.get("messages") or row.get("conversations") or row.get("trajectory") or []
    messages = parse_json_maybe(messages)
    for item in messages:
        if item.get("role") == "user":
            content = str(item.get("content") or item.get("text") or "")
            if content.strip():
                return identity, content
    return identity, ""
