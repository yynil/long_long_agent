#!/usr/bin/env python3
"""Audit Parquet metadata and adapter compatibility for one downloaded source.

The audit never emits trace content. It records file sizes, row counts, source
schema fingerprints, and whether the configured adapter accepts one row per
Parquet file.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_adapter


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}")
    return value


def source_entry(config: dict[str, Any], source_id: str) -> dict[str, Any]:
    matches = [item for item in config["sources"] if item["source_id"] == source_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one source {source_id!r}, found {len(matches)}")
    return matches[0]


def schema_fingerprint(parquet: pq.ParquetFile) -> str:
    payload = parquet.schema_arrow.serialize().to_pybytes()
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--sources-config", type=Path, default=REPO_ROOT / "configs/sources.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources_config = load_yaml(args.sources_config.resolve())
    source = source_entry(sources_config, args.source)
    storage_path = (REPO_ROOT / sources_config["storage_config"]).resolve()
    storage = load_yaml(storage_path)["storage"]
    raw_dir = (
        Path(storage["local_root"]) / storage["raw_dir"] / source["source_id"] / source["revision"]
    ).resolve()
    manifest_path = raw_dir / "download_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Verified download manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != source["revision"]:
        raise SystemExit("Download manifest revision does not match source registry")

    adapter = get_adapter(source["adapter"])
    patterns = source["allow_patterns"]
    parquet_paths = sorted(
        path
        for path in raw_dir.rglob("*.parquet")
        if any(
            fnmatch.fnmatchcase(path.relative_to(raw_dir).as_posix(), pattern)
            for pattern in patterns
        )
    )
    if not parquet_paths:
        raise SystemExit(f"No selected Parquet files found below {raw_dir}")

    files: list[dict[str, Any]] = []
    schema_counts: dict[str, int] = {}
    total_rows = 0
    total_bytes = 0
    sampled_episodes = 0
    sampled_decisions = 0
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        fingerprint = schema_fingerprint(parquet)
        rows = parquet.metadata.num_rows
        size = path.stat().st_size
        total_rows += rows
        total_bytes += size
        schema_counts[fingerprint] = schema_counts.get(fingerprint, 0) + 1

        sample = parquet.read_row_group(0).slice(0, 1).to_pylist()
        if sample:
            sample[0]["__source_file__"] = path.relative_to(raw_dir).as_posix()
            episode = adapter(sample[0], source["revision"], source["declared_license"])
            sampled_episodes += 1
            sampled_decisions += sum(message.role == "assistant" for message in episode.messages)
        files.append(
            {
                "path": path.relative_to(raw_dir).as_posix(),
                "bytes": size,
                "rows": rows,
                "row_groups": parquet.metadata.num_row_groups,
                "schema_fingerprint": fingerprint,
            }
        )

    audit = {
        "schema_version": 1,
        "source_id": source["source_id"],
        "source_revision": source["revision"],
        "adapter": source["adapter"],
        "audited_at": datetime.now(UTC).isoformat(),
        "parquet_file_count": len(files),
        "total_parquet_bytes": total_bytes,
        "total_rows": total_rows,
        "schema_fingerprints": schema_counts,
        "sampled_episodes": sampled_episodes,
        "sampled_assistant_messages": sampled_decisions,
        "files": files,
    }
    output = raw_dir / "source_audit.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"audited {len(files)} parquet files / {total_rows} rows / "
        f"{len(schema_counts)} schemas; adapter samples={sampled_episodes} -> {output}"
    )


if __name__ == "__main__":
    main()
