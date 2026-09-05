#!/usr/bin/env python3
"""Convert one pinned source into canonical Parquet tables.

Production releases require a frozen held-out manifest. Use --preview with a
small --limit while developing adapters before D0 is complete.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import canonicalize_episode, get_adapter
from src.data.blob_store import BlobStore
from src.data.governance import load_heldout_policy
from src.data.schemas import (
    DECISIONS,
    EPISODES,
    FORKS,
    SCHEMA_VERSION,
    SNAPSHOTS,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}")
    return value


def source_entry(config: dict[str, Any], source_id: str) -> dict[str, Any]:
    matches = [item for item in config["sources"] if item["source_id"] == source_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source {source_id!r}, found {len(matches)}")
    return matches[0]


def selected_parquet_files(raw_dir: Path, patterns: list[str]) -> list[Path]:
    files = []
    for path in raw_dir.rglob("*.parquet"):
        relative = path.relative_to(raw_dir).as_posix()
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
            files.append(path)
    return sorted(files)


class BufferedWriter:
    def __init__(self, path: Path, schema: pa.Schema, flush_rows: int):
        self.schema = schema
        self.flush_rows = flush_rows
        self.buffer: list[dict[str, Any]] = []
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def add(self, *rows: dict[str, Any]) -> None:
        self.buffer.extend(rows)
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
            self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--input-pattern",
        action="append",
        default=[],
        help="Preview-only extra file filter; repeatable.",
    )
    parser.add_argument("--sources-config", type=Path, default=REPO_ROOT / "configs/sources.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.input_pattern and not args.preview:
        raise SystemExit("--input-pattern is restricted to --preview conversions")
    if not args.preview:
        raise SystemExit(
            "Production conversion requires audited admission; use scripts/build_a0_release.py. "
            "This adapter development entrypoint only supports --preview --limit N."
        )
    if args.limit is None:
        raise SystemExit("Adapter previews require an explicit --limit")
    heldout_manifest = REPO_ROOT / "data/heldout/manifest.yaml"
    if not args.preview and not heldout_manifest.is_file():
        raise SystemExit(
            "Production conversion requires frozen data/heldout/manifest.yaml; "
            "use --preview --limit N for adapter development"
        )

    sources_config = load_yaml(args.sources_config.resolve())
    source = source_entry(sources_config, args.source)
    heldout_policy = None
    if not args.preview:
        heldout_policy = load_heldout_policy(heldout_manifest)
        heldout_policy.verify_source(source["source_id"], source["revision"])
    storage = load_yaml((REPO_ROOT / sources_config["storage_config"]).resolve())["storage"]
    root = Path(storage["local_root"]).resolve()
    raw_dir = root / storage["raw_dir"] / source["source_id"] / source["revision"]
    if not (raw_dir / "download_manifest.json").is_file():
        raise SystemExit(f"Source download is incomplete or unverified: {raw_dir}")
    parquet_files = selected_parquet_files(raw_dir, source["allow_patterns"])
    if args.input_pattern:
        parquet_files = [
            path
            for path in parquet_files
            if any(
                fnmatch.fnmatchcase(path.relative_to(raw_dir).as_posix(), pattern)
                for pattern in args.input_pattern
            )
        ]
    if not parquet_files:
        raise SystemExit(f"No selected Parquet files found below {raw_dir}")

    release_root = root / storage["releases_dir"] / args.release_id
    destination = release_root / source["source_id"]
    temporary = release_root / f".{source['source_id']}.tmp"
    if destination.exists() or temporary.exists():
        raise SystemExit(f"Refusing to overwrite existing conversion: {destination} or {temporary}")
    temporary.mkdir(parents=True)
    blob_store = BlobStore(root / storage["blobs_dir"])
    episode_writer = BufferedWriter(temporary / "episodes.parquet", EPISODES, 256)
    decision_writer = BufferedWriter(temporary / "decisions.parquet", DECISIONS, 2048)
    adapter = get_adapter(source["adapter"])

    episode_count = 0
    decision_count = 0
    split_counts = {"train": 0, "dev": 0, "test": 0}
    try:
        for parquet_path in parquet_files:
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(batch_size=args.batch_size):
                for row in batch.to_pylist():
                    row["__source_file__"] = parquet_path.relative_to(raw_dir).as_posix()
                    normalized = adapter(row, source["revision"], source["declared_license"])
                    episode_row, decision_rows = canonicalize_episode(normalized, blob_store)
                    if heldout_policy is not None:
                        split = heldout_policy.split_for_identity(episode_row["split_group"])
                        split_counts[split] += 1
                    episode_writer.add(episode_row)
                    decision_writer.add(*decision_rows)
                    episode_count += 1
                    decision_count += len(decision_rows)
                    if args.limit is not None and episode_count >= args.limit:
                        break
                if args.limit is not None and episode_count >= args.limit:
                    break
            if args.limit is not None and episode_count >= args.limit:
                break
    finally:
        episode_writer.close()
        decision_writer.close()

    pq.write_table(pa.Table.from_pylist([], schema=SNAPSHOTS), temporary / "snapshots.parquet")
    pq.write_table(pa.Table.from_pylist([], schema=FORKS), temporary / "forks.parquet")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": args.release_id,
        "preview": args.preview,
        "source_id": source["source_id"],
        "source_revision": source["revision"],
        "adapter": source["adapter"],
        "preview_input_patterns": args.input_pattern,
        "episode_count": episode_count,
        "decision_count": decision_count,
        "split_counts": split_counts if heldout_policy is not None else None,
        "heldout_manifest_sha256": (
            heldout_policy.manifest_sha256 if heldout_policy is not None else None
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "input_files": [path.relative_to(raw_dir).as_posix() for path in parquet_files],
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    print(
        f"canonicalized {episode_count} episodes / {decision_count} decisions "
        f"from {source['source_id']} -> {destination}"
    )


if __name__ == "__main__":
    main()
