#!/usr/bin/env python3
"""Download pinned Hugging Face datasets and RWKV checkpoints.

Every source must use a full 40-character commit. Downloads are resumable through
the Hugging Face cache and are materialized below the configured data root.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import yaml
from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = REPO_ROOT / "configs" / "sources.yaml"
DEFAULT_MODEL = REPO_ROOT / "configs" / "base_model.yaml"
SOURCE_SCHEMA = REPO_ROOT / "schemas" / "source_registry.schema.json"


@dataclass(frozen=True)
class Asset:
    source_id: str
    repo_id: str
    repo_type: str
    revision: str
    license: str
    allow_patterns: tuple[str, ...]
    expected_lfs_bytes: int
    destination_kind: str
    expected_sha256: dict[str, str]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def assert_pinned(revision: str) -> None:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"Revision must be a lowercase 40-character commit: {revision!r}")


def load_data_assets(path: Path) -> tuple[list[Asset], Path]:
    config = load_yaml(path)
    with SOURCE_SCHEMA.open("r", encoding="utf-8") as handle:
        jsonschema.validate(config, json.load(handle))
    storage_config = (REPO_ROOT / config["storage_config"]).resolve()
    assets = []
    for item in config["sources"]:
        assert_pinned(item["revision"])
        assets.append(
            Asset(
                source_id=item["source_id"],
                repo_id=item["repo_id"],
                repo_type=item["repo_type"],
                revision=item["revision"],
                license=item["declared_license"],
                allow_patterns=tuple(item["allow_patterns"]),
                expected_lfs_bytes=item["expected_lfs_bytes"],
                destination_kind="raw",
                expected_sha256={},
            )
        )
    return assets, storage_config


def load_model_asset(path: Path) -> Asset:
    config = load_yaml(path)
    repo = config["model_repo"]
    assert_pinned(repo["revision"])
    roles = config["initial_download_roles"]
    checkpoints = config["checkpoints"]
    selected = [checkpoints[role] for role in roles]
    expected_sha256 = {item["file"]: item["sha256"] for item in selected}
    return Asset(
        source_id=repo["repo_id"].replace("/", "__").lower(),
        repo_id=repo["repo_id"],
        repo_type=repo["repo_type"],
        revision=repo["revision"],
        license=repo["license"],
        allow_patterns=("README.md", *(item["file"] for item in selected)),
        expected_lfs_bytes=sum(item["size_bytes"] for item in selected),
        destination_kind="models",
        expected_sha256=expected_sha256,
    )


def lfs_metadata(sibling: Any) -> tuple[int, str | None]:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return 0, None
    if isinstance(lfs, dict):
        return int(lfs.get("size", 0)), lfs.get("sha256")
    return int(getattr(lfs, "size", 0)), getattr(lfs, "sha256", None)


def is_selected(filename: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def destination_for(root: Path, asset: Asset) -> Path:
    destination = (root / asset.destination_kind / asset.source_id / asset.revision).resolve()
    expected_parent = (root / asset.destination_kind).resolve()
    if not destination.is_relative_to(expected_parent):
        raise ValueError(f"Unsafe destination resolved for {asset.source_id}: {destination}")
    return destination


def preflight(api: HfApi, asset: Asset) -> tuple[list[dict[str, Any]], int]:
    info = api.repo_info(
        repo_id=asset.repo_id,
        repo_type=asset.repo_type,
        revision=asset.revision,
        files_metadata=True,
    )
    if info.sha != asset.revision:
        raise RuntimeError(
            f"Resolved revision mismatch for {asset.repo_id}: {info.sha} != {asset.revision}"
        )

    files: list[dict[str, Any]] = []
    lfs_bytes = 0
    for sibling in info.siblings:
        if not is_selected(sibling.rfilename, asset.allow_patterns):
            continue
        lfs_size, lfs_sha = lfs_metadata(sibling)
        lfs_bytes += lfs_size
        files.append(
            {
                "path": sibling.rfilename,
                "size": int(getattr(sibling, "size", 0) or 0),
                "lfs_size": lfs_size,
                "lfs_sha256": lfs_sha,
            }
        )
    if not files:
        raise RuntimeError(f"No files selected for {asset.repo_id}")
    if lfs_bytes != asset.expected_lfs_bytes:
        raise RuntimeError(
            f"Pinned file-size mismatch for {asset.source_id}: "
            f"registry={asset.expected_lfs_bytes}, upstream={lfs_bytes}"
        )
    return files, lfs_bytes


def verify_local_files(destination: Path, files: list[dict[str, Any]], asset: Asset) -> None:
    for item in files:
        path = destination / item["path"]
        if not path.is_file():
            raise RuntimeError(f"Downloaded file missing: {path}")
        expected_size = item["size"] or item["lfs_size"]
        if expected_size and path.stat().st_size != expected_size:
            raise RuntimeError(
                f"File-size mismatch for {path}: {path.stat().st_size} != {expected_size}"
            )
        expected_sha = asset.expected_sha256.get(item["path"])
        if expected_sha:
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise RuntimeError(f"SHA-256 mismatch for {path}: {actual_sha} != {expected_sha}")
            item["local_sha256"] = actual_sha


def download_asset(
    api: HfApi,
    asset: Asset,
    root: Path,
    cache_dir: Path,
    workers: int,
    inspect_only: bool,
) -> None:
    files, lfs_bytes = preflight(api, asset)
    destination = destination_for(root, asset)
    gib = lfs_bytes / (1024**3)
    print(f"{asset.source_id}: {len(files)} files, {gib:.2f} GiB -> {destination}")
    if inspect_only:
        return

    destination.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=asset.repo_id,
        repo_type=asset.repo_type,
        revision=asset.revision,
        allow_patterns=list(asset.allow_patterns),
        local_dir=destination,
        cache_dir=cache_dir,
        max_workers=workers,
    )
    verify_local_files(destination, files, asset)
    manifest = {
        "schema_version": 1,
        "source_id": asset.source_id,
        "repo_id": asset.repo_id,
        "repo_type": asset.repo_type,
        "revision": asset.revision,
        "declared_license": asset.license,
        "allow_patterns": list(asset.allow_patterns),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "lfs_bytes": lfs_bytes,
        "files": files,
    }
    manifest_path = destination / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"verified: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-config", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--storage-config", type=Path)
    parser.add_argument("--source", action="append", default=[], help="Data source_id; repeatable")
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--models", action="store_true", help="Download initial model roles")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_assets, default_storage = load_data_assets(args.sources_config.resolve())
    storage_path = (args.storage_config or default_storage).resolve()
    storage = load_yaml(storage_path)
    root = Path(storage["storage"]["local_root"]).resolve()
    cache_dir = root / storage["storage"]["cache_dir"]

    selected: list[Asset] = []
    requested = set(args.source)
    known = {asset.source_id for asset in data_assets}
    unknown = requested - known
    if unknown:
        raise SystemExit(f"Unknown source_id(s): {', '.join(sorted(unknown))}")
    if args.all_data:
        selected.extend(data_assets)
    else:
        selected.extend(asset for asset in data_assets if asset.source_id in requested)
    if args.models:
        selected.append(load_model_asset(args.model_config.resolve()))
    if not selected:
        raise SystemExit("Select --all-data, --source SOURCE_ID, and/or --models")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    api = HfApi()
    for asset in selected:
        download_asset(api, asset, root, cache_dir, args.workers, args.inspect_only)


if __name__ == "__main__":
    main()
