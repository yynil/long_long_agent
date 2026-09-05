#!/usr/bin/env python3
"""Print bounded structural summaries for Parquet files without dumping traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def summarize(value: Any, limit: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return {"type": "str", "chars": len(value), "preview": value[:limit]}
    if isinstance(value, bytes):
        return {"type": "bytes", "bytes": len(value)}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": len(value),
            "first": summarize(value[0], limit) if value else None,
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in value),
            "sample": {str(key): summarize(item, limit) for key, item in list(value.items())[:8]},
        }
    return {"type": type(value).__name__, "preview": repr(value)[:limit]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--preview-chars", type=int, default=120)
    args = parser.parse_args()

    parquet = pq.ParquetFile(args.path)
    print(f"path: {args.path}")
    print(f"rows: {parquet.metadata.num_rows}")
    print(f"row_groups: {parquet.metadata.num_row_groups}")
    print("schema:")
    print(parquet.schema_arrow)
    sample = parquet.read_row_group(0).slice(0, 1).to_pylist()[0]
    print("sample_structure:")
    print(
        json.dumps(
            {key: summarize(value, args.preview_chars) for key, value in sample.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
