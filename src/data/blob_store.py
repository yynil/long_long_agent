"""Content-addressed storage for immutable raw trajectory records."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .util import canonical_json


class BlobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def put_json(self, value: Any) -> str:
        payload = canonical_json(value).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        destination = self.root / "sha256" / digest[:2] / f"{digest}.json.gz"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            compressed = gzip.compress(payload, compresslevel=6, mtime=0)
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent, prefix=f".{digest}.", suffix=".tmp"
            )
            try:
                with os.fdopen(file_descriptor, "wb") as handle:
                    handle.write(compressed)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temporary_name).replace(destination)
            finally:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()
        return f"sha256:{digest}"

    def read_json(self, reference: str) -> Any:
        algorithm, separator, digest = reference.partition(":")
        if algorithm != "sha256" or not separator or len(digest) != 64:
            raise ValueError(f"Invalid blob reference: {reference!r}")
        path = self.root / "sha256" / digest[:2] / f"{digest}.json.gz"
        payload = gzip.decompress(path.read_bytes())
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RuntimeError(f"Blob digest mismatch: {path}")
        return json.loads(payload)
