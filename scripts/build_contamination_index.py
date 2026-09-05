#!/usr/bin/env python3
"""Index all five pinned source pools without logging source text."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.contamination import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_index(
        REPO_ROOT / "configs/sources.yaml", REPO_ROOT / "data/heldout/manifest.yaml", args.output
    )
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))
