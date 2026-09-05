#!/usr/bin/env python3
"""Thin entry for isolated real-task verifier plumbing."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.fixture_environment import verify_fixture

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/environment_fixture.yaml"))
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_fixture(
        args.config.resolve(), args.rootfs.resolve(), args.harness.resolve(), args.output.resolve()
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
