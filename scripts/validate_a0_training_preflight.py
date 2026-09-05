#!/usr/bin/env python3
"""Run one preregistered phase of the real-A0 GPU preflight."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.preflight import run_preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/a0_training_preflight.yaml")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["continuous", "resume", "padded"], required=True)
    for name in ("lm", "cuda", "build"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.config, args.run_root, args.phase, args.lm, args.cuda, args.build)
    print(json.dumps({key: result[key] for key in ("status", "stage", "failures")}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
