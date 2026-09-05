#!/usr/bin/env python3
"""Isolate exact fixed-gradient restoration without passing native acceptance."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.fixed_gradient_diagnostic import run_diagnostic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/fixed_gradient_resume_diagnostic.yaml"
    )
    parser.add_argument("--phase", choices=("capture", "replay"), required=True)
    for name in ("run-root", "lm", "cuda", "build"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = run_diagnostic(args.config, args.run_root, args.phase, args.lm, args.cuda, args.build)
    print(
        json.dumps(
            {"phase": args.phase, "status": report["status"], "failures": report["failures"]}
        )
    )
    raise SystemExit(0 if report["status"] == "diagnostic_complete" else 1)
