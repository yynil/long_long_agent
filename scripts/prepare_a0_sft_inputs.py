#!/usr/bin/env python3
"""Build or independently verify complete admitted-A0 success-only SFT inputs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.preflight import write_json_once
from src.training.sft_inputs import build_inputs, verify_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--config", type=Path, default=ROOT / "configs/a0_sft_inputs.yaml")
    verifier = commands.add_parser("verify")
    verifier.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        report = build_inputs(args.config)
    else:
        report = verify_inputs(args.root)
        write_json_once(args.root / "verification.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in (
                    "status",
                    "accepted",
                    "rejected",
                    "excluded",
                    "exception_type",
                    "total_decisions_accounted",
                )
            }
        )
    )
    raise SystemExit(0 if report["status"] in ("built", "passed") else 1)
