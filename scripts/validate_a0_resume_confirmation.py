#!/usr/bin/env python3
"""Run one immutable ADR-020 confirmation phase in a fresh process."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.resume_confirmation import run_confirmation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/a0_resume_confirmation.yaml")
    parser.add_argument("--phase", choices=("reference", "fixed", "native"), required=True)
    for name in ("run-root", "lm", "cuda", "build"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = run_confirmation(
        args.config, args.run_root, args.phase, args.lm, args.cuda, args.build
    )
    print(
        json.dumps(
            {"phase": args.phase, "status": report["status"], "failures": report["failures"]}
        )
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)
