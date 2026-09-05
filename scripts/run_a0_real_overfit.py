#!/usr/bin/env python3
"""Run one independently gated real-A0 capacity or 32/128 overfit phase."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.real_overfit import PHASES, run_overfit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/a0_real_overfit.yaml")
    parser.add_argument("--phase", choices=PHASES, required=True)
    for name in ("run-root", "lm", "cuda", "build"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_overfit(args.config, args.run_root, args.phase, args.lm, args.cuda, args.build)
    print(
        json.dumps(
            {"phase": args.phase, "status": result["status"], "failures": result["failures"]}
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
