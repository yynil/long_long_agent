#!/usr/bin/env python3
"""Run the independently preregistered ADR-019 engineering checks."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.generation_confirmation import run_confirmation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generation_confirmation.yaml"))
    parser.add_argument("--role", choices=["smoke", "local_v0"], required=True)
    for name in ("lm", "cuda", "build", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_confirmation(
        args.config.resolve(),
        args.role,
        args.lm.resolve(),
        args.cuda.resolve(),
        args.build.resolve(),
        args.output.resolve(),
    )
    print(result["status"], flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)
