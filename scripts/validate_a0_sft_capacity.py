#!/usr/bin/env python3
"""Thin entry for the full-A0 packed supervision capacity gate."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.runtime import ROOT
from src.training.sft_capacity import run_capacity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/a0_sft_capacity.yaml")
    for name in ("lm", "cuda", "build"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run_capacity(args.config, lm=args.lm, cuda=args.cuda, build=args.build)
    print(json.dumps({k: result[k] for k in ("status", "stage", "failures")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
