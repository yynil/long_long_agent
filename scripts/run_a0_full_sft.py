#!/usr/bin/env python3
"""Thin entry for one complete A0 SFT epoch and dev validation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.runtime import ROOT
from src.training.full_sft import run_sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/a0_full_sft.yaml")
    for name in ("lm", "cuda", "build"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run_sft(args.config, lm=args.lm, cuda=args.cuda, build=args.build)
    print(json.dumps({k: result[k] for k in ("status", "stage", "failures")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
