#!/usr/bin/env python3
"""Repeat the same real-A0 backward three times without an optimizer update."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.resume_diagnostics import repeat_backward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-root", "output", "lm", "cuda", "build"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = repeat_backward(args.reference_root, args.output, args.lm, args.cuda, args.build)
    print(json.dumps({"status": report["status"]}))
    raise SystemExit(0 if report["status"] == "diagnostic_complete" else 1)
