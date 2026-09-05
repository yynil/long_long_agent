#!/usr/bin/env python3
"""Compare precision and matrix shapes, without changing model deployment math."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.precision_diagnostics import (
    diagnose_precision,
    diagnose_recurrent_precision,
    diagnose_whole_model_precision,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["smoke", "local_v0"], required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--whole-model", action="store_true")
    mode.add_argument("--recurrent-precision", action="store_true")
    for name in ("lm", "cuda", "build", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    action = diagnose_whole_model_precision if args.whole_model else diagnose_precision
    if args.recurrent_precision:
        action = diagnose_recurrent_precision
    action(
        args.role,
        args.lm.resolve(),
        args.cuda.resolve(),
        args.build.resolve(),
        args.output.resolve(),
    )
