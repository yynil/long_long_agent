#!/usr/bin/env python3
"""Thin entry for shape-dependent BF16 numeric diagnosis, not G0/G1 acceptance."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.numeric_diagnostics import diagnose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["smoke", "local_v0"], required=True)
    parser.add_argument("--lm", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(
        args.role,
        args.lm.resolve(),
        args.cuda.resolve(),
        args.build.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {k: result[k] for k in ("status", "matched_shape_comparisons", "kernel")}, indent=2
        )
    )
