#!/usr/bin/env python3
"""Thin entry to pinned generation parity and K=0 diagnostics."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.generation_parity import run_parity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generation_parity.yaml"))
    parser.add_argument("--role", required=True, choices=["smoke", "local_v0"])
    parser.add_argument("--rwkv-lm-worktree", type=Path, required=True)
    parser.add_argument("--rwkv-cuda-worktree", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_parity(
        args.config.resolve(),
        args.role,
        args.rwkv_lm_worktree.resolve(),
        args.rwkv_cuda_worktree.resolve(),
        args.build_root.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "failures", "elapsed_seconds", "peak_allocated_bytes")
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
