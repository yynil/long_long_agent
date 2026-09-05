#!/usr/bin/env python3
"""Audit A0 candidates or build an immutable release from passed admission."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.a0_release import audit_candidates, build_release, verify_release

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "build", "verify"])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/a0_release.yaml")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        if args.release is None:
            parser.error("verify requires --release")
        print(json.dumps(verify_release(args.release), indent=2))
    else:
        if args.index is None or args.admission is None:
            parser.error("audit/build require --index and --admission")
        if args.mode == "audit":
            result = audit_candidates(args.config, args.index, args.admission)
            print(
                json.dumps(
                    {"status": result["status"], "statistics": result["statistics"]}, indent=2
                )
            )
            if result["status"] != "passed":
                raise SystemExit(2)
        else:
            print(build_release(args.config, args.admission, args.index))
