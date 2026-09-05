#!/usr/bin/env python3
"""Prepare, but never train, the real 32/128 A0 overfit input sets."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.governance import sha256_file
from src.training.a0_dataset import write_overfit_input_plan
from src.training.tokenizer import RWKVByteTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = yaml.safe_load((ROOT / "configs/training_data.yaml").read_text())
    vocabulary = ROOT / training["tokenizer"]["vocabulary"]
    if sha256_file(vocabulary) != training["tokenizer"]["sha256"]:
        raise ValueError("tokenizer content hash mismatch")
    tokenizer = RWKVByteTokenizer(vocabulary)
    result = write_overfit_input_plan(args.release.resolve(), tokenizer, args.output.resolve())
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("purpose", "training_executed", "rejected_before_128", "profiles")
            },
            indent=2,
        )
    )
