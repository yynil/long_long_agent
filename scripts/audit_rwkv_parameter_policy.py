#!/usr/bin/env python3
"""Audit exact checkpoint names against the official RWKV-7 optimizer policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.latent_v0 import RWKV7V0LatentControl
from src.model.readout import RWKV7ValueReadout
from src.training.parameter_groups import build_rwkv7_optimizer_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    control = RWKV7V0LatentControl(args.n_embd, args.max_depth)
    value_readout = RWKV7ValueReadout(args.n_embd)
    named_parameters = [(f"network.{name}", tensor) for name, tensor in checkpoint.items()]
    named_parameters.extend(
        (f"latent_control.{name}", parameter) for name, parameter in control.named_parameters()
    )
    named_parameters.extend(
        (f"value_readout.{name}", parameter) for name, parameter in value_readout.named_parameters()
    )
    plan = build_rwkv7_optimizer_plan(
        named_parameters,
        n_layer=args.n_layer,
        weight_decay=args.weight_decay,
    )
    counts = Counter(assignment.group_name for assignment in plan.assignments)
    elements = Counter()
    for assignment in plan.assignments:
        elements[assignment.group_name] += assignment.numel
    result = {
        "checkpoint": args.checkpoint.name,
        "covered_parameter_count": plan.covered_parameter_count,
        "covered_numel": plan.covered_numel,
        "group_parameter_counts": dict(sorted(counts.items())),
        "group_numel": dict(sorted(elements.items())),
        "trainable_numel": plan.trainable_numel,
        "optimizer_trainable_groups": [group.name for group in plan.groups],
        "status": "passed",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
