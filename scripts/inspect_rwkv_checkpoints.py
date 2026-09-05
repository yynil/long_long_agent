#!/usr/bin/env python3
"""Load pinned RWKV checkpoints and verify their structural metadata."""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}")
    return value


def inspect_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise TypeError(f"Expected non-empty state dict in {path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"Checkpoint contains non-tensor values: {path}")

    required = ("emb.weight", "blocks.0.att.r_k", "ln_out.weight", "head.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    block_ids = {int(match.group(1)) for key in state if (match := BLOCK_RE.match(key)) is not None}
    if block_ids != set(range(max(block_ids) + 1)):
        raise ValueError(f"Non-contiguous block IDs in {path}")

    emb = state["emb.weight"]
    head = state["head.weight"]
    r_k = state["blocks.0.att.r_k"]
    actual_architecture = {
        "layers": len(block_ids),
        "embedding_dim": emb.shape[1],
        "vocab_size": emb.shape[0],
        "head_size": r_k.shape[1],
        "attention_heads": r_k.shape[0],
        "deep_embed": any("emb" in key.lower() and key != "emb.weight" for key in state),
        "decay_lora_rank": state["blocks.0.att.w1"].shape[1],
        "aaa_lora_rank": state["blocks.0.att.a1"].shape[1],
        "mv_lora_rank": state["blocks.0.att.v1"].shape[1],
        "gate_lora_rank": state["blocks.0.att.g1"].shape[1],
    }
    expected_architecture = {
        **expected["architecture"],
        "vocab_size": 65536,
    }
    checked_fields = (
        "layers",
        "embedding_dim",
        "vocab_size",
        "head_size",
        "deep_embed",
        "decay_lora_rank",
        "aaa_lora_rank",
        "mv_lora_rank",
        "gate_lora_rank",
    )
    for field in checked_fields:
        if actual_architecture[field] != expected_architecture[field]:
            raise ValueError(
                f"Architecture mismatch for {path.name} field {field}: "
                f"{actual_architecture[field]} != {expected_architecture[field]}"
            )
    if tuple(head.shape) != tuple(emb.shape):
        raise ValueError(f"Embedding/head shape mismatch in {path}")
    rank_keys = {
        "w": "decay_lora_rank",
        "a": "aaa_lora_rank",
        "v": "mv_lora_rank",
        "g": "gate_lora_rank",
    }
    for block_id in block_ids:
        for key_prefix, rank_field in rank_keys.items():
            left = state[f"blocks.{block_id}.att.{key_prefix}1"]
            right = state[f"blocks.{block_id}.att.{key_prefix}2"]
            rank = actual_architecture[rank_field]
            if tuple(left.shape) != (emb.shape[1], rank) or tuple(right.shape) != (
                rank,
                emb.shape[1],
            ):
                raise ValueError(f"Inconsistent {rank_field} shapes in block {block_id} of {path}")

    dtype_tensors = Counter(str(tensor.dtype) for tensor in state.values())
    dtype_elements = Counter()
    for tensor in state.values():
        dtype_elements[str(tensor.dtype)] += tensor.numel()
    result = {
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "state_dict_keys": len(state),
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "tensor_storage_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in state.values()
        ),
        "dtype_tensor_counts": dict(sorted(dtype_tensors.items())),
        "dtype_element_counts": dict(sorted(dtype_elements.items())),
        "architecture": actual_architecture,
        "required_key_check": "passed",
    }
    del state
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs/base_model.yaml")
    parser.add_argument("--storage-config", type=Path, default=REPO_ROOT / "configs/storage.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_yaml(args.model_config.resolve())
    storage = load_yaml(args.storage_config.resolve())["storage"]
    repo = model_config["model_repo"]
    model_root = (
        Path(storage["local_root"])
        / storage["models_dir"]
        / repo["repo_id"].replace("/", "__").lower()
        / repo["revision"]
    ).resolve()
    manifest_path = model_root / "download_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Verified model download manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = {item["path"]: item for item in manifest["files"]}

    checkpoints = []
    for role in model_config["initial_download_roles"]:
        expected = model_config["checkpoints"][role]
        item = manifest_files.get(expected["file"])
        if item is None or item.get("local_sha256") != expected["sha256"]:
            raise ValueError(f"Missing verified SHA-256 for checkpoint role {role}")
        result = inspect_checkpoint(model_root / expected["file"], expected)
        result["role"] = role
        result["sha256"] = item["local_sha256"]
        checkpoints.append(result)

    inventory = {
        "schema_version": 1,
        "inspected_at": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "model_repo": repo["repo_id"],
        "model_revision": repo["revision"],
        "checkpoints": checkpoints,
    }
    output = model_root / "checkpoint_inventory.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    for item in checkpoints:
        print(
            f"{item['role']}: {item['parameter_count']:,} params, "
            f"L{item['architecture']['layers']} D{item['architecture']['embedding_dim']} "
            f"H{item['architecture']['attention_heads']}x{item['architecture']['head_size']}"
        )
    print(f"verified checkpoint structures -> {output}")


if __name__ == "__main__":
    main()
