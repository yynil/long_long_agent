"""Immutable single-process checkpoints including FP32 master state and RNG."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from src.data.governance import sha256_file
from src.data.util import canonical_json

PROVENANCE_KEYS = {
    "checkpoint_sha256",
    "model_code_sha256",
    "tokenizer_sha256",
    "data_release_sha256",
    "config_sha256",
    "environment_sha256",
}


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def _plan_hash(sampler) -> str:
    return hashlib.sha256(canonical_json(asdict(sampler.plan())).encode()).hexdigest()


def _validate_provenance(provenance: dict) -> None:
    import re

    if set(provenance) != PROVENANCE_KEYS or any(
        not re.fullmatch("[a-f0-9]{64}", value) for value in provenance.values()
    ):
        raise ValueError("checkpoint provenance requires all six SHA-256 components")


def save_checkpoint(
    path: Path, trainer, sampler, *, next_row: int, provenance: dict, scheduler=None
) -> str:
    _validate_provenance(provenance)
    if not 0 <= next_row <= len(sampler):
        raise ValueError("checkpoint next_row outside sampler plan")
    if path.exists():
        raise FileExistsError("checkpoint is immutable")
    import numpy as np

    numpy_state = np.random.get_state()
    payload = {
        "schema_version": 1,
        "provenance": provenance,
        "model": _cpu(trainer.network.state_dict()),
        "optimizer": _cpu(trainer.optimizer.state_dict()),
        "trainer": trainer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "sampler": {
            "epoch": sampler.epoch,
            "next_row": next_row,
            "plan_sha256": _plan_hash(sampler),
            "world_size": sampler.world_size,
            "rank": sampler.rank,
        },
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # no replacement even if another writer won the race
    finally:
        Path(temporary).unlink(missing_ok=True)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps({"schema_version": 1, "sha256": digest, "provenance": provenance}, indent=2)
        + "\n"
    )
    return digest


def load_checkpoint(path: Path, trainer, sampler, *, provenance: dict, scheduler=None) -> int:
    _validate_provenance(provenance)
    descriptor = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
    if (
        set(descriptor) != {"schema_version", "sha256", "provenance"}
        or descriptor["schema_version"] != 1
    ):
        raise ValueError("unknown checkpoint manifest")
    if descriptor["provenance"] != provenance or sha256_file(path) != descriptor["sha256"]:
        raise ValueError("checkpoint hash or provenance mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(payload)
        != {
            "schema_version",
            "provenance",
            "model",
            "optimizer",
            "trainer",
            "scheduler",
            "sampler",
            "rng",
        }
        or payload["schema_version"] != 1
    ):
        raise ValueError("unknown checkpoint payload")
    if payload["provenance"] != provenance:
        raise ValueError("checkpoint payload provenance mismatch")
    cursor = payload["sampler"]
    if set(cursor) != {"epoch", "next_row", "plan_sha256", "world_size", "rank"}:
        raise ValueError("unknown sampler checkpoint fields")
    if cursor["world_size"] != sampler.world_size or cursor["rank"] != sampler.rank:
        raise ValueError("checkpoint distributed ownership mismatch")
    sampler.set_epoch(cursor["epoch"])
    if _plan_hash(sampler) != cursor["plan_sha256"] or not 0 <= cursor["next_row"] <= len(sampler):
        raise ValueError("checkpoint sampler identity mismatch")
    current = trainer.network.state_dict()
    if set(payload["model"]) != set(current) or any(
        saved.shape != current[name].shape or saved.dtype != current[name].dtype
        for name, saved in payload["model"].items()
    ):
        raise ValueError("checkpoint model specification mismatch")
    if (payload["scheduler"] is None) != (scheduler is None):
        raise ValueError("checkpoint scheduler mismatch")
    rng = payload["rng"]
    if len(rng["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("checkpoint CUDA RNG device count mismatch")
    trainer.network.load_state_dict(payload["model"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.load_state_dict(payload["trainer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    import numpy as np

    numpy_state = rng["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return cursor["next_row"]
