import copy
import json
from pathlib import Path

import jsonschema
import pytest
import torch
import yaml

from src.training.episode_collator import PackedEpisodeCollator
from src.training.packing import CausalSequence
from src.training.preflight import (
    load_config,
    padded_baseline,
    tree_digest,
    validate_manifest,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[1]


def test_preflight_config_is_closed_and_frozen(tmp_path):
    config = load_config(ROOT / "configs/a0_training_preflight.yaml")
    for key, value in (
        ("unexpected", 1),
        ("gradient_checkpointing", True),
        ("sample_count", 128),
        ("learning_rate", float("nan")),
    ):
        changed = {**config, key: value}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            load_config(path)


def test_fingerprint_preserves_bf16_bits_shapes_and_nested_types():
    value = {
        "weights": torch.arange(12).to(torch.bfloat16).reshape(3, 4),
        "step": torch.tensor(2.0),
    }
    assert tree_digest(value) == tree_digest(copy.deepcopy(value))
    assert tree_digest(value) == tree_digest(dict(reversed(list(value.items()))))
    changed = copy.deepcopy(value)
    changed["weights"][0, 0] = 1
    assert tree_digest(value) != tree_digest(changed)
    changed = {**value, "weights": value["weights"].reshape(4, 3)}
    assert tree_digest(value) != tree_digest(changed)
    assert tree_digest([1, 2]) != tree_digest((1, 2))
    assert tree_digest(torch.tensor([0.0])) != tree_digest(torch.tensor([-0.0]))


def test_padded_baseline_keeps_real_tokens_targets_and_reset_isolation():
    seq = CausalSequence("a", (1, 2, 3), (2, 3, 4), (0.0, 1.0, 2.0))
    batch = PackedEpisodeCollator(object(), max_pack_tokens=64).collate_tokenized([seq])
    old = tree_digest(batch)
    padded = padded_baseline(batch, 64)
    assert tree_digest(batch) == old
    for key in ("input_ids", "targets", "loss_weights", "valid_token_mask"):
        assert torch.equal(padded[key][:, :3], batch[key][:, :3])
    assert padded["sequence_start_mask"].nonzero().tolist() == [[0, 0], [0, 3]]
    assert padded["loss_weights"][:, 3:].sum() == 0
    assert (padded["targets"][:, 3:] == -100).all()
    with pytest.raises(ValueError, match="truncate"):
        padded_baseline(batch, 1)


def test_padded_baseline_adds_reset_for_previously_unpadded_input():
    seq = CausalSequence("a", tuple(range(16)), tuple(range(16)), (1.0,) * 16)
    batch = PackedEpisodeCollator(object(), max_pack_tokens=32).collate_tokenized([seq])
    assert padded_baseline(batch, 32)["sequence_start_mask"].nonzero().tolist() == [[0, 0], [0, 16]]


def test_preflight_json_is_immutable_and_manifest_rejects_missing_fields(tmp_path):
    path = tmp_path / "result.json"
    write_json_once(path, {"schema_version": 1, "status": "failed"})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"schema_version": 1, "status": "passed"})
    assert json.loads(path.read_text())["status"] == "failed"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest({"schema_version": 1})
