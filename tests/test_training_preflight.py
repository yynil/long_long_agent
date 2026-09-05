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
    optimizer_fingerprints,
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


def test_manifest_validates_before_and_after_json_and_rejects_tuple_packages():
    from src.training.checkpoint import PROVENANCE_KEYS

    digest = "a" * 64
    flags = {"fp32_precision": "ieee", "allow_bf16_reduced_precision_reduction": True}
    manifest = {
        "schema_version": 1,
        "purpose": "real_a0_full_parameter_preflight",
        "phase": "continuous",
        "code_commit": "b" * 40,
        "dirty_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "input_plan_sha256": digest,
        "command": ["preflight"],
        "resolved_config": load_config(ROOT / "configs/a0_training_preflight.yaml"),
        "provenance": dict.fromkeys(PROVENANCE_KEYS, digest),
        "environment": {
            "lock_sha256": digest,
            "packages": [["torch", "2.11.0"]],
            "python": "3.11.15",
            "torch": "2.11.0",
            "cuda": "13.0",
            "device": "test-device",
            "capability": [8, 6],
            "numerical_flags": flags,
        },
        "runtime": {
            "checkpoint_sha256": digest,
            "tokenizer_sha256": digest,
            "model_code_sha256": digest,
            "torch": "2.11.0",
            "cuda": "13.0",
            "device": "test-device",
            "matmul_precision": flags,
            "short_inference_bf16_matmul_rows": "native; alignment is diagnostic-only",
            "upstream_trees": [{"revision": "b" * 40, "patched_files": {"model.py": digest}}] * 2,
            "runtime_environment": {
                "RWKV_JIT_ON": "0",
                "RWKV_HEAD_SIZE": "64",
                "RWKV_MY_TESTING": "x070",
                "RWKV_KERNEL": "",
                "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
                "TORCH_CUDA_ARCH_LIST": "8.6",
            },
        },
    }
    validate_manifest(manifest)
    validate_manifest(json.loads(json.dumps(manifest)))
    bad = copy.deepcopy(manifest)
    bad["environment"]["packages"] = [("torch", "2.11.0")]
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["resolved_config"]["unexpected"] = True
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(bad)


def test_optimizer_diagnostics_distinguish_master_from_moments():
    from src.training.parameter_groups import FP32MasterAdamW, OptimizerGroup, OptimizerPlan

    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    group = OptimizerGroup("base", ("weight",), (parameter,), 1.0, 0.0)
    optimizer = FP32MasterAdamW(OptimizerPlan((), (group,)), learning_rate=0.001)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = optimizer_fingerprints(optimizer)
    state = optimizer.state_dict()
    state["master_weights"][0].add_(1e-6)
    optimizer.load_state_dict(state)
    after = optimizer_fingerprints(optimizer)
    assert before["weight"]["master"] != after["weight"]["master"]
    assert before["weight"]["moments"] == after["weight"]["moments"]
