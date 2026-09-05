import copy
import json
from pathlib import Path

import jsonschema
import pytest
import torch
import yaml

from src.data.governance import sha256_file
from src.training.parameter_groups import FP32MasterAdamW, OptimizerGroup, OptimizerPlan
from src.training.preflight import tree_digest
from src.training.resume_comparison import (
    budget_failures,
    capture_gradients,
    gradient_failures,
    install_gradients,
    optimizer_comparison,
    tensor_comparison,
)
from src.training.resume_confirmation import (
    checked_gradients,
    confirmation_sampler,
    load_protocol,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return load_protocol(ROOT / "configs/a0_resume_confirmation.yaml")[0]


def test_confirmation_sampler_uses_real_tuple_contract_and_checks_order():
    from types import SimpleNamespace

    config, base = load_protocol(ROOT / "configs/a0_resume_confirmation.yaml")
    samples = [
        SimpleNamespace(token_ids=(0,) * length, sample_id=str(i))
        for i, length in enumerate((7485, 6644))
    ]
    sampler = confirmation_sampler(samples, base, config)
    assert list(sampler) == [(0,), (1,)]
    assert [r.real_tokens for r in sampler.plan().rank_rows] == [7484, 6643]
    with pytest.raises(ValueError, match="order"):
        confirmation_sampler(list(reversed(samples)), base, config)


def optimizer(model):
    names, parameters = zip(*model.named_parameters())
    group = OptimizerGroup("base", names, parameters, 1.0, 0.0)
    return FP32MasterAdamW(OptimizerPlan((), (group,)), learning_rate=3e-6)


def test_protocol_rejects_unknown_nonfinite_changed_base_and_nonindependent_inputs(tmp_path):
    original = protocol()
    for key, value in (
        ("unexpected", 0),
        ("base_config_sha256", "a" * 64),
        ("sample_indices", [7, 18]),
    ):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**original, key: value}))
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            load_protocol(path)
    for value in (float("nan"), float("inf")):
        changed = copy.deepcopy(original)
        changed["acceptance"]["master"]["max_abs"] = value
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            load_protocol(path)


def test_comparison_full_denominators_exact_boundary_and_shape_dtype_presence():
    a = {"a": torch.tensor([1.0, 2.0]), "b": torch.zeros(4), "none": None}
    b = copy.deepcopy(a)
    b["a"][0] += 0.5
    result = tensor_comparison(a, b, denominator_floor=1e-30)
    assert (result["tensors"], result["elements"], result["different_elements"]) == (2, 6, 1)
    assert result["max_abs"] == 0.5
    assert result["relative_l2"] == pytest.approx(0.5 / 5**0.5)
    assert budget_failures(result, {k: result[k] for k in ("max_abs", "relative_l2")}) == []
    for bad in (
        {},
        {**a, "a": a["a"].double()},
        {**a, "a": a["a"].view(1, 2)},
        {**a, "none": torch.zeros(1)},
    ):
        with pytest.raises(ValueError):
            tensor_comparison(a, bad, denominator_floor=1e-30)
    with pytest.raises(FloatingPointError):
        tensor_comparison(
            {"a": torch.tensor([float("nan")])}, {"a": torch.ones(1)}, denominator_floor=1e-30
        )


def test_gradient_envelope_and_parameter_family_do_not_just_check_global_budget():
    acceptance = protocol()["acceptance"]
    report = {
        "max_abs": 2e-6,
        "relative_l2": 2e-4,
        "differences": [{"parameter": "blocks.0.att.x_r"}],
    }
    assert gradient_failures(report, acceptance, []) == []
    assert gradient_failures(report, acceptance, [{"max_abs": 0, "relative_l2": 0}]) == [
        "envelope_max_abs",
        "envelope_relative_l2",
    ]
    report["differences"][0]["parameter"] = "head.weight"
    assert "unexpected_parameter_family" in gradient_failures(report, acceptance, [])


def test_fixed_gradient_update_matches_after_restore_and_does_not_alias():
    torch.manual_seed(29)
    model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    opt = optimizer(model)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    gradients = {n: torch.full_like(p, 0.125) for n, p in model.named_parameters()}
    install_gradients(model, gradients)
    opt.step()
    expected = tree_digest(opt.state_dict())
    resumed = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    restored = optimizer(resumed)
    restored.load_state_dict(saved)
    install_gradients(resumed, gradients)
    assert tree_digest(capture_gradients(resumed)) == tree_digest(gradients)
    assert all(p.grad.data_ptr() != gradients[n].data_ptr() for n, p in resumed.named_parameters())
    restored.step()
    assert tree_digest(restored.state_dict()) == expected
    assert tree_digest(resumed.state_dict()) == tree_digest(model.state_dict())
    before = tree_digest(capture_gradients(resumed))
    for changed in (
        {},
        {**gradients, "bias": gradients["bias"].float()},
        {**gradients, "bias": torch.full_like(gradients["bias"], float("nan"))},
    ):
        with pytest.raises(ValueError):
            install_gradients(resumed, changed)
        assert tree_digest(capture_gradients(resumed)) == before


def test_optimizer_budgets_steps_dtype_hyperparameters_and_unknown_fields():
    model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    opt = optimizer(model)
    install_gradients(model, {n: torch.ones_like(p) for n, p in model.named_parameters()})
    opt.step()
    state = copy.deepcopy(opt.state_dict())
    acceptance = protocol()["acceptance"]
    assert not optimizer_comparison(state, state, acceptance)["failed_checks"]
    changed = copy.deepcopy(state)
    changed["master_weights"][0].add_(1e-4)
    assert "master:max_abs" in optimizer_comparison(state, changed, acceptance)["failed_checks"]
    for mutate in (
        lambda s: s.update(unexpected=0),
        lambda s: s["optimizer"]["state"][0]["step"].add_(1),
        lambda s: s["optimizer"]["state"][0].update(unexpected=0),
        lambda s: s["optimizer"]["param_groups"][0].update(lr=1),
        lambda s: s["master_weights"].__setitem__(0, s["master_weights"][0].bfloat16()),
    ):
        changed = copy.deepcopy(state)
        mutate(changed)
        with pytest.raises(ValueError):
            optimizer_comparison(state, changed, acceptance)


def test_gradient_artifact_checks_file_payload_and_batch_identity(tmp_path):
    path = tmp_path / "gradients.pt"
    gradients = {"weight": torch.ones(3, dtype=torch.bfloat16)}
    payload = {
        "schema_version": 1,
        "provenance": {"config": "a" * 64},
        "batch_sha256": "b" * 64,
        "gradient_sha256": tree_digest(gradients),
        "gradients": gradients,
    }
    torch.save(payload, path)
    digest = sha256_file(path)
    assert tree_digest(
        checked_gradients(path, digest, payload["provenance"], payload["batch_sha256"])
    ) == tree_digest(gradients)
    for file_hash, provenance, batch_hash in (
        ("c" * 64, payload["provenance"], payload["batch_sha256"]),
        (digest, {}, payload["batch_sha256"]),
        (digest, payload["provenance"], "d" * 64),
    ):
        with pytest.raises(ValueError):
            checked_gradients(path, file_hash, provenance, batch_hash)
    payload["unexpected"] = 0
    torch.save(payload, path)
    with pytest.raises(ValueError):
        checked_gradients(path, sha256_file(path), payload["provenance"], payload["batch_sha256"])


def test_lazy_moments_only_allowed_for_explicit_none_gradients():
    model = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    opt = optimizer(model)
    model.weight.grad = torch.ones_like(model.weight)
    opt.step()
    state = copy.deepcopy(opt.state_dict())
    with pytest.raises(ValueError, match="active.*missing"):
        optimizer_comparison(state, state, protocol()["acceptance"])
    report = optimizer_comparison(
        state, state, protocol()["acceptance"], inactive_parameters=("bias",)
    )
    assert report["parameters_without_moments"] == ["bias"]
    assert report["master"]["tensors"] == 2
    assert report["exp_avg"]["tensors"] == 1
    assert not report["failed_checks"]
    changed = copy.deepcopy(state)
    changed["optimizer"]["state"].clear()
    with pytest.raises(ValueError):
        optimizer_comparison(
            state, changed, protocol()["acceptance"], inactive_parameters=("bias",)
        )


def test_fixed_diagnostic_config_pins_failed_source_and_rejects_unknown_fields(tmp_path):
    from src.training.fixed_gradient_diagnostic import load_diagnostic_config

    config = load_diagnostic_config(ROOT / "configs/fixed_gradient_resume_diagnostic.yaml")
    for changes in (
        {"unexpected": 0},
        {"source_protocol_sha256": "a" * 64},
        {"source_directory": "artifacts/other"},
    ):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**config, **changes}))
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            load_diagnostic_config(path)
    manifest_schema = json.loads(
        (ROOT / "schemas/fixed_gradient_diagnostic_manifest.schema.json").read_text()
    )
    manifest = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "phase": "capture",
        "code_commit": "b" * 40,
        "command": ["diagnostic"],
        **{
            k: "a" * 64
            for k in (
                "config_sha256",
                "source_manifest_sha256",
                "source_result_sha256",
                "runtime_sha256",
                "environment_sha256",
            )
        },
    }
    jsonschema.validate(manifest, manifest_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**manifest, "unexpected": 1}, manifest_schema)


def test_confirmation_manifest_offline_refs_and_closed_schemas():
    # The historical checked-in aggregate provides no runtime payload: use a small
    # synthetic envelope with all official runtime fields, never local artifacts.
    digest = "a" * 64
    flags = {"fp32_precision": "ieee", "allow_bf16_reduced_precision_reduction": True}
    from src.training.checkpoint import PROVENANCE_KEYS

    config, base = load_protocol(ROOT / "configs/a0_resume_confirmation.yaml")
    manifest = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "phase": "reference",
        "code_commit": "b" * 40,
        "dirty_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "provenance": dict.fromkeys(PROVENANCE_KEYS, digest),
        "input_plan_sha256": digest,
        "command": ["confirmation"],
        "resolved_config": config,
        "base_config": base,
        "selected_samples": [
            {"input_plan_index": i, "sample_id": digest, "input_tokens": 7484} for i in (32, 33)
        ],
        "environment": {
            "lock_sha256": digest,
            "packages": [["torch", "2.11.0"]],
            "python": "3.11.15",
            "torch": "2.11.0",
            "cuda": "13.0",
            "device": "synthetic",
            "capability": [8, 6],
            "numerical_flags": flags,
        },
        "runtime": {
            "checkpoint_sha256": digest,
            "tokenizer_sha256": digest,
            "model_code_sha256": digest,
            "torch": "2.11.0",
            "cuda": "13.0",
            "device": "synthetic",
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
    for target in (manifest, manifest["runtime"], manifest["resolved_config"]):
        target["unexpected"] = 0
        with pytest.raises(jsonschema.ValidationError):
            validate_manifest(manifest)
        del target["unexpected"]
