from dataclasses import asdict

import pytest

from src.training.episode_collator import TokenizedEpisode
from src.training.sft_capacity import capacity_plan
from src.training.token_budget_sampler import TokenBudgetPackSampler


def sample(number, length, loss_tokens):
    return TokenizedEpisode(
        str(number),
        "",
        (0,) + (1,) * length,
        (0.0,) * (length + 1 - loss_tokens) + (1.0,) * loss_tokens,
        ("user",) * (length + 1),
        0,
    )


def test_worst_actual_pack_accounts_for_multiple_samples():
    samples = [sample(1, 7, 7), sample(2, 7, 7), sample(3, 15, 10)]
    selected, summary = capacity_plan(samples, seed=20260909, max_tokens=16, alignment=16)
    assert sorted(selected) == [0, 1]
    assert summary["worst_loss_tokens"] == 14
    assert summary["max_individual_loss_tokens"] == 10
    assert summary["sample_count"] == 3
    assert summary["total_real_tokens"] == 29
    assert summary["total_aligned_tokens"] == 32
    assert capacity_plan(samples, seed=20260909, max_tokens=16, alignment=16) == (selected, summary)
    sampler = TokenBudgetPackSampler(
        [7, 7, 15], ["1", "2", "3"], seed=20260909, max_tokens=16, alignment=16, shuffle=True
    )
    assert len(asdict(sampler.plan())["rank_rows"]) == 2


def test_capacity_rejects_empty_duplicate_and_overlong():
    with pytest.raises(ValueError, match="nonempty"):
        capacity_plan([], seed=1, max_tokens=16, alignment=16)
    value = sample(1, 7, 2)
    with pytest.raises(ValueError, match="unique"):
        capacity_plan([value, value], seed=1, max_tokens=16, alignment=16)
    with pytest.raises(ValueError):
        capacity_plan([sample(1, 17, 2)], seed=1, max_tokens=16, alignment=16)


def test_capacity_config_closed_and_offline_manifest(tmp_path):
    import json

    import jsonschema
    import yaml

    from src.model.runtime import ROOT
    from src.training.sft_capacity import load_capacity_config, validate_manifest

    config, base = load_capacity_config(ROOT / "configs/a0_sft_capacity.yaml")
    assert config["updates"] == 2
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(dict(config, updates=3)))
    with pytest.raises(jsonschema.ValidationError):
        load_capacity_config(path)
    from src.training.checkpoint import PROVENANCE_KEYS

    digest = "a" * 64
    flags = {"fp32_precision": "ieee", "allow_bf16_reduced_precision_reduction": True}
    prior = {
        "code_commit": "b" * 40,
        "dirty_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "provenance": dict.fromkeys(PROVENANCE_KEYS, digest),
        "command": ["capacity"],
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
    _, plan = capacity_plan(
        [sample(1, 7, 2)], seed=config["training_seed"], max_tokens=16, alignment=16
    )
    plan["worst_sample_ids"] = ["a" * 64]
    value = {
        k: prior[k]
        for k in (
            "code_commit",
            "dirty_diff_sha256",
            "provenance",
            "runtime",
            "environment",
            "command",
        )
    }
    value.update(
        schema_version=1,
        purpose=config["purpose"],
        resolved_config=config,
        base_config=base,
        plan=plan,
    )
    validate_manifest(value)
    validate_manifest(json.loads(json.dumps(value)))
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest(dict(value, extra=1))
