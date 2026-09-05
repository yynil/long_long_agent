import copy
import json
from pathlib import Path

import jsonschema
import pytest
import torch
import yaml

from src.training.episode_collator import PackedEpisodeCollator
from src.training.packing import CausalSequence
from src.training.real_overfit import (
    evaluate,
    load_overfit_config,
    overfit_failures,
    summarize_losses,
    validate_manifest,
)
from src.training.sft_trainer import PackedAgentSFTTrainer
from src.training.token_budget_sampler import TokenBudgetPackSampler

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_overfit_config(ROOT / "configs/a0_real_overfit.yaml")[0]


def test_real_overfit_config_rejects_scope_and_threshold_changes(tmp_path):
    original = config()
    for changes in (
        {"unexpected": 0},
        {"epochs": 1},
        {"base_config_sha256": "a" * 64},
        {"capacity_sample_index": 1},
    ):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**original, **changes}))
        with pytest.raises((ValueError, jsonschema.ValidationError)):
            load_overfit_config(path)


def test_evaluation_uses_weighted_denominators_and_retains_loss_quantiles():
    rows = [
        {"sample_ids": ["a"], "loss": 2.0, "weight_sum": 1.0, "loss_tokens": 1},
        {"sample_ids": ["b", "c"], "loss": 4.0, "weight_sum": 3.0, "loss_tokens": 2},
    ]
    result = summarize_losses(rows)
    assert result["weighted_loss"] == 3.5
    assert (result["weight_sum"], result["sequences"], result["loss_tokens"]) == (4, 3, 3)
    assert result["row_loss_quantiles"]["1"] == 4
    for bad in ([], [{**rows[0], "loss": float("nan")}], [{**rows[0], "weight_sum": 0}]):
        with pytest.raises(ValueError):
            summarize_losses(bad)


def test_overfit_gate_requires_all_epochs_presentations_final_loss_and_stability():
    cfg = config()
    assert overfit_failures(2.0, [1.0] * 8, 32, 256, cfg) == []
    assert "incomplete_coverage" in overfit_failures(2.0, [0.5], 32, 32, cfg)
    assert "incomplete_coverage" in overfit_failures(2.0, [0.5] * 8, 32, 255, cfg)
    assert "final_loss_ratio" in overfit_failures(2.0, [1.01] * 8, 32, 256, cfg)
    assert "epoch_loss_spike" in overfit_failures(2.0, [0.5, 1.0, *[0.5] * 6], 32, 256, cfg)
    assert overfit_failures(2.0, [float("nan")] * 8, 32, 256, cfg) == ["nonfinite_or_invalid_loss"]
    assert "incomplete_coverage" in overfit_failures(2.0, [], 32, 0, cfg)


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)

    def _forward_features(self, input_ids, starts):
        return self.emb(input_ids)


def test_cpu_evaluation_covers_all_rows_without_mutating_training_counters():
    torch.manual_seed(1)
    network = TinyNetwork()
    trainer = PackedAgentSFTTrainer(network, torch.optim.AdamW(network.parameters(), lr=0.05))
    samples = [CausalSequence(str(i), (1, 2, 3), (2, 3, 4), (0.0, 1.0, 2.0)) for i in range(4)]
    sampler = TokenBudgetPackSampler(
        [3] * 4, [s.sample_id for s in samples], max_tokens=16, shuffle=False
    )
    collator = PackedEpisodeCollator(object(), max_pack_tokens=16)
    before = copy.deepcopy(trainer.state_dict())
    initial = evaluate(trainer, collator, samples, sampler, device="cpu")
    assert trainer.state_dict() == before
    assert initial["sequences"] == 4
    for _ in range(8):
        for indices in sampler:
            trainer.train_step(collator.collate_tokenized([samples[i] for i in indices]))
    final = evaluate(trainer, collator, samples, sampler, device="cpu")
    assert final["weighted_loss"] < initial["weighted_loss"]
    assert trainer.progress.sequences == 32


def test_real_overfit_manifest_offline_schema_roundtrip():
    # Reuse a synthetic official-runtime envelope assembled here, not local artifacts.
    from src.training.checkpoint import PROVENANCE_KEYS

    digest = "a" * 64
    flags = {"fp32_precision": "ieee", "allow_bf16_reduced_precision_reduction": True}
    cfg, base = load_overfit_config(ROOT / "configs/a0_real_overfit.yaml")
    manifest = {
        "schema_version": 1,
        "purpose": cfg["purpose"],
        "phase": "capacity",
        "code_commit": "b" * 40,
        "dirty_diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "provenance": dict.fromkeys(PROVENANCE_KEYS, digest),
        "input_plan_sha256": digest,
        "command": ["overfit"],
        "resolved_config": cfg,
        "base_config": base,
        "sample_count": 1,
        "sample_ids_sha256": digest,
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
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest({**manifest, "unexpected": 0})
