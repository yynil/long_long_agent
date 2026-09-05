import json
from dataclasses import asdict

import jsonschema
import pytest
import torch
import yaml

from src.data.governance import sha256_file
from src.model.runtime import ROOT
from src.training import full_sft
from src.training.episode_collator import TokenizedEpisode
from src.training.sft_trainer import PackedAgentSFTTrainer


def test_full_sft_config_rejects_scope_and_threshold_changes(tmp_path):
    cfg, _ = full_sft.load_sft_config(ROOT / "configs/a0_full_sft.yaml")
    for change in (
        {"epochs": 2},
        {"expected_train_samples": 128},
        {"maximum_dev_to_initial_loss_ratio": 2},
        {"unknown": 0},
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(dict(cfg, **change)))
        with pytest.raises(jsonschema.ValidationError):
            full_sft.load_sft_config(path)


def test_dev_gate_full_denominator_final_nonregression():
    cfg, _ = full_sft.load_sft_config(ROOT / "configs/a0_full_sft.yaml")
    initial = {
        "weighted_loss": 2.0,
        "sequences": 186,
        "rows": 186,
        "weight_sum": 1000,
        "loss_tokens": 600,
    }
    assert (
        full_sft.validation_failures(initial, dict(initial, weighted_loss=2.4), cfg, final=False)
        == []
    )
    assert full_sft.validation_failures(
        initial, dict(initial, weighted_loss=2.4), cfg, final=True
    ) == ["dev_loss_regression"]
    assert full_sft.validation_failures(initial, dict(initial, rows=185), cfg, final=False) == [
        "incomplete_dev_coverage"
    ]
    assert full_sft.validation_failures(
        initial, dict(initial, weighted_loss=float("nan")), cfg, final=True
    ) == ["invalid_dev_loss"]


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)

    def _forward_features(self, tokens, starts):
        return self.emb(tokens)


def test_complete_sft_loop_cpu_with_real_updates_evals_checkpoints(tmp_path, monkeypatch):
    cfg, base = full_sft.load_sft_config(ROOT / "configs/a0_full_sft.yaml")
    cfg.update(
        expected_train_samples=4,
        expected_dev_samples=2,
        checkpoint_every_rows=2,
        validation_every_rows=2,
    )
    base["max_tokens"] = 16
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/storage.yaml").write_text(
        yaml.safe_dump({"storage": {"local_root": str(tmp_path)}})
    )
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("synthetic")
    (tmp_path / "configs/training_data.yaml").write_text(
        yaml.safe_dump({"tokenizer": {"vocabulary": str(vocab)}})
    )
    (tmp_path / base["environment_lock"]).write_text("synthetic lock")

    def sample(i):
        return TokenizedEpisode(
            str(i), "", (0,) + (1, 2, 3) * 5, (0.0,) + (1.0,) * 15, ("assistant_final",) * 16, 0
        )

    samples = {"train": [sample(i) for i in range(4)], "dev": [sample(i) for i in range(4, 6)]}
    _, plan = full_sft.capacity_plan(
        samples["train"], seed=cfg["training_seed"], max_tokens=16, alignment=16
    )
    runtime = {
        "device": "synthetic",
        "matmul_precision": {},
        "checkpoint_sha256": "a" * 64,
        "model_code_sha256": "b" * 64,
        "tokenizer_sha256": sha256_file(vocab),
    }
    environment = {
        "lock_sha256": sha256_file(tmp_path / base["environment_lock"]),
        "packages": [],
        "python": full_sft.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": "synthetic",
        "capability": [8, 6],
        "numerical_flags": {},
    }
    prior_path = tmp_path / cfg["capacity_result"]
    prior_path.parent.mkdir(parents=True)
    prior_manifest = prior_path.parent / "manifest.json"
    prior_manifest.write_text(
        json.dumps(
            {
                "runtime": runtime,
                "environment": environment,
                "plan": plan,
                "resolved_config": {"input_manifest_sha256": cfg["input_manifest_sha256"]},
            }
        )
    )
    prior_path.write_text(
        json.dumps({"status": "passed", "manifest_sha256": sha256_file(prior_manifest)})
    )
    cfg["capacity_result_sha256"] = sha256_file(prior_path)
    monkeypatch.setattr(full_sft, "ROOT", tmp_path)
    monkeypatch.setattr(full_sft, "load_sft_config", lambda path: (cfg, base))
    monkeypatch.setattr(full_sft, "validate_capacity_manifest", lambda value: None)
    monkeypatch.setattr(full_sft, "validate_manifest", lambda value: None)
    monkeypatch.setattr(full_sft, "load_split", lambda root, split, **kwargs: samples[split])
    monkeypatch.setattr(full_sft, "RWKVByteTokenizer", lambda path: object())
    monkeypatch.setattr(
        full_sft.subprocess,
        "check_output",
        lambda args, **kwargs: b"" if "status" in args else "a" * 40,
    )
    monkeypatch.setattr(full_sft.importlib.metadata, "distributions", list)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 6))
    monkeypatch.setattr(full_sft, "load_runtime", lambda *args: (TinyNetwork(), None, runtime))

    def trainer(network, base):
        return PackedAgentSFTTrainer(network, torch.optim.AdamW(network.parameters(), lr=0.05)), {}

    monkeypatch.setattr(full_sft, "make_trainer", trainer)
    original_eval = full_sft.evaluate
    monkeypatch.setattr(full_sft, "evaluate", lambda *args: original_eval(*args, device="cpu"))

    def step(trainer, batch, base):
        return {
            "metrics": asdict(trainer.train_step(batch)),
            "elapsed_seconds": 1.0,
            "peak_allocated_bytes": 0,
            "failed_checks": [],
        }

    monkeypatch.setattr(full_sft, "timed_step", step)
    result = full_sft.run_sft(config_path, lm=tmp_path, cuda=tmp_path, build=tmp_path)
    assert result["status"] == "passed", result
    assert result["progress"]["sequences"] == result["progress"]["optimizer_steps"] == 4
    assert [v["completed_rows"] for v in result["validations"]] == [0, 2, 4]
    assert [c["file"] for c in result["checkpoints"]] == ["row_2.pt", "row_4.pt"]
    assert result["validations"][-1]["weighted_loss"] < result["validations"][0]["weighted_loss"]
    output = tmp_path / cfg["output_directory"]
    for checkpoint in result["checkpoints"]:
        assert sha256_file(output / checkpoint["file"]) == checkpoint["sha256"]
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 4
