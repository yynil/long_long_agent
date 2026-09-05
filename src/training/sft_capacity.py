"""Worst supervision packed-row capacity before a complete A0 SFT job."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import jsonschema
import numpy as np
import torch
import yaml

from src.data.governance import sha256_file
from src.model.runtime import ROOT, load_runtime

from .checkpoint import save_checkpoint
from .episode_collator import PackedEpisodeCollator
from .preflight import (
    load_config,
    make_trainer,
    padded_baseline,
    timed_step,
    tree_digest,
    write_json_once,
)
from .real_overfit import validate_manifest as validate_overfit_manifest
from .sft_inputs import load_split
from .token_budget_sampler import TokenBudgetPackSampler
from .tokenizer import RWKVByteTokenizer


def capacity_plan(samples, *, seed: int, max_tokens: int, alignment: int):
    ids = [s.sample_id for s in samples]
    if not samples or len(ids) != len(set(ids)):
        raise ValueError("capacity needs nonempty unique training data")
    sampler = TokenBudgetPackSampler(
        [len(s.token_ids) - 1 for s in samples],
        ids,
        max_tokens=max_tokens,
        alignment=alignment,
        seed=seed,
        shuffle=True,
    )
    rows = list(sampler)
    if sorted(i for row in rows for i in row) != list(range(len(samples))):
        raise ValueError("complete SFT epoch plan loses or duplicates samples")
    loss_counts = [sum(w > 0 for w in s.token_loss_weights[1:]) for s in samples]
    scores = [sum(loss_counts[i] for i in row) for row in rows]
    worst = max(range(len(rows)), key=lambda i: scores[i])
    plan = sampler.plan()
    return rows[worst], {
        "schema_version": 1,
        "seed": seed,
        "epoch": 0,
        "sample_count": len(samples),
        "pack_rows": len(rows),
        "worst_row_index": worst,
        "worst_loss_tokens": scores[worst],
        "worst_sample_ids": [ids[i] for i in rows[worst]],
        "worst_real_tokens": sum(len(samples[i].token_ids) - 1 for i in rows[worst]),
        "max_individual_loss_tokens": max(loss_counts),
        "total_loss_tokens": sum(loss_counts),
        "total_real_tokens": sum(r.real_tokens for r in plan.rank_rows),
        "total_aligned_tokens": sum(r.aligned_tokens for r in plan.rank_rows),
        "plan_sha256": tree_digest(asdict(plan)),
    }


def load_capacity_config(path: Path):
    config = yaml.safe_load(path.read_text())
    jsonschema.validate(
        config, json.loads((ROOT / "schemas/a0_sft_capacity.schema.json").read_text())
    )
    if sha256_file(ROOT / config["base_config"]) != config["base_config_sha256"]:
        raise ValueError("capacity base config changed")
    return config, load_config(ROOT / config["base_config"])


def run_capacity(config_path: Path, *, lm: Path, cuda: Path, build: Path):
    config, base = load_capacity_config(config_path)
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit capacity protocol before running")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prior_path = data_root / config["overfit128_result"]
    if sha256_file(prior_path) != config["overfit128_result_sha256"]:
        raise ValueError("overfit evidence changed")
    prior = json.loads(prior_path.read_text())
    prior_manifest = json.loads((prior_path.parent / "manifest.json").read_text())
    validate_overfit_manifest(prior_manifest)
    if (
        prior["status"] != "passed"
        or prior["phase"] != "overfit128"
        or sha256_file(prior_path.parent / "manifest.json") != prior["manifest_sha256"]
    ):
        raise ValueError("complete 128 overfit not passed")
    output = data_root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "intent.json",
        {
            "schema_version": 1,
            "code_commit": commit,
            "config_sha256": sha256_file(config_path),
            "command": sys.argv,
        },
    )
    report = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "status": "failed",
        "stage": "inputs",
        "steps": [],
        "failures": [],
        "checkpoints": [],
    }
    trainer = sampler = provenance = None
    cursor = 0
    try:
        torch.set_num_threads(config["cpu_threads"])
        samples = load_split(
            data_root / config["input_directory"],
            "train",
            expected_manifest_sha256=config["input_manifest_sha256"],
        )
        indices, plan = capacity_plan(
            samples,
            seed=config["training_seed"],
            max_tokens=base["max_tokens"],
            alignment=base["alignment"],
        )
        report["plan"] = plan
        write_json_once(output / "full_train_plan.json", plan)
        print(json.dumps({"stage": "full_train_plan", **plan}), flush=True)
        selected = [samples[i] for i in indices]
        del samples
        sampler = TokenBudgetPackSampler(
            [len(s.token_ids) - 1 for s in selected],
            [s.sample_id for s in selected],
            max_tokens=base["max_tokens"],
            alignment=base["alignment"],
            seed=config["training_seed"],
            shuffle=False,
        )
        random.seed(config["training_seed"])
        np.random.seed(config["training_seed"])
        torch.manual_seed(config["training_seed"])
        report["stage"] = "runtime"
        network, _, runtime = load_runtime(
            base["model_role"], lm.resolve(), cuda.resolve(), build.resolve()
        )
        environment = {
            "lock_sha256": sha256_file(ROOT / base["environment_lock"]),
            "packages": sorted(
                [d.metadata["Name"], d.version] for d in importlib.metadata.distributions()
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": runtime["device"],
            "capability": list(torch.cuda.get_device_capability()),
            "numerical_flags": runtime["matmul_precision"],
        }
        if runtime != prior_manifest["runtime"] or environment != prior_manifest["environment"]:
            raise ValueError("capacity runtime differs from accepted overfit")
        provenance = {
            "checkpoint_sha256": runtime["checkpoint_sha256"],
            "model_code_sha256": runtime["model_code_sha256"],
            "tokenizer_sha256": runtime["tokenizer_sha256"],
            "data_release_sha256": base["release_manifest_sha256"],
            "config_sha256": sha256_file(config_path),
            "environment_sha256": tree_digest(environment),
        }
        manifest = {
            "schema_version": 1,
            "purpose": config["purpose"],
            "code_commit": commit,
            "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
            "provenance": provenance,
            "runtime": runtime,
            "environment": environment,
            "command": sys.argv,
            "resolved_config": config,
            "base_config": base,
            "plan": plan,
        }
        validate_manifest(manifest)
        write_json_once(output / "manifest.json", manifest)
        report["manifest_sha256"] = sha256_file(output / "manifest.json")
        trainer, report["parameters"] = make_trainer(network, base)
        training = yaml.safe_load((ROOT / "configs/training_data.yaml").read_text())
        vocab = ROOT / training["tokenizer"]["vocabulary"]
        if sha256_file(vocab) != runtime["tokenizer_sha256"]:
            raise ValueError("capacity tokenizer changed")
        collator = PackedEpisodeCollator(
            RWKVByteTokenizer(vocab), max_pack_tokens=base["max_tokens"], align_to=base["alignment"]
        )
        batch = padded_baseline(collator.collate_tokenized(selected), base["max_tokens"])
        for step in range(config["updates"]):
            sampler.set_epoch(step)
            cursor = 0
            report["stage"] = f"update_{step + 1}"
            entry = timed_step(trainer, batch, base)
            cursor = 1
            report["steps"].append(entry)
            write_json_once(output / f"update_{step + 1}.json", entry)
            print(json.dumps({"step": step + 1, **entry}), flush=True)
            if entry["failed_checks"]:
                report["failures"].extend(entry["failed_checks"])
                raise RuntimeError("complete-input capacity stop rule")
        trainer.optimizer.zero_grad(set_to_none=True)
        digest = save_checkpoint(
            output / "final.pt", trainer, sampler, next_row=cursor, provenance=provenance
        )
        report["checkpoints"].append({"file": "final.pt", "sha256": digest})
        report["status"], report["stage"] = "passed", "complete"
    except Exception as error:  # noqa: BLE001 -- type-only evidence, no source text
        report["failures"].append(type(error).__name__)
        report["exception_sites"] = [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(error.__traceback__)
        ]
        if trainer is not None and provenance is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            try:
                digest = save_checkpoint(
                    output / "stopped.pt", trainer, sampler, next_row=cursor, provenance=provenance
                )
                report["checkpoints"].append({"file": "stopped.pt", "sha256": digest})
            except Exception as save_error:  # noqa: BLE001 -- preserve save failure
                report["checkpoint_save_failure"] = type(save_error).__name__
    finally:
        if trainer is not None:
            report["progress"] = asdict(trainer.progress)
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json_once(output / "result.json", report)
    return report


def validate_manifest(manifest):
    from referencing import Registry, Resource

    registry = Registry()
    for name in ("training_preflight_manifest", "a0_training_preflight", "a0_sft_capacity"):
        registry = registry.with_resource(
            f"https://long-long-agent.local/{name}.schema.json",
            Resource.from_contents(json.loads((ROOT / f"schemas/{name}.schema.json").read_text())),
        )
    schema = json.loads((ROOT / "schemas/a0_sft_capacity_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, registry=registry).validate(manifest)
