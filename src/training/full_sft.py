"""One preregistered full A0 train epoch with complete held-out dev CE checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import sys
import time
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
from .preflight import load_config, make_trainer, timed_step, tree_digest, write_json_once
from .real_overfit import evaluate, summarize_losses
from .sft_capacity import capacity_plan
from .sft_capacity import validate_manifest as validate_capacity_manifest
from .sft_inputs import load_split
from .token_budget_sampler import TokenBudgetPackSampler
from .tokenizer import RWKVByteTokenizer


def load_sft_config(path: Path):
    config = yaml.safe_load(path.read_text())
    jsonschema.validate(config, json.loads((ROOT / "schemas/a0_full_sft.schema.json").read_text()))
    if sha256_file(ROOT / config["base_config"]) != config["base_config_sha256"]:
        raise ValueError("SFT base config changed")
    return config, load_config(ROOT / config["base_config"])


def validation_failures(initial: dict, current: dict, config: dict, *, final: bool):
    losses = (initial["weighted_loss"], current["weighted_loss"])
    if any(not math.isfinite(v) or v < 0 for v in losses) or losses[0] == 0:
        return ["invalid_dev_loss"]
    if initial["sequences"] != config["expected_dev_samples"] or any(
        initial[k] != current[k] for k in ("sequences", "rows", "weight_sum", "loss_tokens")
    ):
        return ["incomplete_dev_coverage"]
    key = (
        "maximum_final_dev_to_initial_loss_ratio" if final else "maximum_dev_to_initial_loss_ratio"
    )
    return ["dev_loss_regression"] if losses[1] / losses[0] > config[key] else []


def validate_manifest(manifest):
    from referencing import Registry, Resource

    registry = Registry()
    for name in (
        "training_preflight_manifest",
        "a0_training_preflight",
        "a0_full_sft",
        "a0_sft_capacity_manifest",
    ):
        registry = registry.with_resource(
            f"https://long-long-agent.local/{name}.schema.json",
            Resource.from_contents(json.loads((ROOT / f"schemas/{name}.schema.json").read_text())),
        )
    schema = json.loads((ROOT / "schemas/a0_full_sft_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, registry=registry).validate(manifest)


def run_sft(config_path: Path, *, lm: Path, cuda: Path, build: Path):
    config, base = load_sft_config(config_path)
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit SFT protocol before execution")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prior_path = data_root / config["capacity_result"]
    if sha256_file(prior_path) != config["capacity_result_sha256"]:
        raise ValueError("capacity evidence changed")
    prior = json.loads(prior_path.read_text())
    prior_manifest = json.loads((prior_path.parent / "manifest.json").read_text())
    validate_capacity_manifest(prior_manifest)
    if (
        prior["status"] != "passed"
        or sha256_file(prior_path.parent / "manifest.json") != prior["manifest_sha256"]
        or prior_manifest["resolved_config"]["input_manifest_sha256"]
        != config["input_manifest_sha256"]
    ):
        raise ValueError("matching complete-input capacity not passed")
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
        "failures": [],
        "validations": [],
        "checkpoints": [],
        "limitation": "one local A0 SFT epoch; teacher-forced dev CE only, not Agent execution, G1/G2 or latent training",
    }
    trainer = sampler = provenance = None
    cursor = 0
    started = time.perf_counter()
    try:
        torch.set_num_threads(config["cpu_threads"])
        samples = {
            split: load_split(
                data_root / config["input_directory"],
                split,
                expected_manifest_sha256=config["input_manifest_sha256"],
            )
            for split in ("train", "dev")
        }
        if any(len(samples[s]) != config[f"expected_{s}_samples"] for s in samples):
            raise ValueError("complete train/dev counts changed")
        train_ids = [s.sample_id for s in samples["train"]]
        if set(train_ids) & {s.sample_id for s in samples["dev"]}:
            raise ValueError("train/dev sample overlap")
        _, plan = capacity_plan(
            samples["train"],
            seed=config["training_seed"],
            max_tokens=base["max_tokens"],
            alignment=base["alignment"],
        )
        if plan != prior_manifest["plan"]:
            raise ValueError("SFT epoch plan differs from capacity-tested plan")

        def make_sampler(split, shuffle):
            return TokenBudgetPackSampler(
                [len(s.token_ids) - 1 for s in samples[split]],
                [s.sample_id for s in samples[split]],
                max_tokens=base["max_tokens"],
                alignment=base["alignment"],
                seed=config["training_seed"],
                shuffle=shuffle,
            )

        sampler, dev_sampler = make_sampler("train", True), make_sampler("dev", False)
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
            raise ValueError("SFT runtime differs from accepted capacity")
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
            "dev_sample_ids_sha256": tree_digest([s.sample_id for s in samples["dev"]]),
        }
        validate_manifest(manifest)
        write_json_once(output / "manifest.json", manifest)
        report["manifest_sha256"] = sha256_file(output / "manifest.json")
        trainer, report["parameters"] = make_trainer(network, base)
        training = yaml.safe_load((ROOT / "configs/training_data.yaml").read_text())
        vocab = ROOT / training["tokenizer"]["vocabulary"]
        if sha256_file(vocab) != runtime["tokenizer_sha256"]:
            raise ValueError("SFT tokenizer changed")
        collator = PackedEpisodeCollator(
            RWKVByteTokenizer(vocab), max_pack_tokens=base["max_tokens"], align_to=base["alignment"]
        )

        def validation(row):
            trainer.optimizer.zero_grad(set_to_none=True)
            value = evaluate(trainer, collator, samples["dev"], dev_sampler)
            write_json_once(output / f"dev_row_{row}.json", value)
            summary = {k: v for k, v in value.items() if k != "row_metrics"}
            report["validations"].append({"completed_rows": row, **summary})
            print(
                json.dumps({"stage": "dev_validation", "completed_rows": row, **summary}),
                flush=True,
            )
            return summary

        report["stage"] = "initial_dev"
        initial = validation(0)
        trained_ids, online_rows = [], []
        peak, update_seconds = 0, 0.0
        sampler.set_epoch(0)
        report["stage"] = "train_epoch_1"
        for row_index, indices in enumerate(sampler):
            batch = collator.collate_tokenized([samples["train"][i] for i in indices])
            entry = timed_step(trainer, batch, base)
            cursor = row_index + 1
            entry.update(completed_rows=cursor, sample_ids=list(batch["sample_ids"]))
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            trained_ids.extend(batch["sample_ids"])
            metrics = entry["metrics"]
            online_rows.append(
                {
                    "loss": metrics["loss"],
                    "weight_sum": metrics["loss_weight_sum"],
                    "loss_tokens": metrics["effective_loss_tokens"],
                    "sample_ids": list(batch["sample_ids"]),
                }
            )
            peak = max(peak, entry["peak_allocated_bytes"])
            update_seconds += entry["elapsed_seconds"]
            if entry["failed_checks"]:
                report["failures"].extend(entry["failed_checks"])
                raise RuntimeError("SFT resource or numerical stop rule")
            if cursor % 32 == 0 or cursor == 1:
                print(
                    json.dumps(
                        {
                            "stage": "training",
                            "completed_rows": cursor,
                            "total_rows": len(sampler),
                            **metrics,
                        }
                    ),
                    flush=True,
                )
            final = cursor == len(sampler)
            if cursor % config["checkpoint_every_rows"] == 0 or final:
                trainer.optimizer.zero_grad(set_to_none=True)
                path = output / f"row_{cursor}.pt"
                digest = save_checkpoint(
                    path, trainer, sampler, next_row=cursor, provenance=provenance
                )
                report["checkpoints"].append({"file": path.name, "sha256": digest})
            if cursor % config["validation_every_rows"] == 0 or final:
                current = validation(cursor)
                failed = validation_failures(initial, current, config, final=final)
                report["failures"].extend(failed)
                write_json_once(
                    output / f"progress_{cursor}.json",
                    {
                        "schema_version": 1,
                        "status": "failed" if failed else "running",
                        "progress": asdict(trainer.progress),
                        "validation": current,
                        "checkpoints": report["checkpoints"],
                        "failures": failed,
                    },
                )
                if failed:
                    raise RuntimeError("full-dev validation stop rule")
        if sorted(trained_ids) != sorted(train_ids) or trainer.progress.sequences != len(train_ids):
            raise ValueError("incomplete SFT epoch coverage")
        report["online_train_metrics"] = summarize_losses(online_rows)
        report["peak_allocated_bytes"] = peak
        report["update_tokens_per_second"] = trainer.progress.real_tokens / update_seconds
        report["effective_token_utilization"] = (
            trainer.progress.real_tokens / trainer.progress.aligned_tokens
        )
        report["status"], report["stage"] = "passed", "complete"
    except Exception as error:  # noqa: BLE001 -- preserve safe failure evidence
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
            except Exception as save_error:  # noqa: BLE001 -- report emergency save failures
                report["checkpoint_save_failure"] = type(save_error).__name__
    finally:
        report["elapsed_seconds_including_setup_validation_checkpoint"] = (
            time.perf_counter() - started
        )
        if trainer is not None:
            report["progress"] = asdict(trainer.progress)
        write_json_once(output / "result.json", report)
    return report
