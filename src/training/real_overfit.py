"""Bounded real train-only 32/128 overfit after independently accepted GPU resume."""

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
from referencing import Registry, Resource

from src.data.governance import sha256_file
from src.model.runtime import ROOT, load_runtime

from .checkpoint import save_checkpoint
from .episode_collator import PackedEpisodeCollator
from .preflight import (
    load_config,
    make_trainer,
    padded_baseline,
    reconstruct_inputs,
    timed_step,
    tree_digest,
    write_json_once,
)
from .sft_trainer import move_packed_batch
from .token_budget_sampler import TokenBudgetPackSampler

PHASES = ("capacity", "padded", "overfit32", "overfit128")


def load_overfit_config(path: Path):
    config = yaml.safe_load(path.read_text())
    jsonschema.validate(
        config, json.loads((ROOT / "schemas/a0_real_overfit.schema.json").read_text())
    )
    json.dumps(config, allow_nan=False)
    if sha256_file(ROOT / config["base_config"]) != config["base_config_sha256"]:
        raise ValueError("overfit base configuration changed")
    return config, load_config(ROOT / config["base_config"])


def validate_manifest(manifest: dict):
    registry = Registry()
    for name in ("training_preflight_manifest", "a0_training_preflight", "a0_real_overfit"):
        registry = registry.with_resource(
            f"https://long-long-agent.local/{name}.schema.json",
            Resource.from_contents(json.loads((ROOT / f"schemas/{name}.schema.json").read_text())),
        )
    schema = json.loads((ROOT / "schemas/a0_real_overfit_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, registry=registry).validate(manifest)


def summarize_losses(rows: list[dict]) -> dict:
    if not rows or any(
        not math.isfinite(r["loss"])
        or r["loss"] < 0
        or not math.isfinite(r["weight_sum"])
        or r["weight_sum"] <= 0
        for r in rows
    ):
        raise ValueError("invalid evaluation loss or denominator")
    weights = sum(r["weight_sum"] for r in rows)
    return {
        "weighted_loss": sum(r["loss"] * r["weight_sum"] for r in rows) / weights,
        "weight_sum": weights,
        "loss_tokens": sum(r["loss_tokens"] for r in rows),
        "sequences": sum(len(r["sample_ids"]) for r in rows),
        "rows": len(rows),
        "row_loss_quantiles": {
            str(q): float(np.quantile([r["loss"] for r in rows], q)) for q in (0, 0.5, 0.9, 1)
        },
    }


def evaluate(trainer, collator, samples, sampler, *, device="cuda") -> dict:
    trainer.network.eval()
    rows = []
    with torch.no_grad():
        for indices in sampler:
            batch = move_packed_batch(
                collator.collate_tokenized([samples[i] for i in indices]), device
            )
            value = trainer.compute_loss(batch)
            rows.append(
                {
                    "sample_ids": list(batch["sample_ids"]),
                    "loss": float(value.total),
                    "weight_sum": value.weight_sum,
                    "loss_tokens": value.effective_tokens,
                }
            )
    return {**summarize_losses(rows), "row_metrics": rows}


def overfit_failures(
    initial: float, losses: list[float], count: int, presentations: int, config: dict
) -> list[str]:
    acceptance = config["acceptance"]
    if (
        not math.isfinite(initial)
        or initial <= 0
        or any(not math.isfinite(x) or x < 0 for x in losses)
    ):
        return ["nonfinite_or_invalid_loss"]
    failed = []
    if len(losses) != config["epochs"] or presentations != count * config["epochs"]:
        failed.append("incomplete_coverage")
    floor = acceptance["loss_ratio_denominator_floor"]
    if (
        not losses
        or losses[-1] / max(initial, floor) > acceptance["maximum_final_to_initial_loss_ratio"]
    ):
        failed.append("final_loss_ratio")
    if any(
        current / max(previous, floor) > acceptance["maximum_epoch_to_previous_loss_ratio"]
        for current, previous in zip(losses, [initial, *losses[:-1]][: len(losses)], strict=True)
    ):
        failed.append("epoch_loss_spike")
    return failed


def run_overfit(config_path: Path, run_root: Path, phase: str, lm: Path, cuda: Path, build: Path):
    config_path, run_root = config_path.resolve(), run_root.resolve()
    config, base = load_overfit_config(config_path)
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    if phase not in PHASES or not run_root.is_relative_to(data_root / "artifacts/real_sft"):
        raise ValueError("invalid real overfit phase or storage root")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit the real-training protocol before GPU execution")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    recovery_path = data_root / config["resume_result"]
    if sha256_file(recovery_path) != config["resume_result_sha256"]:
        raise ValueError("independent resume result changed")
    recovery = json.loads(recovery_path.read_text())
    if recovery["status"] != "passed" or recovery["phase"] != "native":
        raise ValueError("independent native resume not passed")
    prior = None
    if phase == "capacity":
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        previous_phase = PHASES[PHASES.index(phase) - 1]
        prior = json.loads((run_root / previous_phase / "result.json").read_text())
        if prior["status"] != "passed":
            raise ValueError("previous real-training gate has not passed")
    output = run_root / phase
    output.mkdir(exist_ok=False)
    write_json_once(
        output / "intent.json",
        {
            "schema_version": 1,
            "phase": phase,
            "config_sha256": sha256_file(config_path),
            "code_commit": commit,
            "command": sys.argv,
        },
    )
    report = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "phase": phase,
        "status": "failed",
        "stage": "inputs",
        "epochs": [],
        "steps": [],
        "failures": [],
        "checkpoints": [],
        "resume_result_sha256": config["resume_result_sha256"],
        "limitation": "train-only engineering memorization; not all-pool SFT, Agent outcome, G1 or distributed validation",
    }
    trainer = sampler = provenance = None
    cursor = 0
    try:
        torch.set_num_threads(config["cpu_threads"])
        all_samples, _, tokenizer = reconstruct_inputs({**base, "sample_count": 128}, data_root)
        count = 1 if phase in ("capacity", "padded") else int(phase.removeprefix("overfit"))
        samples = (
            [all_samples[config["capacity_sample_index"]]] if count == 1 else all_samples[:count]
        )
        if count == 1 and sum(samples[0].loss_token_counts().values()) != max(
            sum(s.loss_token_counts().values()) for s in all_samples
        ):
            raise ValueError("capacity input no longer covers the longest supervision")
        lengths, ids = [len(s.token_ids) - 1 for s in samples], [s.sample_id for s in samples]

        def make_sampler(shuffle):
            return TokenBudgetPackSampler(
                lengths,
                ids,
                max_tokens=base["max_tokens"],
                alignment=base["alignment"],
                seed=config["training_seed"],
                shuffle=shuffle,
            )

        sampler, evaluation_sampler = make_sampler(count != 1), make_sampler(False)
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
        recovery_manifest = json.loads((recovery_path.parent / "manifest.json").read_text())
        if (
            sha256_file(recovery_path.parent / "manifest.json") != recovery["manifest_sha256"]
            or runtime != recovery_manifest["runtime"]
            or environment != recovery_manifest["environment"]
        ):
            raise ValueError("real training differs from accepted resume runtime")
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
            "phase": phase,
            "code_commit": commit,
            "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
            "provenance": provenance,
            "runtime": runtime,
            "environment": environment,
            "input_plan_sha256": base["input_plan_sha256"],
            "command": sys.argv,
            "resolved_config": config,
            "base_config": base,
            "sample_count": count,
            "sample_ids_sha256": tree_digest(ids),
        }
        validate_manifest(manifest)
        write_json_once(output / "manifest.json", manifest)
        report["manifest_sha256"] = sha256_file(output / "manifest.json")
        if prior is not None:
            prior_path = run_root / PHASES[PHASES.index(phase) - 1] / "manifest.json"
            previous_manifest = json.loads(prior_path.read_text())
            validate_manifest(previous_manifest)
            if (
                sha256_file(prior_path) != prior["manifest_sha256"]
                or previous_manifest["code_commit"] != commit
                or previous_manifest["provenance"] != provenance
            ):
                raise ValueError("training phases have incompatible provenance")
        trainer, report["parameters"] = make_trainer(network, base)
        collator = PackedEpisodeCollator(
            tokenizer, max_pack_tokens=base["max_tokens"], align_to=base["alignment"]
        )
        report["sample_count"] = count
        report["sample_ids"] = ids
        started = time.perf_counter()
        if count != 1:
            report["stage"] = "initial_evaluation"
            initial = evaluate(trainer, collator, samples, evaluation_sampler)
            write_json_once(output / "evaluation_initial.json", initial)
            report["initial_evaluation"] = {k: v for k, v in initial.items() if k != "row_metrics"}
            print(
                json.dumps({"phase": phase, "epoch": 0, **report["initial_evaluation"]}), flush=True
            )
        epochs = config["capacity_updates"] if count == 1 else config["epochs"]
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            cursor = 0
            report["stage"] = f"epoch_{epoch + 1}"
            epoch_samples = []
            for row_index, indices in enumerate(sampler):
                batch = collator.collate_tokenized([samples[i] for i in indices])
                if phase == "padded":
                    batch = padded_baseline(batch, base["max_tokens"])
                entry = timed_step(trainer, batch, base)
                cursor = row_index + 1
                entry.update(
                    epoch=epoch + 1, row_index=row_index, sample_ids=list(batch["sample_ids"])
                )
                report["steps"].append(entry)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + "\n")
                epoch_samples.extend(batch["sample_ids"])
                if entry["failed_checks"]:
                    report["failures"].extend(entry["failed_checks"])
                    raise RuntimeError("real-training resource or finite stop rule")
                if count == 1 or cursor % 32 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": phase,
                                "epoch": epoch + 1,
                                "row": cursor,
                                **entry["metrics"],
                                "seconds": entry["elapsed_seconds"],
                            }
                        ),
                        flush=True,
                    )
            if sorted(epoch_samples) != sorted(ids):
                raise ValueError("epoch sample coverage mismatch")
            trainer.optimizer.zero_grad(set_to_none=True)
            if count != 1:
                evaluation = evaluate(trainer, collator, samples, evaluation_sampler)
                write_json_once(output / f"evaluation_epoch_{epoch + 1}.json", evaluation)
                report["epochs"].append(
                    {
                        "epoch": epoch + 1,
                        **{k: v for k, v in evaluation.items() if k != "row_metrics"},
                    }
                )
                print(json.dumps({"phase": phase, **report["epochs"][-1]}), flush=True)
                previous_loss = (
                    report["epochs"][-2]["weighted_loss"] if epoch else initial["weighted_loss"]
                )
                if (
                    evaluation["weighted_loss"]
                    / max(previous_loss, config["acceptance"]["loss_ratio_denominator_floor"])
                    > config["acceptance"]["maximum_epoch_to_previous_loss_ratio"]
                ):
                    report["failures"].append("epoch_loss_spike")
                    raise RuntimeError("epoch loss spike")
            if epoch + 1 in config["checkpoint_epochs"] or epoch == epochs - 1:
                path = output / f"epoch_{epoch + 1}.pt"
                digest = save_checkpoint(
                    path, trainer, sampler, next_row=cursor, provenance=provenance
                )
                report["checkpoints"].append({"file": path.name, "sha256": digest})
        report["elapsed_seconds_including_evaluation_and_checkpoint"] = (
            time.perf_counter() - started
        )
        report["progress"] = asdict(trainer.progress)
        report["peak_allocated_bytes"] = max(s["peak_allocated_bytes"] for s in report["steps"])
        report["effective_token_utilization"] = (
            trainer.progress.real_tokens / trainer.progress.aligned_tokens
        )
        report["update_tokens_per_second"] = trainer.progress.real_tokens / sum(
            s["elapsed_seconds"] for s in report["steps"]
        )
        if phase == "padded":
            report["packed_to_padded_throughput_ratio"] = sum(
                s["elapsed_seconds"] for s in report["steps"]
            ) / sum(s["elapsed_seconds"] for s in prior["steps"])
            report["timing_limitation"] = (
                "n=2 includes first-step allocation/warmup, excludes checkpoint IO; no speedup claim"
            )
        elif count != 1:
            losses = [e["weighted_loss"] for e in report["epochs"]]
            report["failures"].extend(
                overfit_failures(
                    initial["weighted_loss"], losses, count, trainer.progress.sequences, config
                )
            )
            report["final_to_initial_loss_ratio"] = losses[-1] / max(
                initial["weighted_loss"], config["acceptance"]["loss_ratio_denominator_floor"]
            )
        if not report["failures"]:
            report["status"], report["stage"] = "passed", "complete"
    except Exception as error:  # noqa: BLE001 -- preserve failures, not source text
        report["failures"].append(type(error).__name__)
        report["exception_sites"] = [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(error.__traceback__)
        ]
        if trainer is not None and provenance is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
            try:
                digest = save_checkpoint(
                    output / "stopped.pt", trainer, sampler, next_row=cursor, provenance=provenance
                )
                report["checkpoints"].append({"file": "stopped.pt", "sha256": digest})
                report["progress"] = asdict(trainer.progress)
            except Exception as save_error:  # noqa: BLE001 -- record a failed emergency save separately
                report["checkpoint_save_failure"] = type(save_error).__name__
    finally:
        write_json_once(output / "result.json", report)
    return report
