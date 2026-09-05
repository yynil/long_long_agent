"""Preregistered, three-process ADR-020 confirmation on independent train tasks."""

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
from pathlib import Path

import jsonschema
import numpy as np
import torch
import yaml
from referencing import Registry, Resource

from src.data.governance import sha256_file
from src.model.runtime import ROOT, load_runtime

from .checkpoint import load_checkpoint, save_checkpoint
from .episode_collator import PackedEpisodeCollator
from .preflight import (
    load_config,
    make_trainer,
    random_probe,
    reconstruct_inputs,
    timed_step,
    training_fingerprints,
    tree_digest,
    write_json_once,
)
from .resume_comparison import (
    capture_gradients,
    gradient_failures,
    install_gradients,
    optimizer_comparison,
    tensor_comparison,
)
from .sft_trainer import move_packed_batch
from .token_budget_sampler import TokenBudgetPackSampler


def load_protocol(path: Path) -> tuple[dict, dict]:
    config = yaml.safe_load(path.read_text())
    jsonschema.validate(
        config, json.loads((ROOT / "schemas/a0_resume_confirmation.schema.json").read_text())
    )
    # JSON serialization rejects non-finite YAML floats even in deeply nested budgets.
    json.dumps(config, allow_nan=False)
    base_path = ROOT / config["base_config"]
    if sha256_file(base_path) != config["base_config_sha256"]:
        raise ValueError("base preflight configuration changed")
    return config, load_config(base_path)


def validate_manifest(manifest: dict) -> None:
    registry = Registry()
    for name in ("training_preflight_manifest", "a0_training_preflight", "a0_resume_confirmation"):
        schema = json.loads((ROOT / f"schemas/{name}.schema.json").read_text())
        registry = registry.with_resource(
            f"https://long-long-agent.local/{name}.schema.json", Resource.from_contents(schema)
        )
    schema = json.loads((ROOT / "schemas/resume_confirmation_manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema, registry=registry).validate(manifest)


def checked_gradients(path: Path, digest: str, provenance: dict, batch_hash: str) -> dict:
    if sha256_file(path) != digest:
        raise ValueError("gradient artifact hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(payload)
        != {"schema_version", "provenance", "batch_sha256", "gradient_sha256", "gradients"}
        or payload["schema_version"] != 1
    ):
        raise ValueError("unknown gradient payload")
    if payload["provenance"] != provenance or payload["batch_sha256"] != batch_hash:
        raise ValueError("gradient provenance mismatch")
    if tree_digest(payload["gradients"]) != payload["gradient_sha256"]:
        raise ValueError("gradient tensor hash mismatch")
    return payload["gradients"]


def no_update_controls(trainer, batch, base: dict, protocol: dict, report: dict) -> None:
    first = None
    report["controls"] = []
    for index in range(protocol["backward_repeats"]):
        trainer.network.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss = trainer.compute_loss(move_packed_batch(batch, "cuda"))
        if not torch.isfinite(loss.total):
            raise FloatingPointError("control loss not finite")
        loss.total.backward()
        parameters = [p for group in trainer.optimizer.param_groups for p in group["params"]]
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, base["gradient_clip_norm"], error_if_nonfinite=True
        )
        torch.cuda.synchronize()
        entry = {
            "repeat": index,
            "loss": float(loss.total.detach()),
            "gradient_norm": float(norm),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        actual = capture_gradients(trainer.network)
        comparison = tensor_comparison(
            actual if first is None else first,
            actual,
            denominator_floor=protocol["acceptance"]["relative_l2_denominator_floor"],
        )
        entry["gradient_comparison"] = comparison
        entry["gradient_sha256"] = tree_digest(actual)
        entry["failed_checks"] = gradient_failures(comparison, protocol["acceptance"], [])
        for key, value, limit in (
            ("loss", entry["loss"], base["acceptance"]["maximum_loss"]),
            (
                "gradient_norm",
                entry["gradient_norm"],
                base["acceptance"]["maximum_preclip_gradient_norm"],
            ),
            ("seconds", entry["elapsed_seconds"], base["acceptance"]["maximum_step_seconds"]),
            (
                "memory",
                entry["peak_allocated_bytes"] / 2**20,
                base["acceptance"]["maximum_peak_allocated_mib"],
            ),
        ):
            if not math.isfinite(value) or value > limit:
                entry["failed_checks"].append(key)
        if index and (entry["loss"], entry["gradient_norm"]) != (
            report["controls"][0]["loss"],
            report["controls"][0]["gradient_norm"],
        ):
            entry["failed_checks"].append("control_metrics_not_exact")
        report["controls"].append(entry)
        print(
            json.dumps(
                {
                    "control": index,
                    "different_tensors": comparison["different_tensors"],
                    "failed_checks": entry["failed_checks"],
                }
            ),
            flush=True,
        )
        if entry["failed_checks"]:
            raise RuntimeError("no-update control failed")
        if first is None:
            first = actual
        del actual, loss
    trainer.optimizer.zero_grad(set_to_none=True)


def run_confirmation(
    config_path: Path, run_root: Path, phase: str, lm: Path, cuda: Path, build: Path
) -> dict:
    config_path, run_root = config_path.resolve(), run_root.resolve()
    protocol, base = load_protocol(config_path)
    if phase not in {"reference", "fixed", "native"}:
        raise ValueError("unknown confirmation phase")
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    if not run_root.is_relative_to(data_root / "artifacts/training_preflight"):
        raise ValueError("confirmation artifacts must be under configured data root")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit the protocol before GPU execution")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    reference = None
    if phase == "reference":
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        reference = json.loads((run_root / "reference/result.json").read_text())
        if reference["status"] != "passed":
            raise ValueError("reference has not passed")
        if phase == "native":
            fixed = json.loads((run_root / "fixed/result.json").read_text())
            if fixed["status"] != "passed" or fixed["reference_result_sha256"] != sha256_file(
                run_root / "reference/result.json"
            ):
                raise ValueError("fixed-gradient confirmation has not passed")
    output = run_root / phase
    output.mkdir(exist_ok=False)
    write_json_once(
        output / "intent.json",
        {
            "schema_version": 1,
            "phase": phase,
            "code_commit": commit,
            "config_sha256": sha256_file(config_path),
            "command": sys.argv,
        },
    )
    report = {
        "schema_version": 1,
        "purpose": protocol["purpose"],
        "phase": phase,
        "status": "failed",
        "stage": "inputs",
        "steps": [],
        "failures": [],
        "limitation": "two independent train tasks; bounded local GPU resume only, not 32/128 overfit or G1",
    }
    try:
        torch.set_num_threads(protocol["cpu_threads"])
        # Revalidate the entire immutable 128-input plan; base YAML remains untouched.
        all_samples, _, tokenizer = reconstruct_inputs({**base, "sample_count": 128}, data_root)
        samples = [all_samples[i] for i in protocol["sample_indices"]]
        sampler = TokenBudgetPackSampler(
            [len(s.token_ids) - 1 for s in samples],
            [s.sample_id for s in samples],
            max_tokens=base["max_tokens"],
            alignment=base["alignment"],
            seed=protocol["training_seed"],
            shuffle=False,
        )
        if list(sampler) != [[0], [1]]:
            raise ValueError("confirmation row order changed")
        random.seed(protocol["training_seed"])
        np.random.seed(protocol["training_seed"])
        torch.manual_seed(protocol["training_seed"])
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
            "purpose": protocol["purpose"],
            "phase": phase,
            "code_commit": commit,
            "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
            "provenance": provenance,
            "runtime": runtime,
            "environment": environment,
            "input_plan_sha256": base["input_plan_sha256"],
            "command": sys.argv,
            "resolved_config": protocol,
            "base_config": base,
            "selected_samples": [
                {
                    "input_plan_index": i,
                    "sample_id": s.sample_id,
                    "input_tokens": len(s.token_ids) - 1,
                }
                for i, s in zip(protocol["sample_indices"], samples, strict=True)
            ],
        }
        validate_manifest(manifest)
        write_json_once(output / "manifest.json", manifest)
        report["manifest_sha256"] = sha256_file(output / "manifest.json")
        if reference is not None:
            prior_path = run_root / "reference/manifest.json"
            prior = json.loads(prior_path.read_text())
            validate_manifest(prior)
            if sha256_file(prior_path) != reference["manifest_sha256"] or any(
                prior[key] != manifest[key]
                for key in (
                    "provenance",
                    "code_commit",
                    "selected_samples",
                    "runtime",
                    "environment",
                    "resolved_config",
                    "base_config",
                )
            ):
                raise ValueError("cross-process provenance changed")
            report["reference_result_sha256"] = sha256_file(run_root / "reference/result.json")
        trainer, report["parameters"] = make_trainer(network, base)
        collator = PackedEpisodeCollator(
            tokenizer, max_pack_tokens=base["max_tokens"], align_to=base["alignment"]
        )
        batch = collator.collate_tokenized([samples[1]])
        report["next_batch_sha256"] = tree_digest(batch)
        report["sample_ids"] = list(batch["sample_ids"])
        if phase == "reference":
            report["stage"] = "warmup_update"
            warmup = timed_step(trainer, collator.collate_tokenized([samples[0]]), base)
            report["steps"].append(warmup)
            trainer.optimizer.zero_grad(set_to_none=True)
            report["step1_checkpoint_sha256"] = save_checkpoint(
                output / "step1.pt", trainer, sampler, next_row=1, provenance=provenance
            )
            report["step1_fingerprints"] = training_fingerprints(trainer, sampler, 1)
            if warmup["failed_checks"]:
                raise RuntimeError("warmup resource stop")
            report["rng_probe"] = random_probe()
            before = training_fingerprints(trainer, sampler, 1)
            report["stage"] = "no_update_controls"
            no_update_controls(trainer, batch, base, protocol, report)
            report["controls_state_unchanged"] = (
                training_fingerprints(trainer, sampler, 1) == before
            )
            if not report["controls_state_unchanged"]:
                raise ValueError("controls mutated model, optimizer, counters or RNG")
        else:
            report["stage"] = "restore"
            cursor = load_checkpoint(
                run_root / "reference/step1.pt", trainer, sampler, provenance=provenance
            )
            restored = training_fingerprints(trainer, sampler, cursor)
            report["restored_fingerprints"] = restored
            if (
                cursor != 1
                or restored != reference["step1_fingerprints"]
                or report["next_batch_sha256"] != reference["next_batch_sha256"]
            ):
                raise ValueError("restore or next batch not exact")
            report["rng_probe"] = random_probe()
            if report["rng_probe"] != reference["rng_probe"]:
                raise ValueError("restored RNG stream differs")
        report["stage"] = "fixed_update" if phase == "fixed" else "native_update"
        if phase == "fixed":
            gradients = checked_gradients(
                run_root / "reference/gradients.pt",
                reference["gradient_file_sha256"],
                provenance,
                report["next_batch_sha256"],
            )
            install_gradients(network, gradients)
            if tree_digest(capture_gradients(network)) != tree_digest(gradients):
                raise ValueError("installed gradients not exact")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            trainer.optimizer.step()
            torch.cuda.synchronize()
            report["optimizer_only_seconds"] = time.perf_counter() - started
            report["optimizer_only_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if report["optimizer_only_seconds"] > base["acceptance"]["maximum_step_seconds"]:
                report["failures"].append("fixed_update_seconds")
            if (
                report["optimizer_only_peak_allocated_bytes"] / 2**20
                > base["acceptance"]["maximum_peak_allocated_mib"]
            ):
                report["failures"].append("fixed_update_memory")
            report["optimizer_only_updates"] = 1
            report["backward_calls"] = 0
            del gradients
        else:
            step = timed_step(trainer, batch, base)
            report["steps"].append(step)
            print(
                json.dumps(
                    {"phase": phase, **step["metrics"], "failed_checks": step["failed_checks"]}
                ),
                flush=True,
            )
            gradients = capture_gradients(network)
            report["gradient_sha256"] = tree_digest(gradients)
            if phase == "reference":
                with (output / "gradients.pt").open("xb") as stream:
                    torch.save(
                        {
                            "schema_version": 1,
                            "provenance": provenance,
                            "batch_sha256": report["next_batch_sha256"],
                            "gradient_sha256": report["gradient_sha256"],
                            "gradients": gradients,
                        },
                        stream,
                    )
                report["gradient_file_sha256"] = sha256_file(output / "gradients.pt")
            else:
                expected = checked_gradients(
                    run_root / "reference/gradients.pt",
                    reference["gradient_file_sha256"],
                    provenance,
                    report["next_batch_sha256"],
                )
                comparison = tensor_comparison(
                    expected,
                    gradients,
                    denominator_floor=protocol["acceptance"]["relative_l2_denominator_floor"],
                )
                report["gradient_comparison"] = comparison
                report["failures"].extend(
                    gradient_failures(
                        comparison,
                        protocol["acceptance"],
                        [c["gradient_comparison"] for c in reference["controls"]],
                    )
                )
                del expected
            del gradients
            report["failures"].extend(step["failed_checks"])
        trainer.optimizer.zero_grad(set_to_none=True)
        cursor = 1 if phase == "fixed" else 2
        report["final_fingerprints"] = training_fingerprints(trainer, sampler, cursor)
        # Save finite states BEFORE testing equivalence; never discard a failed branch.
        report["final_checkpoint_sha256"] = save_checkpoint(
            output / "final.pt", trainer, sampler, next_row=cursor, provenance=provenance
        )
        report["stage"] = "comparison"
        if phase == "fixed":
            final = report["final_fingerprints"]
            report["failures"].extend(
                key
                for key in ("model", "optimizer", "rng")
                if final[key] != reference["final_fingerprints"][key]
            )
            report["failures"].extend(
                key
                for key in ("trainer", "sampler")
                if final[key] != reference["step1_fingerprints"][key]
            )
        elif phase == "native":
            report["failures"].extend(
                key
                for key in protocol["acceptance"]["exact_sections"]
                if report["final_fingerprints"][key] != reference["final_fingerprints"][key]
            )
            if report["steps"][0]["metrics"] != reference["steps"][1]["metrics"]:
                report["failures"].append("metrics_not_exact")
            checkpoint_path = run_root / "reference/final.pt"
            if sha256_file(checkpoint_path) != reference["final_checkpoint_sha256"]:
                raise ValueError("reference final checkpoint hash changed")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            report["optimizer_comparison"] = optimizer_comparison(
                payload["optimizer"], trainer.optimizer.state_dict(), protocol["acceptance"]
            )
            report["failures"].extend(report["optimizer_comparison"]["failed_checks"])
            del payload
        network.eval()
        with torch.no_grad():
            post_loss = float(trainer.compute_loss(move_packed_batch(batch, "cuda")).total)
        report["post_update_loss"] = post_loss
        if not math.isfinite(post_loss) or post_loss > base["acceptance"]["maximum_loss"]:
            report["failures"].append("post_update_loss")
        if reference is not None and post_loss != reference["post_update_loss"]:
            report["failures"].append("post_update_loss_not_exact")
        if not report["failures"]:
            report["status"], report["stage"] = "passed", "complete"
    except Exception as error:  # noqa: BLE001 -- preserve evidence, never print trajectory text
        report["failures"].append(type(error).__name__)
        report["exception_sites"] = [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(error.__traceback__)
        ]
        if torch.cuda.is_initialized():
            report["exception_memory"] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
    finally:
        write_json_once(output / "result.json", report)
    return report
