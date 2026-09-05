"""Exact fixed-gradient control; does not pass the failed native acceptance."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import jsonschema
import torch
import yaml

from src.data.governance import sha256_file
from src.model.runtime import ROOT, load_runtime

from .checkpoint import load_checkpoint, save_checkpoint
from .episode_collator import PackedEpisodeCollator
from .preflight import (
    make_trainer,
    random_probe,
    reconstruct_inputs,
    timed_step,
    training_fingerprints,
    tree_digest,
    write_json_once,
)
from .resume_comparison import capture_gradients, install_gradients
from .resume_confirmation import (
    checked_gradients,
    confirmation_sampler,
    load_protocol,
    validate_manifest,
)


def load_diagnostic_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    jsonschema.validate(
        config, json.loads((ROOT / "schemas/fixed_gradient_diagnostic.schema.json").read_text())
    )
    if sha256_file(ROOT / config["source_protocol"]) != config["source_protocol_sha256"]:
        raise ValueError("diagnostic source protocol changed")
    return config


def run_diagnostic(
    config_path: Path, run_root: Path, phase: str, lm: Path, cuda: Path, build: Path
) -> dict:
    config = load_diagnostic_config(config_path)
    protocol, base = load_protocol(ROOT / config["source_protocol"])
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    run_root = run_root.resolve()
    if phase not in {"capture", "replay"} or not run_root.is_relative_to(
        data_root / "artifacts/training_preflight"
    ):
        raise ValueError("invalid diagnostic phase or output root")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit diagnostic protocol before execution")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = data_root / config["source_directory"]
    for kind in ("manifest", "result"):
        if sha256_file(source / f"{kind}.json") != config[f"source_{kind}_sha256"]:
            raise ValueError("diagnostic source identity mismatch")
    source_manifest = json.loads((source / "manifest.json").read_text())
    validate_manifest(source_manifest)
    source_result = json.loads((source / "result.json").read_text())
    if (
        source_result["stage"] != "no_update_controls"
        or source_result["status"] != "failed"
        or source_result["steps"][0]["failed_checks"]
    ):
        raise ValueError(
            "diagnostic requires the pinned failed-control source with valid first update"
        )
    if sha256_file(source / "step1.pt") != source_result["step1_checkpoint_sha256"]:
        raise ValueError("source checkpoint identity changed")
    if phase == "capture":
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        capture = json.loads((run_root / "capture/result.json").read_text())
        if capture["status"] != "diagnostic_complete":
            raise ValueError("gradient capture has not completed")
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
        "purpose": config["purpose"],
        "phase": phase,
        "status": "failed",
        "stage": "inputs",
        "failures": [],
        "limitation": "fixed-gradient mechanism only; native confirmation remains failed",
    }
    try:
        torch.set_num_threads(protocol["cpu_threads"])
        all_samples, _, tokenizer = reconstruct_inputs({**base, "sample_count": 128}, data_root)
        samples = [all_samples[i] for i in protocol["sample_indices"]]
        sampler = confirmation_sampler(samples, base, protocol)
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
        if runtime != source_manifest["runtime"] or environment != source_manifest["environment"]:
            raise ValueError("diagnostic runtime or environment differs from source")
        manifest = {
            "schema_version": 1,
            "purpose": config["purpose"],
            "phase": phase,
            "code_commit": commit,
            "config_sha256": sha256_file(config_path),
            "source_manifest_sha256": config["source_manifest_sha256"],
            "source_result_sha256": config["source_result_sha256"],
            "runtime_sha256": tree_digest(runtime),
            "environment_sha256": tree_digest(environment),
            "command": sys.argv,
        }
        jsonschema.validate(
            manifest,
            json.loads(
                (ROOT / "schemas/fixed_gradient_diagnostic_manifest.schema.json").read_text()
            ),
        )
        write_json_once(output / "manifest.json", manifest)
        report["manifest_sha256"] = sha256_file(output / "manifest.json")
        if phase == "replay":
            previous = json.loads((run_root / "capture/manifest.json").read_text())
            if sha256_file(run_root / "capture/manifest.json") != capture["manifest_sha256"] or any(
                previous[key] != manifest[key]
                for key in manifest
                if key not in {"phase", "command"}
            ):
                raise ValueError("capture/replay manifest mismatch")
            report["capture_result_sha256"] = sha256_file(run_root / "capture/result.json")
        provenance = source_manifest["provenance"]
        trainer, report["parameters"] = make_trainer(network, base)
        report["stage"] = "restore"
        cursor = load_checkpoint(source / "step1.pt", trainer, sampler, provenance=provenance)
        report["restored_fingerprints"] = training_fingerprints(trainer, sampler, cursor)
        if cursor != 1 or report["restored_fingerprints"] != source_result["step1_fingerprints"]:
            raise ValueError("restored source state not exact")
        report["rng_probe"] = random_probe()
        if report["rng_probe"] != source_result["rng_probe"]:
            raise ValueError("restored source RNG differs")
        batch = PackedEpisodeCollator(
            tokenizer, max_pack_tokens=base["max_tokens"], align_to=base["alignment"]
        ).collate_tokenized([samples[1]])
        batch_hash = tree_digest(batch)
        if batch_hash != source_result["next_batch_sha256"]:
            raise ValueError("source next batch differs")
        report["next_batch_sha256"] = batch_hash
        report["stage"] = "update"
        if phase == "capture":
            step = timed_step(trainer, batch, base)
            report["step"] = step
            report["failures"].extend(step["failed_checks"])
            gradients = capture_gradients(network)
            report["gradient_sha256"] = tree_digest(gradients)
            report["inactive_gradient_parameters"] = [n for n, g in gradients.items() if g is None]
            with (output / "gradients.pt").open("xb") as stream:
                torch.save(
                    {
                        "schema_version": 1,
                        "provenance": provenance,
                        "batch_sha256": batch_hash,
                        "gradient_sha256": report["gradient_sha256"],
                        "gradients": gradients,
                    },
                    stream,
                )
            report["gradient_file_sha256"] = sha256_file(output / "gradients.pt")
            del gradients
            cursor = 2
            report["backward_calls"] = 1
        else:
            gradients = checked_gradients(
                run_root / "capture/gradients.pt",
                capture["gradient_file_sha256"],
                provenance,
                batch_hash,
            )
            install_gradients(network, gradients)
            report["gradient_sha256"] = tree_digest(capture_gradients(network))
            if report["gradient_sha256"] != capture["gradient_sha256"]:
                raise ValueError("installed gradients not exact")
            del gradients
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            trainer.optimizer.step()
            torch.cuda.synchronize()
            report["optimizer_only_seconds"] = time.perf_counter() - started
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if (
                report["optimizer_only_seconds"] > base["acceptance"]["maximum_step_seconds"]
                or report["peak_allocated_bytes"] / 2**20
                > base["acceptance"]["maximum_peak_allocated_mib"]
            ):
                report["failures"].append("resource_limit")
            report["backward_calls"] = 0
        report["optimizer_updates"] = 1
        trainer.optimizer.zero_grad(set_to_none=True)
        report["final_fingerprints"] = training_fingerprints(trainer, sampler, cursor)
        report["final_checkpoint_sha256"] = save_checkpoint(
            output / "final.pt", trainer, sampler, next_row=cursor, provenance=provenance
        )
        if phase == "replay":
            for key, expected in (
                *[(k, capture["final_fingerprints"][k]) for k in ("model", "optimizer", "rng")],
                *[(k, source_result["step1_fingerprints"][k]) for k in ("trainer", "sampler")],
            ):
                if report["final_fingerprints"][key] != expected:
                    report["failures"].append(key + "_not_exact")
        if not report["failures"]:
            report["status"], report["stage"] = "diagnostic_complete", "complete"
    except Exception as error:  # noqa: BLE001 -- keep failure evidence without source text
        report["failures"].append(type(error).__name__)
        report["exception_sites"] = [
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(error.__traceback__)
        ]
    finally:
        write_json_once(output / "result.json", report)
    return report
