"""Bounded real-A0 GPU updates and a fresh-process, exact resume comparison.

This is an engineering preflight, not a 32/128 overfit or an Agent outcome run.
Only aggregate metrics, hashes, and trusted checkpoint tensors are persisted.
"""

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
from src.data.util import canonical_json
from src.model.runtime import ROOT, load_runtime

from .a0_dataset import prepare_overfit_inputs
from .checkpoint import load_checkpoint, save_checkpoint
from .episode_collator import PackedEpisodeCollator
from .parameter_groups import build_fp32_master_adamw, build_rwkv7_optimizer_plan
from .sft_trainer import PackedAgentSFTTrainer, move_packed_batch, validate_packed_tensor_batch
from .token_budget_sampler import TokenBudgetPackSampler
from .tokenizer import RWKVByteTokenizer


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    schema = json.loads((ROOT / "schemas/a0_training_preflight.schema.json").read_text())
    jsonschema.validate(config, schema)
    for value in config.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite preflight configuration")
    for value in config["acceptance"].values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite preflight acceptance")
    return config


def write_json_once(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_manifest(manifest: dict) -> None:
    schema = json.loads((ROOT / "schemas/training_preflight_manifest.schema.json").read_text())
    # Validate the embedded config with the same closed schema as the input file.
    jsonschema.validate(manifest, schema)
    jsonschema.validate(
        manifest["resolved_config"],
        json.loads((ROOT / "schemas/a0_training_preflight.schema.json").read_text()),
    )


def tree_digest(value) -> str:
    """Content fingerprint independent of torch.save archive layout; supports BF16."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(
                canonical_json(["tensor", str(tensor.dtype), list(tensor.shape)]).encode()
            )
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(canonical_json([type(item).__name__, item]).encode())

    visit(value)
    return digest.hexdigest()


def training_fingerprints(trainer, sampler, next_row: int) -> dict:
    numpy_state = np.random.get_state()
    return {
        "model": tree_digest(trainer.network.state_dict()),
        "optimizer": tree_digest(trainer.optimizer.state_dict()),
        "trainer": tree_digest(trainer.state_dict()),
        "sampler": tree_digest({"plan": asdict(sampler.plan()), "next_row": next_row}),
        "rng": tree_digest(
            {
                "python": random.getstate(),
                "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
        ),
    }


def random_probe() -> dict:
    return {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch": tree_digest(torch.rand(8)),
        "cuda": tree_digest(torch.rand(8, device="cuda")),
    }


def padded_baseline(batch: dict, max_tokens: int) -> dict:
    validate_packed_tensor_batch(batch)
    extra = max_tokens - batch["input_ids"].shape[1]
    if extra < 0:
        raise ValueError("padded baseline would truncate input")
    values = dict(batch)
    for name, fill in (
        ("input_ids", 0),
        ("targets", -100),
        ("loss_weights", 0),
        ("sequence_start_mask", 0),
        ("valid_token_mask", False),
        ("segment_ids", -1),
    ):
        values[name] = torch.nn.functional.pad(batch[name], (0, extra), value=fill)
    real = int(batch["cu_seqlens"][-1])
    if real < max_tokens:
        values["sequence_start_mask"][0, real] = 1
    validate_packed_tensor_batch(values)
    return values


def reconstruct_inputs(config: dict, data_root: Path):
    release = data_root / "releases" / config["release_id"]
    plan_path = data_root / config["input_plan"]
    if sha256_file(release / "manifest.json") != config["release_manifest_sha256"]:
        raise ValueError("A0 release identity mismatch")
    if sha256_file(plan_path) != config["input_plan_sha256"]:
        raise ValueError("real input plan identity mismatch")
    plan = json.loads(plan_path.read_text())
    jsonschema.validate(
        plan, json.loads((ROOT / "schemas/overfit_input_plan.schema.json").read_text())
    )
    training = yaml.safe_load((ROOT / "configs/training_data.yaml").read_text())
    vocabulary = ROOT / training["tokenizer"]["vocabulary"]
    if sha256_file(vocabulary) != plan["tokenizer_sha256"]:
        raise ValueError("input tokenizer identity mismatch")
    if (plan["seed"], plan["max_tokens"], plan["release_manifest_sha256"]) != (
        config["seed"],
        config["max_tokens"],
        config["release_manifest_sha256"],
    ):
        raise ValueError("input plan configuration mismatch")
    tokenizer = RWKVByteTokenizer(vocabulary)
    samples, metadata, rejected = prepare_overfit_inputs(
        release, tokenizer, count=128, max_tokens=config["max_tokens"], seed=config["seed"]
    )
    if metadata != plan["samples"] or rejected != plan["rejected_before_128"]:
        raise ValueError("reconstructed real inputs differ from frozen plan")
    samples = samples[: config["sample_count"]]
    sampler = TokenBudgetPackSampler(
        [len(sample.token_ids) - 1 for sample in samples],
        [sample.sample_id for sample in samples],
        max_tokens=config["max_tokens"],
        alignment=config["alignment"],
        seed=config["seed"],
        shuffle=config["shuffle"],
    )
    if sampler.plan().rank_rows[0].real_tokens < config["acceptance"]["minimum_first_row_tokens"]:
        raise ValueError("preflight does not cover the required long context")
    return samples, sampler, tokenizer


def make_trainer(network, config):
    network.requires_grad_(True)
    if network.args.grad_cp != 0 or config["gradient_checkpointing"]:
        raise ValueError("preflight requires the unrecomputed official path")
    plan = build_rwkv7_optimizer_plan(
        network.named_parameters(),
        n_layer=network.args.n_layer,
        weight_decay=config["weight_decay"],
        base_prefix="",
        latent_prefix=None,
        value_prefix=None,
    )
    if plan.trainable_numel != sum(p.numel() for p in network.parameters()):
        raise ValueError("preflight is not full-parameter training")
    optimizer = build_fp32_master_adamw(
        plan,
        learning_rate=config["learning_rate"],
        beta1=config["beta1"],
        beta2=config["beta2"],
        epsilon=config["epsilon"],
        fused=config["fused_adamw"],
    )
    return PackedAgentSFTTrainer(
        network,
        optimizer,
        head_chunk_tokens=config["head_chunk_tokens"],
        gradient_clip_norm=config["gradient_clip_norm"],
    ), {"trainable_numel": plan.trainable_numel, "parameter_tensors": plan.covered_parameter_count}


def timed_step(trainer, batch, config: dict) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    metrics = asdict(trainer.train_step(move_packed_batch(batch, "cuda")))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    acceptance = config["acceptance"]
    failed = []
    for key, value, limit in (
        ("loss", metrics["loss"], acceptance["maximum_loss"]),
        ("gradient_norm", metrics["gradient_norm"], acceptance["maximum_preclip_gradient_norm"]),
        ("peak_memory", peak / 2**20, acceptance["maximum_peak_allocated_mib"]),
        ("step_seconds", elapsed, acceptance["maximum_step_seconds"]),
    ):
        if not math.isfinite(value) or value > limit:
            failed.append(key)
    if trainer.optimizer.optimizer_state_dtypes != {torch.float32}:
        failed.append("optimizer_dtype")
    return {
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "effective_tokens_per_second": metrics["real_tokens"] / elapsed,
        "failed_checks": failed,
    }


def run_preflight(config_path: Path, run_root: Path, phase: str, lm: Path, cuda: Path, build: Path):
    config_path, run_root = config_path.resolve(), run_root.resolve()
    config = load_config(config_path)
    if phase not in {"continuous", "resume", "padded"}:
        raise ValueError("unknown preflight phase")
    data_root = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    if not run_root.is_relative_to(data_root / "artifacts/training_preflight"):
        raise ValueError("run directory must be under the configured training_preflight artifacts")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
    if dirty:
        raise ValueError("commit the preregistered implementation before GPU execution")
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    reference = None
    if phase != "continuous":
        reference = json.loads((run_root / "continuous/result.json").read_text())
        if reference["status"] != "passed":
            raise ValueError("continuous preflight has not passed")
        if phase == "padded":
            resumed = json.loads((run_root / "resume/result.json").read_text())
            if resumed["status"] != "passed":
                raise ValueError("GPU resume must pass before the padded control")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
    output = run_root / phase
    output.mkdir(exist_ok=False)
    write_json_once(
        output / "intent.json",
        {
            "schema_version": 1,
            "phase": phase,
            "code_commit": code_commit,
            "config_sha256": sha256_file(config_path),
            "resolved_config": config,
            "input_plan_sha256": config["input_plan_sha256"],
            "command": sys.argv,
        },
    )
    result = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "phase": phase,
        "status": "failed",
        "stage": "inputs",
        "steps": [],
        "failures": [],
        "limitation": "two real updates only; not 32/128 overfit, Agent success, or distributed resume",
    }
    try:
        samples, sampler, tokenizer = reconstruct_inputs(config, data_root)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        result["stage"] = "runtime"
        network, _, runtime = load_runtime(
            config["model_role"], lm.resolve(), cuda.resolve(), build.resolve()
        )
        packages = sorted(
            [d.metadata["Name"], d.version] for d in importlib.metadata.distributions()
        )
        environment = {
            "lock_sha256": sha256_file(ROOT / config["environment_lock"]),
            "packages": packages,
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
            "data_release_sha256": config["release_manifest_sha256"],
            "config_sha256": sha256_file(config_path),
            "environment_sha256": tree_digest(environment),
        }
        manifest = {
            "schema_version": 1,
            "purpose": config["purpose"],
            "phase": phase,
            "code_commit": code_commit,
            "dirty_diff_sha256": hashlib.sha256(b"").hexdigest(),
            "provenance": provenance,
            "runtime": runtime,
            "environment": environment,
            "input_plan_sha256": config["input_plan_sha256"],
            "command": sys.argv,
            "resolved_config": config,
        }
        validate_manifest(manifest)
        write_json_once(output / "manifest.json", manifest)
        result["manifest_sha256"] = sha256_file(output / "manifest.json")
        if reference is not None:
            prior_path = run_root / "continuous/manifest.json"
            prior_manifest = json.loads(prior_path.read_text())
            validate_manifest(prior_manifest)
            if (
                sha256_file(prior_path) != reference["manifest_sha256"]
                or prior_manifest["code_commit"] != code_commit
                or prior_manifest["provenance"] != provenance
            ):
                raise ValueError("cross-process run provenance mismatch")
        trainer, result["parameters"] = make_trainer(network, config)
        collator = PackedEpisodeCollator(
            tokenizer, max_pack_tokens=config["max_tokens"], align_to=config["alignment"]
        )
        rows = list(sampler)
        cursor = 0
        initial_model = tree_digest(network.state_dict())
        if phase == "resume":
            result["stage"] = "restore"
            cursor = load_checkpoint(
                run_root / "continuous/step1.pt", trainer, sampler, provenance=provenance
            )
            restored = training_fingerprints(trainer, sampler, cursor)
            result["restored_fingerprints"] = restored
            if cursor != 1 or restored != reference["step1_fingerprints"]:
                raise ValueError("restored training state mismatch")
            result["rng_probe"] = random_probe()
            if result["rng_probe"] != reference["rng_probe"]:
                raise ValueError("restored RNG stream mismatch")
        for row_index in range(cursor, config["steps"]):
            result["stage"] = f"update_{row_index + 1}"
            batch = collator.collate_tokenized([samples[index] for index in rows[row_index]])
            if phase == "padded":
                batch = padded_baseline(batch, config["max_tokens"])
            entry = timed_step(trainer, batch, config)
            entry["row_index"] = row_index
            entry["sample_ids"] = list(batch["sample_ids"])
            result["steps"].append(entry)
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "row": row_index,
                        **entry["metrics"],
                        "seconds": entry["elapsed_seconds"],
                        "failed_checks": entry["failed_checks"],
                    }
                ),
                flush=True,
            )
            if entry["failed_checks"]:
                result["failures"].extend(entry["failed_checks"])
                raise RuntimeError("preflight stop rule")
            trainer.optimizer.zero_grad(set_to_none=True)
            if phase == "continuous" and row_index == 0:
                result["stage"] = "save_step1"
                save_checkpoint(
                    output / "step1.pt", trainer, sampler, next_row=1, provenance=provenance
                )
                result["step1_fingerprints"] = training_fingerprints(trainer, sampler, 1)
                result["rng_probe"] = random_probe()
            elif phase == "padded" and row_index == 0:
                random_probe()  # same RNG progression as the continuous control
        result["stage"] = "final_state"
        final = training_fingerprints(trainer, sampler, config["steps"])
        result["final_fingerprints"] = final
        if final["model"] == initial_model:
            raise RuntimeError("no model parameter update")
        if phase == "resume":
            expected = reference["steps"][1]
            actual = result["steps"][0]
            if (
                final != reference["final_fingerprints"]
                or actual["metrics"] != expected["metrics"]
                or actual["sample_ids"] != expected["sample_ids"]
            ):
                raise ValueError("resumed next update differs from continuous reference")
        if phase != "padded":
            result["stage"] = "save_step2"
            result["checkpoint_sha256"] = save_checkpoint(
                output / "step2.pt", trainer, sampler, next_row=2, provenance=provenance
            )
        else:
            packed_seconds = sum(x["elapsed_seconds"] for x in reference["steps"])
            padded_seconds = sum(x["elapsed_seconds"] for x in result["steps"])
            result["packed_to_padded_throughput_ratio"] = padded_seconds / packed_seconds
            result["timing_limitation"] = (
                "n=2, includes first-step allocation/warmup; excludes checkpoint IO, not a speedup claim"
            )
        result["status"] = "passed"
        result["stage"] = "complete"
    except Exception as error:  # noqa: BLE001 -- preserve failures without leaking input text
        result["failures"].append(type(error).__name__)
        result["exception_sites"] = [
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)
        ]
        if torch.cuda.is_initialized():
            result["exception_memory"] = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
    finally:
        write_json_once(output / "result.json", result)
    return result
