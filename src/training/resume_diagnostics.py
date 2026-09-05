"""Isolate native backward repeatability without save/restore between repeats."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.data.governance import sha256_file
from src.model.runtime import ROOT, load_runtime

from .checkpoint import load_checkpoint
from .episode_collator import PackedEpisodeCollator
from .preflight import (
    load_config,
    make_trainer,
    random_probe,
    reconstruct_inputs,
    tree_digest,
    validate_manifest,
    write_json_once,
)
from .sft_trainer import move_packed_batch


def gradient_differences(reference: dict, actual: dict) -> list[dict]:
    if set(reference) != set(actual):
        raise ValueError("gradient parameter coverage changed")
    differences = []
    for name, expected in reference.items():
        observed = actual[name]
        if expected is None or observed is None:
            if expected is not observed:
                raise ValueError("gradient presence changed")
            continue
        if expected.dtype != observed.dtype or expected.shape != observed.shape:
            raise ValueError("gradient specification changed")
        if not torch.isfinite(expected).all() or not torch.isfinite(observed).all():
            raise FloatingPointError("non-finite diagnostic gradient")
        count = int((expected != observed).sum())
        if count:
            differences.append(
                {
                    "parameter": name,
                    "different_elements": count,
                    "numel": expected.numel(),
                    "max_abs": float((expected.float() - observed.float()).abs().max()),
                }
            )
    return differences


def repeat_backward(reference_root: Path, output: Path, lm: Path, cuda: Path, build: Path):
    import subprocess

    if output.exists():
        raise FileExistsError("diagnostic report is immutable")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("commit the diagnostic protocol before GPU execution")
    config_path = ROOT / "configs/a0_training_preflight.yaml"
    config = load_config(config_path)
    manifest_path = reference_root / "continuous/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    report = {
        "schema_version": 1,
        "purpose": "native_backward_repeat_diagnostic_only",
        "status": "failed",
        "repeats": [],
        "optimizer_updates": 0,
        "checkpoint_loads": 0,
        "planned_repeats": 3,
        "reference_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    try:
        storage = yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]
        data_root = Path(storage["local_root"])
        if not output.resolve().is_relative_to(data_root / "artifacts/training_preflight"):
            raise ValueError("diagnostic artifact outside data root")
        samples, sampler, tokenizer = reconstruct_inputs(config, data_root)
        network, _, runtime = load_runtime("smoke", lm.resolve(), cuda.resolve(), build.resolve())
        provenance = manifest["provenance"]
        for key in ("checkpoint_sha256", "model_code_sha256", "tokenizer_sha256"):
            if runtime[key] != provenance[key]:
                raise ValueError("diagnostic runtime differs from reference")
        if report["config_sha256"] != provenance["config_sha256"]:
            raise ValueError("diagnostic configuration differs from reference")
        trainer, _ = make_trainer(network, config)
        cursor = load_checkpoint(
            reference_root / "continuous/step1.pt", trainer, sampler, provenance=provenance
        )
        report["checkpoint_loads"] = 1
        random_probe()
        collator = PackedEpisodeCollator(tokenizer, max_pack_tokens=8192, align_to=16)
        row = list(sampler)[cursor]
        batch = move_packed_batch(collator.collate_tokenized([samples[i] for i in row]), "cuda")
        report["sample_ids"] = list(batch["sample_ids"])
        report["real_tokens"] = int(batch["cu_seqlens"][-1])
        before_model = tree_digest(network.state_dict())
        before_optimizer = tree_digest(trainer.optimizer.state_dict())
        reference_gradients = None
        for index in range(report["planned_repeats"]):
            network.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            loss = trainer.compute_loss(batch)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("non-finite diagnostic loss")
            loss.total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                list(network.parameters()), 1.0, error_if_nonfinite=True
            )
            actual = {
                name: None if p.grad is None else p.grad.detach().cpu().clone()
                for name, p in network.named_parameters()
            }
            differences = (
                []
                if reference_gradients is None
                else gradient_differences(reference_gradients, actual)
            )
            report["repeats"].append(
                {
                    "repeat": index,
                    "loss": float(loss.total.detach()),
                    "gradient_norm": float(norm),
                    "gradient_sha256": tree_digest(actual),
                    "differences_from_first": differences,
                    "different_tensors": len(differences),
                    "different_elements": sum(row["different_elements"] for row in differences),
                }
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in report["repeats"][-1].items()
                        if k != "differences_from_first"
                    }
                ),
                flush=True,
            )
            if reference_gradients is None:
                reference_gradients = actual
            del actual, loss
        trainer.optimizer.zero_grad(set_to_none=True)
        report["model_unchanged"] = tree_digest(network.state_dict()) == before_model
        report["optimizer_unchanged"] = (
            tree_digest(trainer.optimizer.state_dict()) == before_optimizer
        )
        if not report["model_unchanged"] or not report["optimizer_unchanged"]:
            raise RuntimeError("diagnostic mutated training weights or optimizer")
        report["native_gradient_variation_observed"] = any(
            r["different_tensors"] for r in report["repeats"]
        )
        report["status"] = "diagnostic_complete"
    except Exception as error:  # noqa: BLE001 -- report type only, never trajectory text
        report["exception_type"] = type(error).__name__
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_once(output, report)
    return report
