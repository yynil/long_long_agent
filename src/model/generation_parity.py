"""Preregistered full-sequence versus recurrent generation diagnostics."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import jsonschema
import torch
import yaml

from src.data.governance import sha256_file

from .generation import GenerationConfig, generate
from .latent_v0 import RWKV7V0LatentControl, run_v0_latent_steps
from .runtime import ROOT, load_runtime
from .rwkv7_stateful import stateful_forward
from .slow_fast_state import SlowFastRWKVState

PROMPTS = (
    "System: You are a coding agent. Preserve tests and report evidence.\n\nUser: Inspect the repository to find why an empty configuration raises an exception. Run the failing test before editing.\n\nAssistant:",
    "System: Return a JSON tool call. Tools: execute(command: string), finish(summary: string).\n\nUser: Fix a regression in path normalization, including Unicode paths. Do not change unrelated files.\n\nAssistant:",
    "System: You are a careful coding agent.\n\nUser: A previous patch fixed parsing but broke round-trip serialization. Reproduce both cases and repair the implementation.\n\nAssistant:",
)


def comparison(reference, actual) -> dict:
    reference, actual = reference.float(), actual.float()
    log_p, log_q = reference.log_softmax(-1), actual.log_softmax(-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).flatten()
    return {
        "relative_rms": float(
            (actual - reference).square().mean().sqrt()
            / reference.square().mean().sqrt().clamp_min(1e-12)
        ),
        "mean_kl": float(kl.mean()),
        "p95_kl": float(torch.quantile(kl, 0.95)),
        "top1_agreement": float((reference.argmax(-1) == actual.argmax(-1)).float().mean()),
        "finite_logits": bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()),
        "positions": reference.shape[1],
    }


def metric_failures(metrics: dict, thresholds: dict) -> list[str]:
    failures = []
    for name in ("relative_rms", "mean_kl", "p95_kl"):
        if metrics[name] > thresholds[name + "_max"]:
            failures.append(name)
    if metrics["top1_agreement"] < thresholds["top1_agreement_min"]:
        failures.append("top1_agreement")
    if not metrics["finite_logits"]:
        failures.append("nonfinite")
    return failures


@torch.inference_mode()
def run_parity(
    config_path: Path,
    role: str,
    lm_tree: Path,
    cuda_tree: Path,
    build_root: Path,
    output: Path,
    *,
    diagnostic_only: bool = False,
):
    if type(diagnostic_only) is not bool:
        raise ValueError("diagnostic_only must be boolean")
    if output.exists():
        raise FileExistsError("refusing to overwrite parity evidence")
    config = yaml.safe_load(config_path.read_text())
    jsonschema.validate(
        config, json.loads((ROOT / "schemas/generation_parity.schema.json").read_text())
    )
    if role not in config["roles"]:
        raise ValueError("role was not preregistered")
    network, tokenizer, provenance = load_runtime(role, lm_tree, cuda_tree, build_root)
    torch.cuda.reset_peak_memory_stats()
    rows, failures = [], []
    started = time.monotonic()
    control = RWKV7V0LatentControl(network.args.n_embd).to("cuda")
    control.initialize_from_token_embeddings(network.emb.weight)
    for index, seed in enumerate(config["seeds"]):
        torch.manual_seed(seed)
        encoded = tokenizer.encode(PROMPTS[index % len(PROMPTS)])
        for length in config["prefix_lengths"]:
            ids = (encoded * (length // len(encoded) + 1))[:length]
            tokens = torch.tensor([ids], device="cuda")
            official = network(tokens)
            whole, _ = stateful_forward(network, tokens)
            parts, state = [], None
            for token in ids:
                logits, state = stateful_forward(
                    network, torch.tensor([[token]], device="cuda"), state=state
                )
                parts.append(logits)
            recurrent = torch.cat(parts, dim=1)
            decision = SlowFastRWKVState.begin_decision(
                state, decision_id=f"parity-{seed}-{length}"
            )
            k0 = run_v0_latent_steps(network, control, decision, steps=0)
            probe = torch.tensor([[ids[-1]]], device="cuda")
            anchor, _ = stateful_forward(network, probe, state=decision.fast)
            after_k0, _ = stateful_forward(network, probe, state=k0.decision.fast)
            k0_identical = bool(
                torch.equal(anchor, after_k0)
                and k0.decision is decision
                and k0.hidden.shape[1] == 0
            )
            row = {
                "seed": seed,
                "length": length,
                "official_vs_stateful_prefill": comparison(official, whole),
                "official_vs_token_recurrent": comparison(official, recurrent),
                "k0_identical": k0_identical,
            }
            for name in ("official_vs_stateful_prefill", "official_vs_token_recurrent"):
                for failure in metric_failures(row[name], config["thresholds"]):
                    failures.append(f"{seed}/{length}/{name}/{failure}")
            if not k0_identical:
                failures.append(f"{seed}/{length}/k0")
            rows.append(row)
            print(json.dumps(row), flush=True)
    generations = []
    for index, prompt in enumerate(PROMPTS):
        sampling = GenerationConfig(
            max_new_tokens=config["generation_tokens"],
            seed=config["seeds"][index % len(config["seeds"])],
        )
        first = generate(network, tokenizer, tokenizer.encode(prompt), sampling)
        second = generate(network, tokenizer, tokenizer.encode(prompt), sampling)
        identical = (
            first["token_ids"] == second["token_ids"]
            and first["stop_reason"] == second["stop_reason"]
        )
        if not identical:
            failures.append(f"generation/{index}/reproducibility")
        generations.append({"prompt_id": index, "first": first, "repeated_identical": identical})
    report = {
        "schema_version": 1,
        "status": "diagnostic_only"
        if diagnostic_only
        else ("passed" if not failures else "failed"),
        "thresholds_satisfied": not failures,
        "role": role,
        "provenance": provenance,
        "config_sha256": sha256_file(config_path),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_diff_sha256": __import__("hashlib")
        .sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT))
        .hexdigest(),
        "thresholds": config["thresholds"],
        "comparisons": rows,
        "generations": generations,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
