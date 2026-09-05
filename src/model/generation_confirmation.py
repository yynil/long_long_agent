"""ADR-019 independent engineering confirmation, not Agent outcome evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import jsonschema
import torch
import yaml

from src.data.governance import sha256_file
from src.data.util import canonical_json

from .generation import GenerationConfig, generate
from .generation_parity import comparison, metric_failures
from .inference_math import diagnostic_reference_rows
from .latent_v0 import RWKV7V0LatentControl, run_v0_latent_steps
from .runtime import ROOT, load_runtime
from .rwkv7_stateful import stateful_forward
from .slow_fast_state import SlowFastRWKVState


def load_confirmation_config(path: Path):
    config = yaml.safe_load(path.read_text())
    schema = json.loads((ROOT / "schemas/generation_confirmation.schema.json").read_text())
    jsonschema.validate(config, schema)
    if len({p["id"] for p in config["prompts"]}) != len(config["prompts"]):
        raise ValueError("duplicate confirmation prompt ID")
    if any(n <= config["probe_tokens"] for ns in config["native_lengths"].values() for n in ns):
        raise ValueError("native windows must have a nonempty prefix")
    return config


def probe_tokens(tokenizer, prompt, length, seed):
    """Deterministic synthetic shape probe; never enters canonical training data."""
    beginning = tokenizer.encode(
        "System: You are a coding agent.\n\nUser: " + prompt["task"] + "\n"
    )
    ending = tokenizer.encode("\nUser: " + prompt["task"] + "\n\n" + prompt["continuation"])
    if len(beginning) + len(ending) >= length:
        # Strict short-window probes deliberately use a token prefix, not an Agent episode.
        return (beginning + ending)[:length]
    records = []
    for index in range(length // 8 + 1):
        code = hashlib.sha256(f"{seed}/{prompt['id']}/{index}".encode()).hexdigest()[:16]
        records.append(f"Observation {index:05d}: inspected fixture {code}; no change applied.\n")
    filler = tokenizer.encode("".join(records))
    remaining = length - len(beginning) - len(ending)
    if len(filler) < remaining:
        raise ValueError("insufficient deterministic filler")
    return beginning + filler[:remaining] + ending


def native_metrics(reference, actual, thresholds):
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("nonfinite confirmation logits")
    metrics = comparison(reference, actual)
    confidence, selected = reference.float().softmax(-1).max(-1)
    confident = confidence >= thresholds["confident_probability_min"]
    changed = selected != actual.argmax(-1)
    metrics["confident_positions"] = int(confident.sum())
    metrics["confident_argmax_changes"] = int((confident & changed).sum())
    failures = metric_failures(metrics, thresholds)
    if metrics["confident_argmax_changes"] > thresholds["confident_argmax_changes_max"]:
        failures.append("confident_argmax_changes")
    return metrics, failures


def states_exact(first, second):
    return first.spec == second.spec and all(
        torch.equal(getattr(a, name), getattr(b, name))
        for a, b in zip(first.layers, second.layers, strict=True)
        for name in ("wkv_matrix", "time_mix_previous_x", "channel_mix_previous_x")
    )


def _fingerprint(ids):
    return hashlib.sha256(canonical_json(ids).encode()).hexdigest()


@torch.inference_mode()
def run_confirmation(config_path, role, lm, cuda, build, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite independent confirmation")
    config = load_confirmation_config(config_path)
    if role not in config["roles"]:
        raise ValueError("role not preregistered")
    # No numeric/deployment override: confirm the original native runtime.
    matmul = torch.backends.cuda.matmul
    if matmul.fp32_precision != "none" or not matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError("confirmation requires the original matmul flags")
    torch.manual_seed(config["seed"])
    network, tokenizer, provenance = load_runtime(role, lm, cuda, build)
    if max(config["native_lengths"][role]) > network.args.ctx_len:
        raise ValueError("confirmation exceeds pinned checkpoint context")
    control = RWKV7V0LatentControl(network.args.n_embd).to("cuda")
    control.initialize_from_token_embeddings(network.emb.weight)
    report = {
        "schema_version": 1,
        "protocol": "ADR-019_independent_engineering_confirmation_v1",
        "status": "running",
        "role": role,
        "config_sha256": sha256_file(config_path),
        "provenance": provenance,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "strict_cases": [],
        "native_cases": [],
        "generations": [],
        "failures": [],
        "planned_case_counts": {
            "strict": len(config["prompts"]) * len(config["strict_lengths"]),
            "native": len(config["prompts"]) * len(config["native_lengths"][role]),
            "generations": len(config["prompts"]) * 2,
        },
        "limitations": "Three synthetic prompts; next-token/repeatability checks, not executable Agent success or G1.",
    }
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for prompt in config["prompts"]:
            for length in config["strict_lengths"]:
                ids = probe_tokens(tokenizer, prompt, length, config["seed"])
                if len(ids) != length:
                    raise ValueError("short confirmation probe could not fill target length")
                tokens = torch.tensor([ids], device="cuda")
                starts = torch.zeros_like(tokens, dtype=torch.uint8)
                starts[:, length // 3] = 1
                reference = network(tokens, sequence_start_mask=starts)
                whole, whole_state = stateful_forward(network, tokens, sequence_start_mask=starts)
                parts, state = [], None
                with diagnostic_reference_rows(length):
                    for index in range(length):
                        part, state = stateful_forward(
                            network,
                            tokens[:, index : index + 1],
                            state=state,
                            sequence_start_mask=starts[:, index : index + 1],
                        )
                        parts.append(part)
                row = {
                    "prompt_id": prompt["id"],
                    "length": length,
                    "tokens_sha256": _fingerprint(ids),
                    "reset_position": length // 3,
                    "official_prefill_exact": bool(torch.equal(reference, whole)),
                    "matched_shape_logits_exact": bool(
                        torch.equal(reference, torch.cat(parts, dim=1))
                    ),
                    "all_three_states_exact": states_exact(whole_state, state),
                }
                report["strict_cases"].append(row)
                for key in (
                    "official_prefill_exact",
                    "matched_shape_logits_exact",
                    "all_three_states_exact",
                ):
                    if not row[key]:
                        report["failures"].append(f"strict/{prompt['id']}/{length}/{key}")
                del reference, whole, whole_state, parts, state
                print(json.dumps({"strict": row}), flush=True)
            for length in config["native_lengths"][role]:
                ids = probe_tokens(tokenizer, prompt, length, config["seed"])
                tokens = torch.tensor([ids], device="cuda")
                tail = config["probe_tokens"]
                reference = network(tokens)[:, -tail:].clone()
                state = None
                prefix = length - tail
                for offset in range(0, prefix, config["native_prefill_chunk"]):
                    _, state = stateful_forward(
                        network,
                        tokens[:, offset : min(prefix, offset + config["native_prefill_chunk"])],
                        state=state,
                        return_logits=False,
                    )
                parts = []
                for index in range(prefix, length):
                    logits, state = stateful_forward(
                        network, tokens[:, index : index + 1], state=state
                    )
                    parts.append(logits)
                metrics, failures = native_metrics(
                    reference, torch.cat(parts, dim=1), config["native_thresholds"]
                )
                decision = SlowFastRWKVState.begin_decision(
                    state, decision_id=f"confirmation/{prompt['id']}/{length}"
                )
                k0 = run_v0_latent_steps(network, control, decision, steps=0)
                anchor, _ = stateful_forward(network, tokens[:, -1:], state=decision.fast)
                after, _ = stateful_forward(network, tokens[:, -1:], state=k0.decision.fast)
                k0_identical = bool(k0.decision is decision and torch.equal(anchor, after))
                if not k0_identical:
                    failures.append("k0")
                row = {
                    "prompt_id": prompt["id"],
                    "length": length,
                    "tokens_sha256": _fingerprint(ids),
                    "probe_positions": tail,
                    "metrics": metrics,
                    "k0_identical": k0_identical,
                }
                report["native_cases"].append(row)
                report["failures"].extend(
                    f"native/{prompt['id']}/{length}/{key}" for key in failures
                )
                del reference, parts, state, decision, k0, anchor, after
                print(json.dumps({"native": row}), flush=True)
            prompt_ids = tokenizer.encode(
                "System: You are a coding agent.\n\nUser: " + prompt["task"] + "\n\nAssistant:"
            )
            for temperature in (0.0, 0.7):
                sampling = GenerationConfig(
                    max_new_tokens=config["generation_tokens"],
                    temperature=temperature,
                    seed=config["seed"],
                    max_seconds=config["generation_max_seconds"],
                )
                first = generate(network, tokenizer, prompt_ids, sampling)
                second = generate(network, tokenizer, prompt_ids, sampling)
                identical = (
                    first["token_ids"] == second["token_ids"]
                    and first["stop_reason"] == second["stop_reason"]
                )
                row = {
                    "prompt_id": prompt["id"],
                    "temperature": temperature,
                    "identical": identical,
                    "tokens_sha256": _fingerprint(first["token_ids"]),
                    "generated_tokens": len(first["token_ids"]),
                    "stop_reason": first["stop_reason"],
                    "utf8_valid": first["utf8_valid"],
                }
                report["generations"].append(row)
                if not identical or first["stop_reason"] == "time_budget":
                    report["failures"].append(f"generation/{prompt['id']}/{temperature}")
        report["status"] = "passed" if not report["failures"] else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["failures"].append("execution_exception/" + type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return report
