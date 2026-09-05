"""Read-only precision/shape controls on real first-layer projection operands.

Higher precision uses the same already-quantized BF16 operands. This isolates
arithmetic, not checkpoint quantization. Nothing changes deployed model math.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.governance import sha256_file

from .generation_parity import PROMPTS, run_parity
from .generation_parity import comparison as logit_comparison
from .runtime import ROOT, load_runtime
from .rwkv7_stateful import stateful_forward


def error_metrics(reference, actual) -> dict:
    if reference.shape != actual.shape or reference.numel() == 0:
        raise ValueError("nonempty matching shapes required")
    reference, actual = reference.double(), actual.double()
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("nonfinite precision probe")
    difference = actual - reference
    return {
        "unequal_fraction": float((reference != actual).double().mean()),
        "absolute_max": float(difference.abs().max()),
        "relative_rms": float(
            difference.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-30)
        ),
    }


@contextmanager
def precision_controls(reduced_bf16: bool):
    if type(reduced_bf16) is not bool:
        raise ValueError("reduced_bf16 must be boolean")
    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision, matmul.allow_bf16_reduced_precision_reduction
    try:
        matmul.fp32_precision = "ieee"
        matmul.allow_bf16_reduced_precision_reduction = reduced_bf16
        yield
    finally:
        matmul.fp32_precision, matmul.allow_bf16_reduced_precision_reduction = previous


@torch.inference_mode()
def projection_controls(x, weight, lengths=(16, 32, 64, 128)) -> list[dict]:
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.device != weight.device
    ):
        raise ValueError("expected matching two-dimensional BF16 operands")
    if not lengths or any(type(n) is not int or not 0 < n <= len(x) for n in lengths):
        raise ValueError("invalid reference lengths")
    reference = F.linear(x[:1].double(), weight.double())
    rounded_reference = reference.to(torch.bfloat16)
    rows = []
    modes = (
        ("bf16_reduction_on", torch.bfloat16, True),
        ("bf16_reduction_off", torch.bfloat16, False),
        ("fp32_ieee", torch.float32, False),
        ("fp32_ieee_then_bf16", torch.float32, False),
        ("fp64", torch.float64, False),
    )
    for mode, dtype, reduced in modes:
        with precision_controls(reduced):
            values, matrix = x.to(dtype), weight.to(dtype)
            single = F.linear(values[:1], matrix)
            if mode == "fp32_ieee_then_bf16":
                single = single.to(torch.bfloat16)
            for length in lengths:
                batched = F.linear(values[:length], matrix)[:1]
                padded = F.linear(F.pad(values[:1], (0, 0, 0, length - 1)), matrix)[:1]
                if mode == "fp32_ieee_then_bf16":
                    batched, padded = batched.to(torch.bfloat16), padded.to(torch.bfloat16)
                rows.append(
                    {
                        "mode": mode,
                        "matrix_rows": length,
                        "single_vs_batch": error_metrics(batched, single),
                        "padded_vs_batch": error_metrics(batched, padded),
                        "single_vs_fp64": error_metrics(reference, single),
                        "batch_vs_fp64": error_metrics(reference, batched),
                        "single_vs_fp64_rounded_bf16": error_metrics(rounded_reference, single),
                        "batch_vs_fp64_rounded_bf16": error_metrics(rounded_reference, batched),
                    }
                )
    return rows


def _tensor_sha256(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy()).hexdigest()


def diagnose_whole_model_precision(role, lm, cuda, build, output):
    """Test reduced-precision-off against old thresholds, never reclassify the old run."""
    with precision_controls(False):
        return run_parity(
            ROOT / "configs/generation_parity.yaml",
            role,
            lm,
            cuda,
            build,
            output,
            diagnostic_only=True,
        )


@contextmanager
def diagnostic_projection_accumulation(dtype):
    """Process-local readout/projection arithmetic probe; never a deployment option."""
    from . import rwkv7_stateful as recurrent

    if torch.is_grad_enabled() or dtype not in (torch.float32, torch.float64):
        raise ValueError("projection precision control requires inference and FP32/FP64")
    original = recurrent.inference_linear, recurrent.inference_matmul
    converted = {}

    def matrix(value):
        if id(value) not in converted:
            converted[id(value)] = value.to(dtype)
        return converted[id(value)]

    def linear(module, x):
        bias = None if module.bias is None else matrix(module.bias)
        return F.linear(x.to(dtype), matrix(module.weight), bias).to(x.dtype)

    def matmul(x, weight):
        return (x.to(dtype) @ matrix(weight)).to(x.dtype)

    try:
        recurrent.inference_linear, recurrent.inference_matmul = linear, matmul
        yield
    finally:
        recurrent.inference_linear, recurrent.inference_matmul = original
        converted.clear()


@torch.inference_mode()
def diagnose_recurrent_precision(role, lm, cuda, build, output):
    if output.exists():
        raise FileExistsError("refusing to overwrite recurrence precision evidence")
    network, tokenizer, provenance = load_runtime(role, lm, cuda, build)
    rows = []
    for dtype in (torch.float32, torch.float64):
        with precision_controls(False), diagnostic_projection_accumulation(dtype):
            for length in (16, 64):
                ids = tokenizer.encode(PROMPTS[0] * 4)[:length]
                tokens = torch.tensor([ids], device="cuda")
                whole, whole_state = stateful_forward(network, tokens)
                parts, state = [], None
                for token in tokens[0]:
                    part, state = stateful_forward(network, token.reshape(1, 1), state=state)
                    parts.append(part)
                state_metrics = {}
                for field in ("wkv_matrix", "time_mix_previous_x", "channel_mix_previous_x"):
                    metrics = [
                        error_metrics(getattr(a, field), getattr(b, field))
                        for a, b in zip(whole_state.layers, state.layers, strict=True)
                    ]
                    state_metrics[field] = {k: max(m[k] for m in metrics) for k in metrics[0]}
                row = {
                    "projection_accumulation": str(dtype),
                    "output_activation_dtype": "torch.bfloat16",
                    "length": length,
                    "logits": logit_comparison(whole, torch.cat(parts, dim=1)),
                    "states": state_metrics,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    result = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "role": role,
        "provenance": provenance,
        "diagnostic_code_sha256": sha256_file(Path(__file__)),
        "comparison": "stateful_whole_vs_stateful_single_not_official_acceptance",
        "operand_policy": "existing_bf16_weights_upcast_projections_then_round_output_bf16",
        "matmul_policy": "fp32_ieee_bf16_reduced_reduction_false",
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


@torch.inference_mode()
def diagnose_precision(role: str, lm: Path, cuda: Path, build: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite precision evidence")
    network, tokenizer, provenance = load_runtime(role, lm, cuda, build)
    matmul = torch.backends.cuda.matmul
    original_flags = {
        "fp32_precision": matmul.fp32_precision,
        "bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
    }
    first_layer = network.blocks[0].att
    captured, hooks = {}, []
    phase = "official"

    def capture(name):
        def hook(module, inputs, result):
            captured[(phase, name)] = inputs[0].detach().clone(), result.detach().clone()

        return hook

    try:
        for name in ("receptance", "key", "value"):
            hooks.append(getattr(first_layer, name).register_forward_hook(capture(name)))
        ids = tokenizer.encode(PROMPTS[0] * 4)[:128]
        tokens = torch.tensor([ids], device="cuda")
        network(tokens)
        phase = "single"
        stateful_forward(network, tokens[:, :1])
    finally:
        for hook in hooks:
            hook.remove()
    results = []
    for name in ("receptance", "key", "value"):
        full_input, full_output = captured[("official", name)]
        single_input, single_output = captured[("single", name)]
        weight = getattr(first_layer, name).weight
        if not torch.equal(full_input[:, :1], single_input):
            raise ValueError("first projection inputs differ before matrix multiplication")
        rows = projection_controls(full_input[0], weight)
        results.append(
            {
                "layer": 0,
                "projection": name,
                "same_input_exact": True,
                "input_sha256": _tensor_sha256(full_input),
                "weight_sha256": _tensor_sha256(weight),
                "original_first_output": error_metrics(full_output[:, :1], single_output),
                "controls": rows,
            }
        )
        print(json.dumps({"projection": name, "status": "diagnosed"}), flush=True)
    if original_flags != {
        "fp32_precision": matmul.fp32_precision,
        "bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
    }:
        raise RuntimeError("diagnostic precision flags were not restored")
    result = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "role": role,
        "provenance": provenance,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "diagnostic_code_sha256": sha256_file(Path(__file__)),
        "prompt_sha256": hashlib.sha256(PROMPTS[0].encode()).hexdigest(),
        "original_flags": original_flags,
        "flags_restored": True,
        "operand_policy": "identical_existing_bf16_values_upcast_only",
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result
