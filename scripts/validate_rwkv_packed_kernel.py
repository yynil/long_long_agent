#!/usr/bin/env python3
"""Compile and validate reset-aware RWKV-7 CUDA operators."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import torch


def relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    error = (left.float() - right.float()).square().mean().sqrt()
    scale = left.float().square().mean().sqrt().clamp_min(1e-12)
    return float((error / scale).detach())


def concatenate_grads(parts: list[torch.Tensor], lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([part.grad[:, :length] for part, length in zip(parts, lengths)], dim=1)


def combine_outputs(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(value, tuple):
        return torch.stack(value).sum(dim=0)
    return value


def validate_shift_operator(
    operation: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]],
    inputs: list[torch.Tensor],
    shared: list[torch.Tensor],
    starts: torch.Tensor,
    lengths: tuple[int, ...],
) -> tuple[float, float, float]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)

    packed_inputs = [value.detach().clone().requires_grad_(True) for value in inputs]
    packed_shared = [value.detach().clone().requires_grad_(True) for value in shared]
    packed_result = combine_outputs(operation(*packed_inputs, *packed_shared, starts))
    packed_weight = torch.randn_like(packed_result)
    (packed_result * packed_weight).sum().backward()

    separate_inputs: list[list[torch.Tensor]] = [[] for _ in inputs]
    separate_shared = [value.detach().clone().requires_grad_(True) for value in shared]
    separate_results = []
    for begin, end in pairwise(offsets):
        part_inputs = [
            value[:, begin:end].detach().clone().requires_grad_(True) for value in inputs
        ]
        for destination, value in zip(separate_inputs, part_inputs):
            destination.append(value)
        part_starts = torch.zeros((1, end - begin), dtype=torch.uint8, device=starts.device)
        part_starts[:, 0] = 1
        separate_results.append(
            combine_outputs(operation(*part_inputs, *separate_shared, part_starts))
        )
    separate_result = torch.cat(separate_results, dim=1)
    (separate_result * packed_weight).sum().backward()

    input_grad_error = max(
        relative_rms(packed.grad, concatenate_grads(parts, lengths))
        for packed, parts in zip(packed_inputs, separate_inputs)
    )
    shared_grad_error = max(
        relative_rms(packed.grad, separate.grad)
        for packed, separate in zip(packed_shared, separate_shared)
    )
    return (
        relative_rms(packed_result, separate_result),
        input_grad_error,
        shared_grad_error,
    )


def validate_wkv(
    operation: Callable[..., torch.Tensor],
    starts: torch.Tensor,
    lengths: tuple[int, ...],
) -> tuple[float, list[float]]:
    total = sum(lengths)
    base = [torch.randn((1, total, 64), device="cuda", dtype=torch.bfloat16) for _ in range(6)]
    packed = [value.detach().clone().requires_grad_(True) for value in base]
    packed_result = operation(*packed, starts)
    output_weight = torch.randn_like(packed_result)
    (packed_result * output_weight).sum().backward()

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    separate: list[list[torch.Tensor]] = [[] for _ in base]
    real_outputs = []
    for begin, end in pairwise(offsets):
        part_values = []
        for destination, value in zip(separate, base):
            padded = torch.zeros((1, 16, 64), device="cuda", dtype=torch.bfloat16)
            padded[:, : end - begin] = value[:, begin:end]
            padded.requires_grad_(True)
            destination.append(padded)
            part_values.append(padded)
        part_starts = torch.zeros((1, 16), dtype=torch.uint8, device="cuda")
        part_starts[:, 0] = 1
        real_outputs.append(operation(*part_values, part_starts)[:, : end - begin])
    separate_result = torch.cat(real_outputs, dim=1)
    (separate_result * output_weight).sum().backward()

    separate_grads = [concatenate_grads(parts, lengths) for parts in separate]
    return (
        relative_rms(packed_result, separate_result),
        [relative_rms(value.grad, reference) for value, reference in zip(packed, separate_grads)],
    )


def cuda_elapsed(operation: Callable[[], None], repeats: int) -> float:
    measurements = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end))
    return statistics.median(measurements)


def benchmark_full_model(network: torch.nn.Module, vocab_size: int) -> dict[str, float]:
    lengths = (16, 32, 64, 128)
    pack_rows = 4
    tokens_per_row = sum(lengths)
    real_tokens = pack_rows * tokens_per_row
    packed_tokens = torch.randint(0, vocab_size, (pack_rows, tokens_per_row), device="cuda")
    packed_starts = torch.zeros_like(packed_tokens, dtype=torch.uint8)
    offset = 0
    for length in lengths:
        packed_starts[:, offset] = 1
        offset += length

    padded_tokens = torch.zeros(
        (pack_rows * len(lengths), max(lengths)), dtype=torch.long, device="cuda"
    )
    padded_starts = torch.zeros_like(padded_tokens, dtype=torch.uint8)
    padded_valid = torch.zeros_like(padded_tokens, dtype=torch.bool)
    for pack_row in range(pack_rows):
        offset = 0
        for segment, length in enumerate(lengths):
            padded_row = pack_row * len(lengths) + segment
            padded_tokens[padded_row, :length] = packed_tokens[pack_row, offset : offset + length]
            padded_starts[padded_row, 0] = 1
            padded_valid[padded_row, :length] = True
            offset += length

    def packed_forward() -> None:
        with torch.no_grad():
            network._forward_features(packed_tokens, packed_starts)

    def padded_forward() -> None:
        with torch.no_grad():
            network._forward_features(padded_tokens, padded_starts)

    packed_forward()
    padded_forward()
    packed_ms = cuda_elapsed(packed_forward, 5)
    padded_ms = cuda_elapsed(padded_forward, 5)

    def packed_train_step() -> None:
        network.zero_grad(set_to_none=True)
        hidden = network._forward_features(packed_tokens, packed_starts)
        hidden.float().square().mean().backward()

    def padded_train_step() -> None:
        network.zero_grad(set_to_none=True)
        hidden = network._forward_features(padded_tokens, padded_starts)
        hidden[padded_valid].float().square().mean().backward()

    packed_train_step()
    padded_train_step()
    torch.cuda.reset_peak_memory_stats()
    packed_train_ms = cuda_elapsed(packed_train_step, 3)
    packed_peak = torch.cuda.max_memory_allocated() / (1024**3)
    torch.cuda.reset_peak_memory_stats()
    padded_train_ms = cuda_elapsed(padded_train_step, 3)
    padded_peak = torch.cuda.max_memory_allocated() / (1024**3)
    network.zero_grad(set_to_none=True)

    return {
        "benchmark_real_tokens": float(real_tokens),
        "benchmark_packed_storage_tokens": float(packed_tokens.numel()),
        "benchmark_padded_storage_tokens": float(padded_tokens.numel()),
        "benchmark_forward_packed_ms": packed_ms,
        "benchmark_forward_padded_ms": padded_ms,
        "benchmark_forward_speedup": padded_ms / packed_ms,
        "benchmark_train_packed_ms": packed_train_ms,
        "benchmark_train_padded_ms": padded_train_ms,
        "benchmark_train_speedup": padded_train_ms / packed_train_ms,
        "benchmark_train_packed_peak_gib": packed_peak,
        "benchmark_train_padded_peak_gib": padded_peak,
    }


def validate_full_model(
    model_module: object, checkpoint: Path, *, benchmark: bool
) -> dict[str, float]:
    args = SimpleNamespace(
        n_layer=24,
        n_embd=1024,
        vocab_size=65536,
        ctx_len=8192,
        head_size=64,
        dim_att=1024,
        dim_ffn=3584,
        grad_cp=0,
        my_testing="x070",
        decay_lora_rank=64,
        aaa_lora_rank=64,
        mv_lora_rank=32,
        gate_lora_rank=128,
    )
    network = model_module.RWKV(args)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(state_dict, strict=True)
    del state_dict
    network = network.to(device="cuda", dtype=torch.bfloat16).eval()

    lengths = (5, 11)
    tokens = torch.randint(0, args.vocab_size, (1, sum(lengths)), device="cuda")
    starts = torch.zeros_like(tokens, dtype=torch.uint8)
    starts[:, 0] = 1
    starts[:, lengths[0]] = 1
    with torch.no_grad():
        packed = network._forward_features(tokens, starts)
        separate = []
        offset = 0
        for length in lengths:
            padded = torch.zeros((1, 16), dtype=tokens.dtype, device="cuda")
            padded[:, :length] = tokens[:, offset : offset + length]
            part_starts = torch.zeros_like(padded, dtype=torch.uint8)
            part_starts[:, 0] = 1
            separate.append(network._forward_features(padded, part_starts)[:, :length])
            offset += length
        unpacked = torch.cat(separate, dim=1)

        perturbed_tokens = tokens.clone()
        perturbed_tokens[:, : lengths[0]] = torch.randint(
            0, args.vocab_size, (1, lengths[0]), device="cuda"
        )
        perturbed = network._forward_features(perturbed_tokens, starts)

    result = {
        "full_model_forward_relative_rms": relative_rms(packed, unpacked),
        "full_model_second_segment_isolation_relative_rms": relative_rms(
            packed[:, lengths[0] :], perturbed[:, lengths[0] :]
        ),
    }
    if benchmark:
        result.update(benchmark_full_model(network, args.vocab_size))
    del network
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwkv-worktree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    train_dir = args.rwkv_worktree.resolve() / "RWKV-v7" / "train_temp"
    os.chdir(train_dir)
    sys.path.insert(0, str(train_dir / "src"))
    model = importlib.import_module("model")

    torch.manual_seed(20260904)
    lengths = (3, 5, 8)
    starts = torch.zeros((1, sum(lengths)), dtype=torch.uint8, device="cuda")
    offset = 0
    for length in lengths:
        starts[:, offset] = 1
        offset += length

    x = torch.randn((1, 16, 64), device="cuda", dtype=torch.bfloat16)
    mixes = [torch.randn(64, device="cuda", dtype=torch.bfloat16) for _ in range(6)]
    tmix = validate_shift_operator(
        model.tmix_mix6_bf16_v5,
        [x],
        mixes,
        starts,
        lengths,
    )

    cmix_shared = [
        torch.randn(64, device="cuda", dtype=torch.bfloat16),
        torch.randn((256, 64), device="cuda", dtype=torch.bfloat16),
        torch.randn((64, 256), device="cuda", dtype=torch.bfloat16),
    ]
    cmix = validate_shift_operator(
        lambda value, x_k, key, output, mask: model._CmixLayerV2Fn.apply(
            value, x_k, key, output, mask
        ),
        [x],
        cmix_shared,
        starts,
        lengths,
    )
    wkv = validate_wkv(model.RWKV7_CLAMPW_CUDA, starts, lengths)

    result = {
        "device": torch.cuda.get_device_name(0),
        "tmix_forward_relative_rms": tmix[0],
        "tmix_input_grad_relative_rms": tmix[1],
        "tmix_shared_grad_relative_rms": tmix[2],
        "cmix_forward_relative_rms": cmix[0],
        "cmix_input_grad_relative_rms": cmix[1],
        "cmix_shared_grad_relative_rms": cmix[2],
        "wkv_forward_relative_rms": wkv[0],
        "wkv_r_grad_relative_rms": wkv[1][0],
        "wkv_w_grad_relative_rms": wkv[1][1],
        "wkv_k_grad_relative_rms": wkv[1][2],
        "wkv_v_grad_relative_rms": wkv[1][3],
        "wkv_a_grad_relative_rms": wkv[1][4],
        "wkv_b_grad_relative_rms": wkv[1][5],
    }
    if args.checkpoint is not None:
        result.update(
            validate_full_model(model, args.checkpoint.resolve(), benchmark=args.benchmark)
        )
    parity_errors = [value for key, value in result.items() if "relative_rms" in key]
    if max(parity_errors) > 0.02:
        raise RuntimeError(f"packed CUDA parity failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
