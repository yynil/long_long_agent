#!/usr/bin/env python3
"""Compile and validate reset-aware RWKV-7 state-passing CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

W_SCALE = -0.6065306597


def relative_rms(reference: torch.Tensor, actual: torch.Tensor) -> float:
    error = (reference.float() - actual.float()).square().mean().sqrt()
    scale = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float((error / scale).detach())


def state_passing_reference(
    s0: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    sequence_start_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straightforward float32 recurrence with an update-before-read reset."""
    state = s0
    outputs = []
    decay = torch.exp(W_SCALE / (1.0 + torch.exp(-w)))
    for timestep in range(r.shape[1]):
        reset = sequence_start_mask[:, timestep, None, None, None].bool()
        state = state.masked_fill(reset, 0.0)
        state_a = torch.einsum("bhij,bhj->bhi", state, a[:, timestep])
        state = (
            state * decay[:, timestep, :, None, :]
            + torch.einsum("bhi,bhj->bhij", state_a, b[:, timestep])
            + torch.einsum("bhi,bhj->bhij", v[:, timestep], k[:, timestep])
        )
        outputs.append(torch.einsum("bhij,bhj->bhi", state, r[:, timestep]))
    return torch.stack(outputs, dim=1), state


def compile_extension(
    worktree: Path, build_root: Path, dtype: torch.dtype, head_size: int, chunk_len: int
) -> None:
    source_root = worktree / "rwkv7_fast_fused"
    build_directory = build_root / (
        f"state_passing_varlen_{'fp32' if dtype == torch.float32 else 'bf16'}"
        f"_n{head_size}_c{chunk_len}"
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    flags = [
        "-res-usage",
        f"-D_N_={head_size}",
        f"-D_CHUNK_LEN_={chunk_len}",
        "--use_fast_math",
        "-O3",
        "-Xptxas=-O3",
        "--extra-device-vectorization",
    ]
    extra_cflags: list[str] = []
    if dtype == torch.float32:
        flags.append("-D_FP32_")
        extra_cflags.append("-D_FP32_")
    load(
        name="rwkv7_statepassing_clampw",
        sources=[
            str(source_root / "cuda/rwkv7_statepassing_clampw.cu"),
            str(source_root / "cuda/rwkv7_statepassing_clampw.cpp"),
        ],
        build_directory=str(build_directory),
        is_python_module=False,
        verbose=False,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=flags,
    )


def make_cuda_operation(dtype: torch.dtype, chunk_len: int):
    class StatePassingOperation(torch.autograd.Function):
        @staticmethod
        def forward(ctx, s0, r, w, k, v, a, b, sequence_start_mask):
            batch, timesteps, heads, head_size = r.shape
            if timesteps % chunk_len:
                raise ValueError("timesteps must be divisible by chunk_len")
            y = torch.empty_like(r)
            s_t = torch.empty_like(s0)
            checkpoints = torch.empty(
                batch,
                heads,
                timesteps // chunk_len,
                head_size,
                head_size,
                dtype=torch.float32,
                device=r.device,
            )
            state_a = torch.empty(
                batch,
                timesteps,
                heads,
                head_size,
                dtype=torch.float32,
                device=r.device,
            )
            torch.ops.rwkv7_statepassing_clampw.forward(
                s0,
                r,
                w,
                k,
                v,
                a,
                b,
                sequence_start_mask,
                y,
                s_t,
                checkpoints,
                state_a,
            )
            ctx.save_for_backward(s0, r, w, k, v, a, b, sequence_start_mask, checkpoints, state_a)
            return y, s_t

        @staticmethod
        def backward(ctx, grad_y, grad_s_t):
            s0, r, w, k, v, a, b, starts, checkpoints, state_a = ctx.saved_tensors
            grad_y = grad_y.contiguous()
            grad_s_t = grad_s_t.contiguous()
            grad_s0 = torch.empty_like(s0)
            token_grads = [torch.empty_like(value) for value in (r, w, k, v, a, b)]
            torch.ops.rwkv7_statepassing_clampw.backward(
                s0,
                r,
                w,
                k,
                v,
                a,
                b,
                starts,
                grad_y,
                grad_s_t,
                checkpoints,
                state_a,
                grad_s0,
                *token_grads,
            )
            return grad_s0, *token_grads, None

    def operation(s0, r, w, k, v, a, b, starts):
        expected = torch.float32 if dtype == torch.float32 else torch.bfloat16
        if any(value.dtype != expected for value in (r, w, k, v, a, b)):
            raise TypeError(f"token tensors must use {expected}")
        return StatePassingOperation.apply(s0, r, w, k, v, a, b, starts)

    return operation


def make_inputs(
    dtype: torch.dtype, batch: int, timesteps: int, heads: int, head_size: int
) -> tuple[list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(20260904)
    shape = (batch, timesteps, heads, head_size)
    s0 = torch.empty(
        (batch, heads, head_size, head_size), device="cuda", dtype=torch.float32
    ).uniform_(-0.6, 0.6, generator=generator)
    r = torch.empty(shape, device="cuda").uniform_(-1.0, 1.0, generator=generator)
    w = torch.empty(shape, device="cuda").uniform_(-6.0, 0.0, generator=generator)
    k = torch.empty(shape, device="cuda").uniform_(-0.7, 0.7, generator=generator)
    v = torch.empty(shape, device="cuda").uniform_(-0.7, 0.7, generator=generator)
    a = torch.nn.functional.normalize(
        torch.empty(shape, device="cuda").uniform_(-1.0, 1.0, generator=generator),
        dim=-1,
    )
    b = torch.nn.functional.normalize(
        torch.empty(shape, device="cuda").uniform_(-1.0, 1.0, generator=generator),
        dim=-1,
    )
    starts = torch.zeros((batch, timesteps), device="cuda", dtype=torch.uint8)
    starts[0, [5, 19]] = 1
    starts[1, [0, 7, 16, 29]] = 1
    values = [s0, *(value.to(dtype) for value in (r, w, k, v, a, b))]
    return values, starts


def run_validation(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    compile_extension(
        args.rwkv_cuda_worktree,
        args.build_root,
        dtype,
        args.head_size,
        args.chunk_len,
    )
    operation = make_cuda_operation(dtype, args.chunk_len)
    base_values, starts = make_inputs(dtype, 2, 32, 2, args.head_size)

    cuda_values = [value.detach().clone().requires_grad_(True) for value in base_values]
    ref_values = [value.detach().float().clone().requires_grad_(True) for value in base_values]
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    grad_y = torch.randn((2, 32, 2, args.head_size), device="cuda", generator=generator)
    grad_s_t = torch.randn(
        (2, 2, args.head_size, args.head_size), device="cuda", generator=generator
    )

    cuda_y, cuda_s_t = operation(*cuda_values, starts)
    ((cuda_y.float() * grad_y).sum() + (cuda_s_t * grad_s_t).sum()).backward()
    ref_y, ref_s_t = state_passing_reference(*ref_values, starts)
    ((ref_y * grad_y).sum() + (ref_s_t * grad_s_t).sum()).backward()
    torch.cuda.synchronize()

    names = ("s0", "r", "w", "k", "v", "a", "b")
    gradient_errors = {
        name: relative_rms(reference.grad, actual.grad)
        for name, reference, actual in zip(names, ref_values, cuda_values)
    }

    with torch.no_grad():
        perturbed = [value.detach().clone() for value in base_values]
        perturbed[0].add_(3.0)
        for value in perturbed[1:]:
            value[0, :19].zero_()
            value[1, :29].zero_()
        isolated_y, isolated_s_t = operation(*perturbed, starts)
    isolation_error = max(
        relative_rms(cuda_y[0, 19:], isolated_y[0, 19:]),
        relative_rms(cuda_y[1, 29:], isolated_y[1, 29:]),
        relative_rms(cuda_s_t, isolated_s_t),
    )

    state_only_values = [value.detach().clone().requires_grad_(True) for value in base_values]
    _, state_only_s_t = operation(*state_only_values, starts)
    (state_only_s_t * grad_s_t).sum().backward()
    state_only_prefix_max = 0.0
    for value in state_only_values[1:]:
        state_only_prefix_max = max(
            state_only_prefix_max,
            float(value.grad[0, :19].abs().max()),
            float(value.grad[1, :29].abs().max()),
        )

    forward_tolerance = 2e-4 if dtype == torch.float32 else 1.5e-2
    gradient_tolerance = 8e-4 if dtype == torch.float32 else 2.5e-2
    result: dict[str, object] = {
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0),
        "forward_y_relative_rms": relative_rms(ref_y, cuda_y),
        "final_state_relative_rms": relative_rms(ref_s_t, cuda_s_t),
        "gradient_relative_rms": gradient_errors,
        "max_gradient_relative_rms": max(gradient_errors.values()),
        "batch1_ds0_max_abs": float(cuda_values[0].grad[1].abs().max()),
        "final_segment_isolation_relative_rms": isolation_error,
        "dsT_prefix_gradient_max_abs": state_only_prefix_max,
        "forward_tolerance": forward_tolerance,
        "gradient_tolerance": gradient_tolerance,
    }
    failures = []
    if result["forward_y_relative_rms"] > forward_tolerance:
        failures.append("forward y parity")
    if result["final_state_relative_rms"] > forward_tolerance:
        failures.append("final state parity")
    if result["max_gradient_relative_rms"] > gradient_tolerance:
        failures.append("gradient parity")
    for key in (
        "batch1_ds0_max_abs",
        "final_segment_isolation_relative_rms",
        "dsT_prefix_gradient_max_abs",
    ):
        if result[key] > 1e-7:
            failures.append(key)
    result["status"] = "passed" if not failures else "failed"
    result["failures"] = failures
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rwkv-cuda-worktree", type=Path, required=True)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/home/yueyulin/data/long_long_agent/tmp/torch_extensions"),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--head-size", type=int, default=16)
    parser.add_argument("--chunk-len", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_validation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
