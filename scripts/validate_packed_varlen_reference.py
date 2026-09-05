#!/usr/bin/env python3
"""Validate packed RWKV recurrence semantics against separate sequences."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise

import torch


def reset_shift_mix(
    x: torch.Tensor, mix: torch.Tensor, sequence_start_mask: torch.Tensor
) -> torch.Tensor:
    previous = torch.cat((torch.zeros_like(x[:1]), x[:-1]), dim=0)
    previous = torch.where(sequence_start_mask[:, None], torch.zeros_like(previous), previous)
    return x + (previous - x) * mix


def wkv7_reference(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    sequence_start_mask: torch.Tensor,
) -> torch.Tensor:
    time, heads, width = r.shape
    state = torch.zeros((heads, width, width), dtype=r.dtype, device=r.device)
    outputs = []
    for index in range(time):
        if bool(sequence_start_mask[index]):
            state = torch.zeros_like(state)
        state_ab = torch.einsum("hij,hj,hk->hik", state, a[index], b[index])
        state = state * w[index, :, None, :] + state_ab
        state = state + torch.einsum("hj,hi->hij", k[index], v[index])
        outputs.append(torch.einsum("hj,hij->hi", r[index], state))
    return torch.stack(outputs)


def clone_leaves(values: list[torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().clone().requires_grad_(True) for value in values]


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach())


def validate(device: torch.device) -> dict[str, float | str]:
    torch.manual_seed(20260904)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    lengths = (3, 5, 2)
    total = sum(lengths)
    starts = torch.zeros(total, dtype=torch.bool, device=device)
    offsets = [0]
    for length in lengths:
        starts[offsets[-1]] = True
        offsets.append(offsets[-1] + length)

    x_base = torch.randn(total, 8, dtype=dtype, device=device)
    mix_base = torch.sigmoid(torch.randn(8, dtype=dtype, device=device))
    x_packed, mix_packed = clone_leaves([x_base, mix_base])
    mixed_packed = reset_shift_mix(x_packed, mix_packed, starts)
    mixed_weight = torch.randn_like(mixed_packed)
    (mixed_packed * mixed_weight).sum().backward()

    x_parts = []
    mix_separate = mix_base.detach().clone().requires_grad_(True)
    separate_mixed = []
    for begin, end in pairwise(offsets):
        part = x_base[begin:end].detach().clone().requires_grad_(True)
        x_parts.append(part)
        part_starts = torch.zeros(end - begin, dtype=torch.bool, device=device)
        part_starts[0] = True
        separate_mixed.append(reset_shift_mix(part, mix_separate, part_starts))
    mixed_separate = torch.cat(separate_mixed)
    (mixed_separate * mixed_weight).sum().backward()

    shape = (total, 2, 4)
    bases = [torch.randn(shape, dtype=dtype, device=device) for _ in range(6)]
    bases[1] = torch.sigmoid(bases[1])
    packed_leaves = clone_leaves(bases)
    wkv_packed = wkv7_reference(*packed_leaves, starts)
    wkv_weight = torch.randn_like(wkv_packed)
    (wkv_packed * wkv_weight).sum().backward()

    separate_leaves: list[list[torch.Tensor]] = [[] for _ in bases]
    separate_outputs = []
    for begin, end in pairwise(offsets):
        part_values = clone_leaves([value[begin:end] for value in bases])
        for destination, value in zip(separate_leaves, part_values):
            destination.append(value)
        part_starts = torch.zeros(end - begin, dtype=torch.bool, device=device)
        part_starts[0] = True
        separate_outputs.append(wkv7_reference(*part_values, part_starts))
    wkv_separate = torch.cat(separate_outputs)
    (wkv_separate * wkv_weight).sum().backward()

    shift_grad_error = max_error(x_packed.grad, torch.cat([part.grad for part in x_parts]))
    shift_mix_grad_error = max_error(mix_packed.grad, mix_separate.grad)
    wkv_grad_error = max(
        max_error(packed.grad, torch.cat([part.grad for part in parts]))
        for packed, parts in zip(packed_leaves, separate_leaves)
    )
    result = {
        "device": str(device),
        "shift_forward_max_abs_error": max_error(mixed_packed, mixed_separate),
        "shift_input_grad_max_abs_error": shift_grad_error,
        "shift_mix_grad_max_abs_error": shift_mix_grad_error,
        "wkv_forward_max_abs_error": max_error(wkv_packed, wkv_separate),
        "wkv_grad_max_abs_error": wkv_grad_error,
    }
    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    if any(value > tolerance for key, value in result.items() if key != "device"):
        raise RuntimeError(f"packed reference parity failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(json.dumps(validate(device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
