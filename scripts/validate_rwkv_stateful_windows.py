#!/usr/bin/env python3
"""Validate differentiable RWKV-7 state continuation on a real checkpoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.latent_v0 import RWKV7V0LatentControl, run_v0_latent_steps
from src.model.readout import BINARY_VALUE_TASKS, RWKV7ValueReadout
from src.model.rwkv7_stateful import (
    state_passing,
    state_passing_reference,
    stateful_forward,
    stateful_forward_embeddings,
)
from src.model.slow_fast_state import SlowFastRWKVState
from src.model.state import RWKVState
from src.training.losses import multitask_value_loss, weighted_action_cross_entropy


def relative_rms(reference: torch.Tensor, actual: torch.Tensor) -> float:
    error = (reference.float() - actual.float()).square().mean().sqrt()
    scale = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float((error / scale).detach())


def compile_state_passing(worktree: Path, build_root: Path) -> None:
    source_root = worktree / "rwkv7_fast_fused"
    build_directory = build_root / "state_passing_bf16_n64_c16"
    build_directory.mkdir(parents=True, exist_ok=True)
    load(
        name="rwkv7_statepassing_clampw",
        sources=[
            str(source_root / "cuda/rwkv7_statepassing_clampw.cu"),
            str(source_root / "cuda/rwkv7_statepassing_clampw.cpp"),
        ],
        build_directory=str(build_directory),
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=[
            "-res-usage",
            "-D_N_=64",
            "-D_CHUNK_LEN_=16",
            "--use_fast_math",
            "-O3",
            "-Xptxas=-O3",
            "--extra-device-vectorization",
        ],
    )


def make_network(model_module, checkpoint: Path):
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
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def max_state_error(reference: RWKVState, actual: RWKVState) -> dict[str, float]:
    fields = {
        "time_mix_previous_x": 0.0,
        "wkv_matrix": 0.0,
        "channel_mix_previous_x": 0.0,
    }
    for expected_layer, actual_layer in zip(reference.layers, actual.layers):
        for name, current_error in fields.items():
            fields[name] = max(
                current_error,
                relative_rms(getattr(expected_layer, name), getattr(actual_layer, name)),
            )
    return fields


def state_round_trip(state: RWKVState, temporary_root: Path) -> tuple[float, bool]:
    detached = state.detached()
    clone = detached.clone()
    no_alias = all(
        source.time_mix_previous_x.data_ptr() != copied.time_mix_previous_x.data_ptr()
        and source.wkv_matrix.data_ptr() != copied.wkv_matrix.data_ptr()
        and source.channel_mix_previous_x.data_ptr() != copied.channel_mix_previous_x.data_ptr()
        for source, copied in zip(detached.layers, clone.layers)
    )
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        path = Path(directory) / "rwkv_state.pt"
        detached.save(path)
        restored = RWKVState.load(path, map_location="cuda")
    errors = max_state_error(detached, restored)
    return max(errors.values()), no_alias


def validate_slow_fast_ownership(
    network, tokens: torch.Tensor, slow_state: RWKVState
) -> dict[str, float | bool]:
    reference_slow = slow_state.clone(detach=True)
    decision = SlowFastRWKVState.begin_decision(
        slow_state,
        decision_id="validation-episode:decision-1",
    )
    initial_no_alias = not decision.slow.shares_storage_with(decision.fast)

    next_fast = decision.fast.clone()
    next_fast.layers[0].wkv_matrix.add_(1)
    decision = decision.with_fast(next_fast)
    contamination = max(max_state_error(reference_slow, decision.slow).values())

    continuation_starts = torch.zeros_like(tokens[:, :16], dtype=torch.uint8)
    with torch.no_grad():
        _, next_slow = stateful_forward(
            network,
            tokens[:, :16],
            state=decision.slow,
            sequence_start_mask=continuation_starts,
            return_logits=False,
        )
    advanced = decision.advance_slow(
        next_slow,
        next_decision_id="validation-episode:decision-2",
        confirmed_token_count=16,
    )
    rederived_fast_error = max(max_state_error(advanced.slow, advanced.fast).values())

    direct_commit_rejected = False
    try:
        decision.advance_slow(
            decision.fast,
            next_decision_id="invalid",
            confirmed_token_count=1,
        )
    except ValueError:
        direct_commit_rejected = True

    return {
        "slow_fast_initial_no_storage_alias": initial_no_alias,
        "fast_mutation_slow_contamination_relative_rms": contamination,
        "next_decision_fast_clone_relative_rms": rederived_fast_error,
        "fast_to_slow_direct_commit_rejected": direct_commit_rejected,
    }


def validate_state_kernel_continuation() -> dict[str, float]:
    generator = torch.Generator(device="cuda").manual_seed(20260908)
    shape = (1, 32, 128)
    base_s0 = torch.empty((1, 2, 64, 64), device="cuda").uniform_(-0.2, 0.2, generator=generator)
    base_values = [
        torch.empty(shape, device="cuda", dtype=torch.bfloat16).uniform_(
            -0.5, 0.5, generator=generator
        )
        for _ in range(6)
    ]
    grad_y = torch.randn(shape, device="cuda", generator=generator)
    grad_state = torch.randn(base_s0.shape, device="cuda", generator=generator)
    starts = torch.zeros((1, 32), device="cuda", dtype=torch.uint8)

    whole_values = [base_s0.clone().requires_grad_(True)] + [
        value.clone().requires_grad_(True) for value in base_values
    ]
    whole_y, whole_state = state_passing(*whole_values[1:], whole_values[0], starts)
    ((whole_y.float() * grad_y).sum() + (whole_state * grad_state).sum()).backward()

    split_values = [base_s0.clone().requires_grad_(True)] + [
        value.clone().requires_grad_(True) for value in base_values
    ]
    first_y, first_state = state_passing(
        *(value[:, :16] for value in split_values[1:]),
        split_values[0],
        starts[:, :16].contiguous(),
    )
    second_y, split_state = state_passing(
        *(value[:, 16:] for value in split_values[1:]),
        first_state,
        starts[:, 16:].contiguous(),
    )
    split_y = torch.cat((first_y, second_y), dim=1)
    ((split_y.float() * grad_y).sum() + (split_state * grad_state).sum()).backward()

    gradient_errors = [
        relative_rms(reference.grad, actual.grad)
        for reference, actual in zip(whole_values, split_values)
    ]

    cuda_values = [base_s0.clone().requires_grad_(True)] + [
        value[:, :16].clone().requires_grad_(True) for value in base_values
    ]
    cuda_y, cuda_state = state_passing(
        *cuda_values[1:], cuda_values[0], starts[:, :16].contiguous()
    )
    ((cuda_y.float() * grad_y[:, :16]).sum() + (cuda_state * grad_state).sum()).backward()

    reference_values = [base_s0.clone().requires_grad_(True)] + [
        value[:, :16].clone().requires_grad_(True) for value in base_values
    ]
    reference_y, reference_state = state_passing_reference(
        *reference_values[1:],
        reference_values[0],
        starts[:, :16].contiguous(),
    )
    ((reference_y.float() * grad_y[:, :16]).sum() + (reference_state * grad_state).sum()).backward()
    reference_gradient_errors = [
        relative_rms(expected.grad, actual.grad)
        for expected, actual in zip(cuda_values, reference_values)
    ]
    return {
        "wkv_whole_vs_continuation_output_relative_rms": relative_rms(whole_y, split_y),
        "wkv_whole_vs_continuation_state_relative_rms": relative_rms(whole_state, split_state),
        "wkv_whole_vs_continuation_max_gradient_relative_rms": max(gradient_errors),
        "wkv_cuda_vs_short_reference_output_relative_rms": relative_rms(cuda_y, reference_y),
        "wkv_cuda_vs_short_reference_state_relative_rms": relative_rms(cuda_state, reference_state),
        "wkv_cuda_vs_short_reference_max_gradient_relative_rms": max(reference_gradient_errors),
    }


def validate_v0_latent(network, slow_state: RWKVState, tokens: torch.Tensor) -> dict[str, Any]:
    control = RWKV7V0LatentControl(n_embd=slow_state.spec.n_embd, max_depth=16).to("cuda")
    control.initialize_from_token_embeddings(network.emb.weight)
    value_readout = RWKV7ValueReadout(slow_state.spec.n_embd).to("cuda")
    decision = SlowFastRWKVState.begin_decision(
        slow_state.detached(),
        decision_id="validation-episode:latent-decision",
    )
    reference_slow = decision.slow.clone(detach=True)

    k_zero = run_v0_latent_steps(network, control, decision, steps=0)
    head_calls = 0

    def count_head_calls(_module, _inputs, _output):
        nonlocal head_calls
        head_calls += 1

    hook = network.head.register_forward_hook(count_head_calls)
    rollout = run_v0_latent_steps(network, control, decision, steps=4)
    latent_head_calls = head_calls
    hook.remove()

    value_predictions = value_readout(rollout.hidden[:, -1])
    binary_targets = torch.tensor(
        [[1, 1, 1, 1, 0, 1, 1, 0]],
        device="cuda",
        dtype=torch.float32,
    )
    if binary_targets.shape[-1] != len(BINARY_VALUE_TASKS):
        raise AssertionError("validation target count does not match value task schema")
    value_loss = multitask_value_loss(
        value_predictions,
        binary_targets,
        torch.ones_like(binary_targets, dtype=torch.bool),
        torch.tensor([4.0], device="cuda"),
        torch.ones(1, device="cuda", dtype=torch.bool),
    )
    value_loss.total.backward()
    latent_gradient_rms = float(control.latent_embedding.grad.square().mean().sqrt())
    used_depth_gradient_rms = float(control.depth_embedding.grad[:4].square().mean().sqrt())
    unused_depth_gradient_max_abs = float(control.depth_embedding.grad[4:].abs().max())
    value_gradient_rms = float(value_readout.outcomes.weight.grad.square().mean().sqrt())
    slow_contamination = max(max_state_error(reference_slow, rollout.decision.slow).values())

    head_calls = 0
    hook = network.head.register_forward_hook(count_head_calls)
    with torch.no_grad():
        action_logits, _ = stateful_forward(
            network,
            tokens[:, :1],
            state=rollout.decision.fast.detached(),
            return_logits=True,
        )
    action_head_calls = head_calls
    hook.remove()
    action_loss = weighted_action_cross_entropy(
        action_logits,
        tokens[:, :1],
        torch.ones_like(tokens[:, :1], dtype=torch.float32),
    )

    return {
        "latent_k0_decision_identity_preserved": k_zero.decision is decision,
        "latent_k0_hidden_timesteps": k_zero.hidden.shape[1],
        "latent_k4_recorded_fast_steps": rollout.decision.fast_steps,
        "latent_k4_lm_head_calls": latent_head_calls,
        "action_lm_head_calls": action_head_calls,
        "latent_embedding_gradient_rms": latent_gradient_rms,
        "used_depth_embedding_gradient_rms": used_depth_gradient_rms,
        "unused_depth_embedding_gradient_max_abs": unused_depth_gradient_max_abs,
        "value_readout_gradient_rms": value_gradient_rms,
        "value_loss": float(value_loss.total.detach()),
        "action_weighted_ce_loss": float(action_loss.total.detach()),
        "action_effective_loss_tokens": action_loss.effective_tokens,
        "latent_fast_slow_storage_alias": rollout.decision.fast.shares_storage_with(
            rollout.decision.slow
        ),
        "latent_fast_mutation_slow_contamination_relative_rms": slow_contamination,
    }


def validate_forward(network, tokens: torch.Tensor) -> tuple[dict[str, float], RWKVState]:
    starts = torch.zeros_like(tokens, dtype=torch.uint8)
    starts[:, 0] = 1
    with torch.no_grad():
        official = network._forward_features(tokens, starts)
        stateful_whole, whole_state = stateful_forward(
            network, tokens, sequence_start_mask=starts, return_logits=False
        )

        first_hidden, first_state = stateful_forward(
            network,
            tokens[:, :16],
            sequence_start_mask=starts[:, :16].contiguous(),
            return_logits=False,
        )
        continuation_starts = torch.zeros_like(tokens[:, 16:], dtype=torch.uint8)
        second_hidden, split_state = stateful_forward(
            network,
            tokens[:, 16:],
            state=first_state,
            sequence_start_mask=continuation_starts,
            return_logits=False,
        )
        split_hidden = torch.cat((first_hidden, second_hidden), dim=1)

        reset_starts = starts.clone()
        reset_starts[:, 16] = 1
        reset_whole, _ = stateful_forward(
            network, tokens, sequence_start_mask=reset_starts, return_logits=False
        )
        independent_parts = []
        for part in (tokens[:, :16], tokens[:, 16:]):
            part_starts = torch.zeros_like(part, dtype=torch.uint8)
            part_starts[:, 0] = 1
            hidden, _ = stateful_forward(
                network, part, sequence_start_mask=part_starts, return_logits=False
            )
            independent_parts.append(hidden)
        independent = torch.cat(independent_parts, dim=1)

    state_errors = max_state_error(whole_state, split_state)
    return (
        {
            "official_vs_stateful_hidden_relative_rms": relative_rms(official, stateful_whole),
            "whole_vs_continuation_hidden_relative_rms": relative_rms(stateful_whole, split_hidden),
            "internal_reset_vs_independent_hidden_relative_rms": relative_rms(
                reset_whole, independent
            ),
            "continuation_time_mix_state_relative_rms": state_errors["time_mix_previous_x"],
            "continuation_wkv_state_relative_rms": state_errors["wkv_matrix"],
            "continuation_channel_mix_state_relative_rms": state_errors["channel_mix_previous_x"],
        },
        split_state,
    )


def validate_gradient(network, tokens: torch.Tensor) -> dict[str, float]:
    generator = torch.Generator(device="cuda").manual_seed(20260907)
    base_embeddings = network.emb(tokens).detach()
    output_weight = torch.randn(
        base_embeddings.shape, device="cuda", dtype=torch.float32, generator=generator
    )

    whole_embeddings = base_embeddings.clone().requires_grad_(True)
    whole_hidden, _ = stateful_forward_embeddings(network, whole_embeddings)
    (whole_hidden[:, 16:].float() * output_weight[:, 16:]).sum().backward()
    whole_gradient = whole_embeddings.grad.detach().clone()

    split_embeddings = base_embeddings.clone().requires_grad_(True)
    first_hidden, first_state = stateful_forward_embeddings(network, split_embeddings[:, :16])
    second_hidden, _ = stateful_forward_embeddings(
        network, split_embeddings[:, 16:], state=first_state
    )
    split_hidden = torch.cat((first_hidden, second_hidden), dim=1)
    (split_hidden[:, 16:].float() * output_weight[:, 16:]).sum().backward()
    split_gradient = split_embeddings.grad.detach().clone()

    detached_embeddings = base_embeddings.clone().requires_grad_(True)
    _, attached_state = stateful_forward_embeddings(network, detached_embeddings[:, :16])
    detached_hidden, _ = stateful_forward_embeddings(
        network,
        detached_embeddings[:, 16:],
        state=attached_state.detached(),
    )
    (detached_hidden.float() * output_weight[:, 16:]).sum().backward()
    detached_gradient = detached_embeddings.grad.detach()

    return {
        "whole_vs_continuation_input_gradient_relative_rms": relative_rms(
            whole_gradient, split_gradient
        ),
        "full_bptt_first_window_gradient_rms": float(
            whole_gradient[:, :16].float().square().mean().sqrt()
        ),
        "detached_first_window_gradient_max_abs": float(
            detached_gradient[:, :16].float().abs().max()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rwkv-lm-worktree", type=Path, required=True)
    parser.add_argument("--rwkv-cuda-worktree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/home/yueyulin/data/long_long_agent/tmp/torch_extensions_stateful"),
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=Path("/home/yueyulin/data/long_long_agent/tmp"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ.setdefault("RWKV_JIT_ON", "0")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    compile_state_passing(args.rwkv_cuda_worktree.resolve(), args.build_root.resolve())

    train_root = args.rwkv_lm_worktree.resolve() / "RWKV-v7/train_temp"
    os.chdir(train_root)
    sys.path.insert(0, str(train_root / "src"))
    model_module = importlib.import_module("model")
    network = make_network(model_module, args.checkpoint.resolve())

    torch.manual_seed(20260906)
    tokens = torch.randint(0, 65530, (1, 32), device="cuda")
    kernel = validate_state_kernel_continuation()
    forward, state = validate_forward(network, tokens)
    gradient = validate_gradient(network, tokens)
    serialization_error, no_alias = state_round_trip(state, args.temporary_root.resolve())
    ownership = validate_slow_fast_ownership(network, tokens, state)
    latent = validate_v0_latent(network, state, tokens)
    result = {
        "device": torch.cuda.get_device_name(0),
        **kernel,
        **forward,
        **gradient,
        **ownership,
        **latent,
        "state_serialization_max_relative_rms": serialization_error,
        "state_clone_has_no_storage_alias": no_alias,
    }
    failures = []
    if result["official_vs_stateful_hidden_relative_rms"] > 0.02:
        failures.append("official full-sequence parity")
    if result["wkv_whole_vs_continuation_output_relative_rms"] > 0.01:
        failures.append("WKV continuation output parity")
    if result["wkv_whole_vs_continuation_state_relative_rms"] > 0.01:
        failures.append("WKV continuation state parity")
    if result["wkv_whole_vs_continuation_max_gradient_relative_rms"] > 0.01:
        failures.append("WKV continuation gradient parity")
    if result["wkv_cuda_vs_short_reference_output_relative_rms"] > 0.01:
        failures.append("WKV CUDA vs short reference output parity")
    if result["wkv_cuda_vs_short_reference_state_relative_rms"] > 0.01:
        failures.append("WKV CUDA vs short reference state parity")
    if result["wkv_cuda_vs_short_reference_max_gradient_relative_rms"] > 0.01:
        failures.append("WKV CUDA vs short reference gradient parity")
    model_continuation = [
        result["whole_vs_continuation_hidden_relative_rms"],
        result["internal_reset_vs_independent_hidden_relative_rms"],
        result["whole_vs_continuation_input_gradient_relative_rms"],
    ]
    if max(model_continuation) > 0.06:
        failures.append("24-layer BF16 window continuation drift")
    if result["full_bptt_first_window_gradient_rms"] <= 0:
        failures.append("full-BPTT did not cross the window boundary")
    if result["detached_first_window_gradient_max_abs"] > 0:
        failures.append("detached state leaked gradient into the previous window")
    if not no_alias:
        failures.append("state clone shares storage")
    if not result["slow_fast_initial_no_storage_alias"]:
        failures.append("initial fast state aliases slow state")
    if result["fast_mutation_slow_contamination_relative_rms"] > 0:
        failures.append("fast mutation contaminated slow state")
    if result["next_decision_fast_clone_relative_rms"] > 0:
        failures.append("next decision fast state differs from confirmed slow state")
    if not result["fast_to_slow_direct_commit_rejected"]:
        failures.append("direct fast-to-slow commit was accepted")
    if (
        not result["latent_k0_decision_identity_preserved"]
        or result["latent_k0_hidden_timesteps"] != 0
    ):
        failures.append("K=0 latent anchor changed decision state")
    if result["latent_k4_recorded_fast_steps"] != 4:
        failures.append("latent depth ownership mismatch")
    if result["latent_k4_lm_head_calls"] != 0 or result["action_lm_head_calls"] != 1:
        failures.append("LM head call boundary mismatch")
    if result["latent_embedding_gradient_rms"] <= 0:
        failures.append("latent embedding received no gradient")
    if result["used_depth_embedding_gradient_rms"] <= 0:
        failures.append("used depth embeddings received no gradient")
    if result["unused_depth_embedding_gradient_max_abs"] > 0:
        failures.append("unused depth embeddings received gradient")
    if result["value_readout_gradient_rms"] <= 0:
        failures.append("value readout received no gradient")
    if not torch.isfinite(torch.tensor(result["value_loss"])):
        failures.append("value loss is not finite")
    if not torch.isfinite(torch.tensor(result["action_weighted_ce_loss"])):
        failures.append("action loss is not finite")
    if result["action_effective_loss_tokens"] != 1:
        failures.append("action loss mask effective-token count mismatch")
    if result["latent_fast_slow_storage_alias"]:
        failures.append("latent fast state aliases slow state")
    if result["latent_fast_mutation_slow_contamination_relative_rms"] > 0:
        failures.append("latent rollout contaminated slow state")
    result["status"] = "passed" if not failures else "failed"
    result["failures"] = failures
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
