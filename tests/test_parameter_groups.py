from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from src.training.parameter_groups import (
    build_fp32_master_adamw,
    build_rwkv7_optimizer_plan,
    build_torch_adamw,
    expected_rwkv7_parameter_names,
)


@dataclass
class FakeParameter:
    shape: tuple[int, ...]
    requires_grad: bool = True


def make_parameters(n_layer: int = 2):
    parameters = []
    for name in expected_rwkv7_parameter_names(n_layer, prefix="network"):
        if name.endswith("att.w0"):
            shape = (1, 1, 8)
        elif name.endswith(".weight") and not name.endswith("ln_x.weight") and ".ln" not in name:
            shape = (8, 8)
        else:
            shape = (8,)
        parameters.append((name, FakeParameter(shape)))
    parameters.extend(
        (
            ("latent_control.latent_embedding", FakeParameter((8,))),
            ("latent_control.depth_embedding", FakeParameter((16, 8))),
            ("value_readout.outcomes.weight", FakeParameter((8, 8))),
            ("value_readout.outcomes.bias", FakeParameter((8,))),
            ("value_readout.remaining_turns_head.weight", FakeParameter((1, 8))),
            ("value_readout.remaining_turns_head.bias", FakeParameter((1,))),
        )
    )
    return parameters


def test_expected_g1x_name_set_matches_checkpoint_count() -> None:
    names = expected_rwkv7_parameter_names(24, prefix="network")
    assert len(names) == 798
    assert len(set(names)) == 798


def test_optimizer_config_added_parameter_names_match_policy() -> None:
    config = yaml.safe_load(Path("configs/training_optimizer.yaml").read_text())
    configured = {value["full_name"] for value in config["added_parameters"].values()}
    assert configured == {
        "latent_control.latent_embedding",
        "latent_control.depth_embedding",
        "value_readout.outcomes.weight",
        "value_readout.outcomes.bias",
        "value_readout.remaining_turns_head.weight",
        "value_readout.remaining_turns_head.bias",
    }


def test_parameter_plan_reproduces_official_groups_and_explicit_latent_group() -> None:
    plan = build_rwkv7_optimizer_plan(
        make_parameters(),
        n_layer=2,
        weight_decay=0.1,
    )
    assignments = {assignment.name: assignment for assignment in plan.assignments}
    assert assignments["network.blocks.0.att.w0"].group_name == "base_lr_2x_no_decay"
    assert assignments["network.blocks.0.att.w0"].lr_scale == 2.0
    assert assignments["network.emb.weight"].group_name == "base_lr_1x_decay"
    assert assignments["network.blocks.0.ln1.weight"].group_name == "base_lr_1x_no_decay"
    assert assignments["network.blocks.0.att.w1"].group_name == "base_lr_1x_no_decay"
    assert assignments["latent_control.depth_embedding"].group_name == "latent_lr_1x_no_decay"
    assert {group.name for group in plan.groups} == {
        "base_lr_1x_decay",
        "base_lr_1x_no_decay",
        "base_lr_2x_no_decay",
        "latent_lr_1x_no_decay",
        "value_lr_1x_no_decay",
    }
    grouped_names = [name for group in plan.groups for name in group.parameter_names]
    assert len(grouped_names) == len(set(grouped_names)) == plan.covered_parameter_count


def test_zero_weight_decay_matches_official_fallback_to_no_decay() -> None:
    plan = build_rwkv7_optimizer_plan(
        make_parameters(),
        n_layer=2,
        weight_decay=0,
    )
    assignments = {assignment.name: assignment for assignment in plan.assignments}
    assert assignments["network.emb.weight"].group_name == "base_lr_1x_no_decay"
    assert all(group.weight_decay == 0 for group in plan.groups)


def test_base_only_plan_is_explicit_and_still_fail_closed() -> None:
    parameters = make_parameters()
    base_parameters = [item for item in parameters if item[0].startswith("network.")]
    plan = build_rwkv7_optimizer_plan(
        base_parameters,
        n_layer=2,
        weight_decay=0,
        latent_prefix=None,
        value_prefix=None,
    )
    assert plan.covered_parameter_count == len(expected_rwkv7_parameter_names(2))
    with pytest.raises(ValueError, match="coverage mismatch"):
        build_rwkv7_optimizer_plan(
            parameters,
            n_layer=2,
            weight_decay=0,
            latent_prefix=None,
            value_prefix=None,
        )


def test_torch_optimizer_applies_audited_lr_scales() -> None:
    torch = pytest.importorskip("torch")
    parameters = [
        (name, torch.nn.Parameter(torch.zeros(item.shape))) for name, item in make_parameters(1)
    ]
    plan = build_rwkv7_optimizer_plan(parameters, n_layer=1, weight_decay=0.1)
    optimizer = build_torch_adamw(plan, learning_rate=1e-3)
    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert by_name["base_lr_2x_no_decay"]["lr"] == pytest.approx(2e-3)
    assert by_name["base_lr_1x_decay"]["weight_decay"] == pytest.approx(0.1)
    assert by_name["latent_lr_1x_no_decay"]["lr"] == pytest.approx(1e-3)


def test_fp32_master_optimizer_keeps_state_in_fp32_for_bf16_model() -> None:
    torch = pytest.importorskip("torch")
    parameters = [
        (name, torch.nn.Parameter(torch.zeros(item.shape, dtype=torch.bfloat16)))
        for name, item in make_parameters(1)
    ]
    plan = build_rwkv7_optimizer_plan(parameters, n_layer=1, weight_decay=0)
    optimizer = build_fp32_master_adamw(plan, learning_rate=1e-3)
    loss = sum(parameter.float().square().sum() for _, parameter in parameters)
    loss.backward()
    optimizer.step()
    assert optimizer.optimizer_state_dtypes == {torch.float32}
    assert all(parameter.dtype == torch.bfloat16 for _, parameter in parameters)
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for _, parameter in parameters)


def test_frozen_parameters_are_covered_but_not_sent_to_optimizer() -> None:
    parameters = make_parameters()
    parameters[0][1].requires_grad = False
    plan = build_rwkv7_optimizer_plan(parameters, n_layer=2, weight_decay=0.1)
    assert plan.covered_parameter_count == len(parameters)
    grouped_names = {name for group in plan.groups for name in group.parameter_names}
    assert parameters[0][0] not in grouped_names


@pytest.mark.parametrize(
    "mutation",
    [
        lambda values: values + [("network.unknown.weight", FakeParameter((8, 8)))],
        lambda values: values[:-1],
        lambda values: values + [values[0]],
    ],
)
def test_parameter_policy_fails_closed_on_unknown_missing_or_duplicate(mutation) -> None:
    with pytest.raises(ValueError):
        build_rwkv7_optimizer_plan(
            mutation(make_parameters()),
            n_layer=2,
            weight_decay=0.1,
        )
