"""Fail-closed RWKV-7 optimizer parameter classification."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterAssignment:
    name: str
    group_name: str
    lr_scale: float
    weight_decay: float
    trainable: bool
    numel: int


@dataclass(frozen=True)
class OptimizerGroup:
    name: str
    parameter_names: tuple[str, ...]
    parameters: tuple[Any, ...]
    lr_scale: float
    weight_decay: float

    def as_torch_group(self) -> dict[str, Any]:
        return {
            "group_name": self.name,
            "params": list(self.parameters),
            "weight_decay": self.weight_decay,
            "my_lr_scale": self.lr_scale,
        }


@dataclass(frozen=True)
class OptimizerPlan:
    assignments: tuple[ParameterAssignment, ...]
    groups: tuple[OptimizerGroup, ...]

    @property
    def covered_parameter_count(self) -> int:
        return len(self.assignments)

    @property
    def covered_numel(self) -> int:
        return sum(assignment.numel for assignment in self.assignments)

    @property
    def trainable_numel(self) -> int:
        return sum(assignment.numel for assignment in self.assignments if assignment.trainable)

    def torch_groups(self) -> list[dict[str, Any]]:
        return [group.as_torch_group() for group in self.groups]


def expected_rwkv7_parameter_names(n_layer: int, *, prefix: str = "") -> tuple[str, ...]:
    if n_layer <= 0:
        raise ValueError("n_layer must be positive")
    normalized_prefix = f"{prefix}." if prefix else ""
    names = ["emb.weight"]
    attention_fields = (
        "x_r",
        "x_w",
        "x_k",
        "x_v",
        "x_a",
        "x_g",
        "w0",
        "r_k",
        "w1",
        "w2",
        "a1",
        "a2",
        "a0",
        "g1",
        "g2",
        "v2",
        "v1",
        "v0",
        "k_k",
        "k_a",
    )
    for layer in range(n_layer):
        root = f"blocks.{layer}"
        names.extend(
            (
                f"{root}.ln1.weight",
                f"{root}.ln1.bias",
                f"{root}.ln2.weight",
                f"{root}.ln2.bias",
            )
        )
        if layer == 0:
            names.extend((f"{root}.ln0.weight", f"{root}.ln0.bias"))
        names.extend(f"{root}.att.{field}" for field in attention_fields)
        names.extend(
            f"{root}.att.{projection}.weight"
            for projection in ("receptance", "key", "value", "output")
        )
        names.extend((f"{root}.att.ln_x.weight", f"{root}.att.ln_x.bias"))
        names.extend(
            (
                f"{root}.ffn.x_k",
                f"{root}.ffn.key.weight",
                f"{root}.ffn.value.weight",
            )
        )
    names.extend(("ln_out.weight", "ln_out.bias", "head.weight"))
    return tuple(f"{normalized_prefix}{name}" for name in names)


def _numel(parameter: Any) -> int:
    result = 1
    for dimension in parameter.shape:
        result *= int(dimension)
    return result


def _official_group(
    base_name: str, parameter: Any, weight_decay: float
) -> tuple[str, float, float]:
    if "att.w0" in base_name:
        return "base_lr_2x_no_decay", 2.0, 0.0
    squeezed_rank = sum(int(dimension) != 1 for dimension in parameter.shape)
    if squeezed_rank >= 2 and weight_decay > 0 and ".weight" in base_name:
        return "base_lr_1x_decay", 1.0, weight_decay
    return "base_lr_1x_no_decay", 1.0, 0.0


def build_rwkv7_optimizer_plan(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    n_layer: int,
    weight_decay: float,
    base_prefix: str = "network",
    latent_prefix: str | None = "latent_control",
    value_prefix: str | None = "value_readout",
    latent_lr_scale: float = 1.0,
    value_lr_scale: float = 1.0,
) -> OptimizerPlan:
    """Classify an exact RWKV-7 + V0 parameter set or reject it."""
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not math.isfinite(latent_lr_scale) or latent_lr_scale <= 0:
        raise ValueError("latent_lr_scale must be positive")
    if not math.isfinite(value_lr_scale) or value_lr_scale <= 0:
        raise ValueError("value_lr_scale must be positive")
    provided = list(named_parameters)
    names = [name for name, _ in provided]
    if len(names) != len(set(names)):
        raise ValueError("duplicate parameter names are not allowed")

    base_names = set(expected_rwkv7_parameter_names(n_layer, prefix=base_prefix))
    latent_names = (
        {
            f"{latent_prefix}.latent_embedding",
            f"{latent_prefix}.depth_embedding",
        }
        if latent_prefix is not None
        else set()
    )
    value_names = (
        {
            f"{value_prefix}.outcomes.weight",
            f"{value_prefix}.outcomes.bias",
            f"{value_prefix}.remaining_turns_head.weight",
            f"{value_prefix}.remaining_turns_head.bias",
        }
        if value_prefix is not None
        else set()
    )
    expected = base_names | latent_names | value_names
    actual = set(names)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(
            "parameter policy coverage mismatch: "
            f"unknown={unknown[:8]} missing={missing[:8]} "
            f"unknown_count={len(unknown)} missing_count={len(missing)}"
        )

    assignments = []
    grouped: dict[tuple[str, float, float], list[tuple[str, Any]]] = {}
    prefix_with_dot = f"{base_prefix}." if base_prefix else ""
    for name, parameter in sorted(provided, key=lambda item: item[0]):
        if name in latent_names:
            key = ("latent_lr_1x_no_decay", latent_lr_scale, 0.0)
        elif name in value_names:
            key = ("value_lr_1x_no_decay", value_lr_scale, 0.0)
        else:
            base_name = name.removeprefix(prefix_with_dot)
            key = _official_group(base_name, parameter, weight_decay)
        trainable = bool(getattr(parameter, "requires_grad", False))
        assignments.append(
            ParameterAssignment(
                name=name,
                group_name=key[0],
                lr_scale=key[1],
                weight_decay=key[2],
                trainable=trainable,
                numel=_numel(parameter),
            )
        )
        if trainable:
            grouped.setdefault(key, []).append((name, parameter))

    seen_parameter_ids: set[int] = set()
    groups = []
    for (group_name, lr_scale, group_decay), members in sorted(grouped.items()):
        for _, parameter in members:
            identity = id(parameter)
            if identity in seen_parameter_ids:
                raise ValueError("a trainable parameter appears in more than one optimizer group")
            seen_parameter_ids.add(identity)
        groups.append(
            OptimizerGroup(
                name=group_name,
                parameter_names=tuple(name for name, _ in members),
                parameters=tuple(parameter for _, parameter in members),
                lr_scale=lr_scale,
                weight_decay=group_decay,
            )
        )
    return OptimizerPlan(assignments=tuple(assignments), groups=tuple(groups))


def build_torch_adamw(
    plan: OptimizerPlan,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    epsilon: float = 1e-18,
    fused: bool = False,
):
    """Build a PyTorch AdamW while preserving the audited per-group LR scales."""
    values = (learning_rate, beta1, beta2, epsilon)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("optimizer scalars must be finite")
    if learning_rate <= 0 or epsilon <= 0:
        raise ValueError("learning_rate and epsilon must be positive")
    if not 0 < beta1 < 1 or not 0 < beta2 < 1:
        raise ValueError("optimizer betas must be in (0, 1)")
    if not plan.groups:
        raise ValueError("optimizer plan has no trainable parameter groups")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to build the optimizer") from error
    groups = []
    for group in plan.groups:
        values = group.as_torch_group()
        values["lr"] = learning_rate * group.lr_scale
        groups.append(values)
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=0.0,
        fused=fused,
    )


class FP32MasterAdamW:
    """Single-device AdamW with FP32 master parameters for BF16 model weights."""

    def __init__(
        self,
        plan: OptimizerPlan,
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-18,
        fused: bool = False,
    ) -> None:
        if not plan.groups:
            raise ValueError("optimizer plan has no trainable parameter groups")
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to build the optimizer") from error
        scalar_values = (learning_rate, beta1, beta2, epsilon)
        if any(not math.isfinite(value) for value in scalar_values):
            raise ValueError("optimizer scalars must be finite")
        if learning_rate <= 0 or epsilon <= 0:
            raise ValueError("learning_rate and epsilon must be positive")
        if not 0 < beta1 < 1 or not 0 < beta2 < 1:
            raise ValueError("optimizer betas must be in (0, 1)")

        self.param_groups: list[dict[str, Any]] = []
        self._pairs: list[tuple[Any, Any]] = []
        self._parameter_names = tuple(
            name for group in plan.groups for name in group.parameter_names
        )
        master_groups = []
        for group in plan.groups:
            model_parameters = list(group.parameters)
            if any(not isinstance(parameter, torch.nn.Parameter) for parameter in model_parameters):
                raise TypeError("FP32 master optimizer requires torch.nn.Parameter values")
            if any(not parameter.dtype.is_floating_point for parameter in model_parameters):
                raise TypeError("FP32 master optimizer requires floating-point parameters")
            master_parameters = [
                torch.nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
                for parameter in model_parameters
            ]
            self._pairs.extend(zip(model_parameters, master_parameters, strict=True))
            group_lr = learning_rate * group.lr_scale
            common = {
                "group_name": group.name,
                "weight_decay": group.weight_decay,
                "my_lr_scale": group.lr_scale,
                "lr": group_lr,
            }
            self.param_groups.append({**common, "params": model_parameters})
            master_groups.append({**common, "params": master_parameters})
        self._optimizer = torch.optim.AdamW(
            master_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=0.0,
            fused=fused,
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for model_parameter, _ in self._pairs:
            if set_to_none:
                model_parameter.grad = None
            elif model_parameter.grad is not None:
                model_parameter.grad.zero_()
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        import torch

        with torch.no_grad():
            for model_parameter, master_parameter in self._pairs:
                master_parameter.grad = (
                    None if model_parameter.grad is None else model_parameter.grad.detach().float()
                )
        result = self._optimizer.step()
        with torch.no_grad():
            for model_parameter, master_parameter in self._pairs:
                model_parameter.copy_(master_parameter)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "parameter_names": self._parameter_names,
            "master_weights": [master.detach().cpu().clone() for _, master in self._pairs],
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        import torch

        if set(state) != {"schema_version", "parameter_names", "master_weights", "optimizer"}:
            raise ValueError("unknown or missing FP32 optimizer checkpoint fields")
        if state["schema_version"] != 1 or tuple(state["parameter_names"]) != self._parameter_names:
            raise ValueError("FP32 optimizer checkpoint parameter identity mismatch")
        masters = state["master_weights"]
        if len(masters) != len(self._pairs):
            raise ValueError("FP32 optimizer checkpoint parameter count mismatch")
        for saved, (_, master) in zip(masters, self._pairs, strict=True):
            if (
                saved.shape != master.shape
                or saved.dtype != torch.float32
                or not torch.isfinite(saved).all()
            ):
                raise ValueError("invalid FP32 master checkpoint tensor")
        saved_groups = state["optimizer"]["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            saved["group_name"] != current["group_name"]
            or len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, self.param_groups, strict=True)
        ):
            raise ValueError("FP32 optimizer checkpoint groups mismatch")
        self._optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for saved, (model, master) in zip(masters, self._pairs, strict=True):
                master.copy_(saved)
                model.copy_(master)
        for current, restored in zip(self.param_groups, self._optimizer.param_groups, strict=True):
            current.update({key: value for key, value in restored.items() if key != "params"})

    @property
    def optimizer_state_dtypes(self) -> set[Any]:
        return {
            value.dtype
            for state in self._optimizer.state.values()
            for key, value in state.items()
            if key != "step" and hasattr(value, "dtype")
        }


def build_fp32_master_adamw(
    plan: OptimizerPlan,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.99,
    epsilon: float = 1e-18,
    fused: bool = False,
) -> FP32MasterAdamW:
    return FP32MasterAdamW(
        plan,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        fused=fused,
    )
