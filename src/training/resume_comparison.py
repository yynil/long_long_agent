"""Closed, finite, full-coverage comparisons for ADR-020; no tolerance fitting."""

from __future__ import annotations

import re

import torch

from .preflight import tree_digest


def tensor_comparison(expected: dict, actual: dict, *, denominator_floor: float) -> dict:
    if set(expected) != set(actual):
        raise ValueError("tensor coverage changed")
    report = {"tensors": 0, "elements": 0, "different_elements": 0, "differences": []}
    for name, reference in expected.items():
        observed = actual[name]
        if reference is None or observed is None:
            if reference is not observed:
                raise ValueError("tensor presence changed")
            continue
        if reference.dtype != observed.dtype or reference.shape != observed.shape:
            raise ValueError("tensor specification changed")
        reference, observed = reference.detach().cpu(), observed.detach().cpu()
        if not torch.isfinite(reference).all() or not torch.isfinite(observed).all():
            raise FloatingPointError("non-finite comparison tensor")
        report["tensors"] += 1
        report["elements"] += reference.numel()
        count = int(torch.count_nonzero(reference != observed))
        if count:
            delta = observed.double() - reference.double()
            relative = torch.linalg.vector_norm(delta) / max(
                float(torch.linalg.vector_norm(reference.double())), denominator_floor
            )
            report["differences"].append(
                {
                    "parameter": name,
                    "different_elements": count,
                    "max_abs": float(delta.abs().max()),
                    "relative_l2": float(relative),
                }
            )
            report["different_elements"] += count
    report["different_tensors"] = len(report["differences"])
    for metric in ("max_abs", "relative_l2"):
        report[metric] = max((row[metric] for row in report["differences"]), default=0.0)
    return report


def budget_failures(report: dict, budget: dict) -> list[str]:
    return [name for name in ("max_abs", "relative_l2") if report[name] > budget[name]]


def gradient_failures(report: dict, acceptance: dict, controls: list[dict]) -> list[str]:
    failures = budget_failures(report, acceptance["gradient"])
    if any(
        not re.fullmatch(acceptance["gradient_parameter_pattern"], row["parameter"])
        for row in report["differences"]
    ):
        failures.append("unexpected_parameter_family")
    if controls:
        envelope = {
            metric: max(
                acceptance["envelope_floor"][metric],
                acceptance["envelope_multiplier"] * max(row[metric] for row in controls),
            )
            for metric in ("max_abs", "relative_l2")
        }
        failures.extend("envelope_" + key for key in budget_failures(report, envelope))
    return failures


def capture_gradients(network) -> dict:
    return {
        name: None if p.grad is None else p.grad.detach().cpu().clone()
        for name, p in network.named_parameters()
    }


def install_gradients(network, gradients: dict) -> None:
    """Validate everything before changing any gradient; never alias the artifact."""
    parameters = dict(network.named_parameters())
    if set(parameters) != set(gradients) or not any(g is not None for g in gradients.values()):
        raise ValueError("invalid fixed-gradient coverage")
    for name, p in parameters.items():
        gradient = gradients[name]
        if gradient is not None and (
            gradient.shape != p.shape
            or gradient.dtype != p.dtype
            or not torch.isfinite(gradient).all()
        ):
            raise ValueError("invalid fixed-gradient specification")
    for name, p in parameters.items():
        gradient = gradients[name]
        p.grad = None if gradient is None else gradient.to(device=p.device).clone()


def optimizer_comparison(
    expected: dict, actual: dict, acceptance: dict, *, inactive_parameters: tuple[str, ...] = ()
) -> dict:
    fields = {"schema_version", "parameter_names", "master_weights", "optimizer"}
    if set(expected) != fields or set(actual) != fields:
        raise ValueError("unknown optimizer state fields")
    if set(expected) != set(actual) or expected["parameter_names"] != actual["parameter_names"]:
        raise ValueError("optimizer parameter coverage changed")
    if expected["schema_version"] != actual["schema_version"]:
        raise ValueError("optimizer schema changed")
    names = expected["parameter_names"]
    left, right = expected["optimizer"], actual["optimizer"]
    if set(left) != {"param_groups", "state"} or set(right) != {"param_groups", "state"}:
        raise ValueError("unknown AdamW state fields")
    if set(left) != set(right) or tree_digest(left["param_groups"]) != tree_digest(
        right["param_groups"]
    ):
        raise ValueError("optimizer hyperparameters changed")
    ids = [key for group in left["param_groups"] for key in group["params"]]
    if len(set(ids)) != len(ids) or len(ids) != len(names):
        raise ValueError("optimizer parameter mapping changed")
    if not set(inactive_parameters).issubset(names):
        raise ValueError("unknown inactive parameters")
    if set(left["state"]) != set(right["state"]) or not set(left["state"]).issubset(ids):
        raise ValueError("optimizer moments missing or unexpected")
    if any(
        name not in inactive_parameters and key not in left["state"]
        for name, key in zip(names, ids, strict=True)
    ):
        raise ValueError("active optimizer moments missing")
    active = [(name, key) for name, key in zip(names, ids, strict=True) if key in left["state"]]
    for _, key in active:
        if set(left["state"][key]) != {"step", "exp_avg", "exp_avg_sq"} or set(
            right["state"][key]
        ) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("unknown optimizer moment fields")
        if tree_digest(left["state"][key]["step"]) != tree_digest(right["state"][key]["step"]):
            raise ValueError("optimizer step differs")
    report = {
        "failed_checks": [],
        "parameters_without_moments": [
            name for name, key in zip(names, ids, strict=True) if key not in left["state"]
        ],
    }
    for section in ("master", "exp_avg", "exp_avg_sq"):
        pairs = []
        for source in (expected, actual):
            tensors = (
                source["master_weights"]
                if section == "master"
                else [source["optimizer"]["state"][key][section] for _, key in active]
            )
            if any(t.dtype != torch.float32 for t in tensors):
                raise ValueError("optimizer state must remain FP32")
            section_names = names if section == "master" else [name for name, _ in active]
            pairs.append(dict(zip(section_names, tensors, strict=True)))
        report[section] = tensor_comparison(
            *pairs, denominator_floor=acceptance["relative_l2_denominator_floor"]
        )
        report["failed_checks"].extend(
            section + ":" + key for key in budget_failures(report[section], acceptance[section])
        )
    return report
