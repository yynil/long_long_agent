"""Masked Agent action and environment-value losses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from src.model.readout import BINARY_VALUE_TASKS, VALUE_TASKS, ValuePredictions


@dataclass(frozen=True)
class WeightedTokenLoss:
    total: Any
    effective_tokens: int
    weight_sum: float


@dataclass(frozen=True)
class MultitaskValueLoss:
    total: Any
    per_task: dict[str, Any]
    active_counts: dict[str, int]


def weighted_action_cross_entropy(
    logits: Any,
    targets: Any,
    loss_weights: Any,
    *,
    ignore_index: int = -100,
) -> WeightedTokenLoss:
    if logits.ndim != 3:
        raise ValueError("action logits must have shape [B, T, V]")
    if targets.shape != logits.shape[:2] or loss_weights.shape != targets.shape:
        raise ValueError("action targets and loss weights must have shape [B, T]")
    if targets.dtype != torch.long:
        raise TypeError("action targets must be torch.long")
    if not loss_weights.dtype.is_floating_point:
        raise TypeError("action loss weights must be floating point")
    if not torch.all(torch.isfinite(loss_weights)):
        raise ValueError("action loss weights must be finite")
    if torch.any(loss_weights < 0):
        raise ValueError("action loss weights must be non-negative")
    if torch.any((targets == ignore_index) & (loss_weights > 0)):
        raise ValueError("ignored action targets must have zero loss weight")

    active = (targets != ignore_index) & (loss_weights > 0)
    effective_tokens = int(active.sum())
    if effective_tokens == 0:
        raise ValueError("action batch has no effective loss tokens")
    selected_targets = targets[active]
    if torch.any(selected_targets < 0) or torch.any(selected_targets >= logits.shape[-1]):
        raise ValueError("active action target is outside the vocabulary")
    selected_weights = loss_weights[active].float()
    if not torch.all(torch.isfinite(logits[active])):
        raise ValueError("active action logits must be finite")
    weight_sum = selected_weights.sum()
    losses = F.cross_entropy(logits[active].float(), selected_targets, reduction="none")
    total = (losses * selected_weights).sum() / weight_sum
    return WeightedTokenLoss(
        total=total,
        effective_tokens=effective_tokens,
        weight_sum=float(weight_sum.detach()),
    )


def weighted_action_cross_entropy_from_hidden(
    hidden: Any,
    head: Any,
    targets: Any,
    loss_weights: Any,
    *,
    chunk_tokens: int = 128,
    ignore_index: int = -100,
) -> WeightedTokenLoss:
    """Project only supervised hidden states, bounding the large-vocabulary logits tensor."""
    if hidden.ndim != 3:
        raise ValueError("action hidden must have shape [B, T, C]")
    if targets.shape != hidden.shape[:2] or loss_weights.shape != targets.shape:
        raise ValueError("action targets and loss weights must have shape [B, T]")
    if targets.dtype != torch.long:
        raise TypeError("action targets must be torch.long")
    if not loss_weights.dtype.is_floating_point:
        raise TypeError("action loss weights must be floating point")
    if hidden.device != targets.device or hidden.device != loss_weights.device:
        raise ValueError("action hidden, targets, and loss weights must share a device")
    if not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    if not torch.all(torch.isfinite(loss_weights)):
        raise ValueError("action loss weights must be finite")
    if torch.any(loss_weights < 0):
        raise ValueError("action loss weights must be non-negative")
    if torch.any((targets == ignore_index) & (loss_weights > 0)):
        raise ValueError("ignored action targets must have zero loss weight")

    active = (targets != ignore_index) & (loss_weights > 0)
    effective_tokens = int(active.sum())
    if effective_tokens == 0:
        raise ValueError("action batch has no effective loss tokens")
    selected_hidden = hidden[active]
    selected_targets = targets[active]
    selected_weights = loss_weights[active].float()
    if not torch.all(torch.isfinite(selected_hidden)):
        raise ValueError("active action hidden states must be finite")

    weight_sum = selected_weights.sum()
    numerators = []
    for offset in range(0, effective_tokens, chunk_tokens):
        end = min(offset + chunk_tokens, effective_tokens)
        logits = head(selected_hidden[offset:end])
        if logits.ndim != 2 or logits.shape[0] != end - offset:
            raise ValueError("action head must return logits with shape [N, V]")
        if not torch.all(torch.isfinite(logits)):
            raise ValueError("active action logits must be finite")
        chunk_targets = selected_targets[offset:end]
        if torch.any(chunk_targets < 0) or torch.any(chunk_targets >= logits.shape[-1]):
            raise ValueError("active action target is outside the vocabulary")
        losses = F.cross_entropy(logits.float(), chunk_targets, reduction="none")
        numerators.append((losses * selected_weights[offset:end]).sum())
    total = torch.stack(numerators).sum() / weight_sum
    return WeightedTokenLoss(
        total=total,
        effective_tokens=effective_tokens,
        weight_sum=float(weight_sum.detach()),
    )


def _validated_task_weights(task_weights: dict[str, float] | None) -> dict[str, float]:
    if task_weights is None:
        return {task: 1.0 for task in VALUE_TASKS}
    if set(task_weights) != set(VALUE_TASKS):
        raise ValueError("value task weights must contain the exact versioned task set")
    result = {task: float(weight) for task, weight in task_weights.items()}
    if any(not math.isfinite(weight) or weight < 0 for weight in result.values()) or not any(
        result.values()
    ):
        raise ValueError("value task weights must be non-negative with at least one positive")
    return result


def multitask_value_loss(
    predictions: ValuePredictions,
    binary_targets: Any,
    binary_mask: Any,
    remaining_turns: Any,
    remaining_turns_mask: Any,
    *,
    task_weights: dict[str, float] | None = None,
) -> MultitaskValueLoss:
    if binary_targets.shape != predictions.outcome_logits.shape:
        raise ValueError("binary value targets must match outcome logits")
    if binary_mask.shape != binary_targets.shape or binary_mask.dtype != torch.bool:
        raise ValueError("binary value mask must be bool and match targets")
    if remaining_turns.shape != predictions.remaining_turns_log.shape:
        raise ValueError("remaining-turn targets must match remaining-turn predictions")
    if remaining_turns_mask.shape != remaining_turns.shape:
        raise ValueError("remaining-turn mask must match targets")
    if remaining_turns_mask.dtype != torch.bool:
        raise TypeError("remaining-turn mask must be bool")

    weights = _validated_task_weights(task_weights)
    per_task: dict[str, Any] = {}
    active_counts: dict[str, int] = {}
    weighted_losses = []
    active_weight_sum = 0.0
    for index, task in enumerate(BINARY_VALUE_TASKS):
        active = binary_mask[..., index]
        count = int(active.sum())
        active_counts[task] = count
        if count == 0:
            continue
        targets = binary_targets[..., index][active].float()
        logits = predictions.outcome_logits[..., index][active].float()
        if not torch.all(torch.isfinite(targets)) or torch.any((targets < 0) | (targets > 1)):
            raise ValueError(f"active {task} targets must be finite values in [0, 1]")
        if not torch.all(torch.isfinite(logits)):
            raise ValueError(f"active {task} logits must be finite")
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
        )
        per_task[task] = loss
        if weights[task] > 0:
            weighted_losses.append(loss * weights[task])
            active_weight_sum += weights[task]

    remaining_count = int(remaining_turns_mask.sum())
    active_counts["remaining_turns"] = remaining_count
    if remaining_count:
        targets = remaining_turns[remaining_turns_mask].float()
        logits = predictions.remaining_turns_log[remaining_turns_mask].float()
        if not torch.all(torch.isfinite(targets)) or torch.any(targets < 0):
            raise ValueError("active remaining-turn targets must be finite and non-negative")
        if not torch.all(torch.isfinite(logits)):
            raise ValueError("active remaining-turn predictions must be finite")
        remaining_loss = F.smooth_l1_loss(
            logits,
            torch.log1p(targets),
        )
        per_task["remaining_turns"] = remaining_loss
        if weights["remaining_turns"] > 0:
            weighted_losses.append(remaining_loss * weights["remaining_turns"])
            active_weight_sum += weights["remaining_turns"]

    if not weighted_losses or active_weight_sum <= 0:
        raise ValueError("value batch has no positively weighted active labels")
    total = torch.stack(weighted_losses).sum() / active_weight_sum
    return MultitaskValueLoss(total=total, per_task=per_task, active_counts=active_counts)
