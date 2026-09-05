"""Minimal fail-closed packed Agent SFT training loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch

from .losses import WeightedTokenLoss, weighted_action_cross_entropy_from_hidden


@dataclass(frozen=True)
class SFTStepMetrics:
    loss: float
    gradient_norm: float
    effective_loss_tokens: int
    loss_weight_sum: float
    sequences: int
    real_tokens: int
    aligned_tokens: int
    alignment_tokens: int
    optimizer_step: int


@dataclass(frozen=True)
class SFTProgress:
    optimizer_steps: int
    sequences: int
    real_tokens: int
    aligned_tokens: int
    effective_loss_tokens: int


def move_packed_batch(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def validate_packed_tensor_batch(batch: dict[str, Any]) -> None:
    required = {
        "input_ids",
        "targets",
        "loss_weights",
        "cu_seqlens",
        "sequence_start_mask",
        "valid_token_mask",
        "segment_ids",
        "sample_ids",
    }
    missing = required - set(batch)
    if missing:
        raise ValueError(f"packed batch is missing fields: {sorted(missing)}")

    input_ids = batch["input_ids"]
    targets = batch["targets"]
    loss_weights = batch["loss_weights"]
    starts = batch["sequence_start_mask"]
    valid = batch["valid_token_mask"]
    segment_ids = batch["segment_ids"]
    cu_seqlens = batch["cu_seqlens"]
    tensors = (input_ids, targets, loss_weights, starts, valid, segment_ids, cu_seqlens)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("packed tensor fields must be torch.Tensor instances")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("one packed row must have input_ids shape [1, T]")
    if any(
        tensor.shape != input_ids.shape
        for tensor in (targets, loss_weights, starts, valid, segment_ids)
    ):
        raise ValueError("packed token tensors must all have shape [1, T]")
    if input_ids.dtype != torch.long or targets.dtype != torch.long:
        raise TypeError("input_ids and targets must be torch.long")
    if not loss_weights.dtype.is_floating_point:
        raise TypeError("loss_weights must be floating point")
    if starts.dtype != torch.uint8:
        raise TypeError("sequence_start_mask must be uint8")
    if valid.dtype != torch.bool:
        raise TypeError("valid_token_mask must be bool")
    if segment_ids.dtype != torch.int32:
        raise TypeError("segment_ids must be int32")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.ndim != 1:
        raise TypeError("cu_seqlens must be one-dimensional int32")
    if any(tensor.device != input_ids.device for tensor in tensors[:-1]):
        raise ValueError("packed token tensors must share a device")
    if not torch.all(torch.isfinite(loss_weights)) or torch.any(loss_weights < 0):
        raise ValueError("loss_weights must be finite and non-negative")

    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    sample_ids = tuple(batch["sample_ids"])
    if len(offsets) != len(sample_ids) + 1 or not offsets or offsets[0] != 0:
        raise ValueError("cu_seqlens and sample_ids disagree")
    if len(set(sample_ids)) != len(sample_ids) or any(not value for value in sample_ids):
        raise ValueError("sample_ids must be non-empty and unique within a row")
    if any(right <= left for left, right in pairwise(offsets)):
        raise ValueError("cu_seqlens must be strictly increasing")
    real_tokens = offsets[-1]
    aligned_tokens = input_ids.shape[1]
    if real_tokens > aligned_tokens:
        raise ValueError("cu_seqlens exceeds aligned row length")

    expected_valid = torch.arange(aligned_tokens, device=valid.device) < real_tokens
    if not torch.equal(valid[0], expected_valid):
        raise ValueError("valid_token_mask must be one contiguous real-token prefix")
    expected_starts = torch.zeros(aligned_tokens, dtype=torch.uint8, device=starts.device)
    start_indices = list(offsets[:-1])
    if real_tokens < aligned_tokens:
        start_indices.append(real_tokens)
    expected_starts[start_indices] = 1
    if not torch.equal(starts[0], expected_starts):
        raise ValueError("sequence_start_mask disagrees with cu_seqlens")
    if torch.any((loss_weights > 0) & ~valid):
        raise ValueError("alignment tokens cannot contribute loss")
    if torch.any((targets == -100) & (loss_weights > 0)):
        raise ValueError("ignored targets cannot contribute loss")
    if real_tokens < aligned_tokens:
        if torch.any(targets[0, real_tokens:] != -100):
            raise ValueError("alignment targets must use ignore_index=-100")
        if torch.any(segment_ids[0, real_tokens:] != -1):
            raise ValueError("alignment segment IDs must be -1")


class PackedAgentSFTTrainer:
    """A framework-light trainer for the patched RWKV packed feature interface."""

    def __init__(
        self,
        network: Any,
        optimizer: torch.optim.Optimizer,
        *,
        head_chunk_tokens: int = 128,
        gradient_clip_norm: float | None = None,
    ) -> None:
        if not hasattr(network, "_forward_features") or not hasattr(network, "head"):
            raise TypeError("network must expose _forward_features and head")
        if not isinstance(head_chunk_tokens, int) or head_chunk_tokens <= 0:
            raise ValueError("head_chunk_tokens must be a positive integer")
        if gradient_clip_norm is not None and (
            not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be finite and positive")
        self.network = network
        self.optimizer = optimizer
        self.head_chunk_tokens = head_chunk_tokens
        self.gradient_clip_norm = gradient_clip_norm
        self._optimizer_steps = 0
        self._sequences = 0
        self._real_tokens = 0
        self._aligned_tokens = 0
        self._effective_loss_tokens = 0

    @property
    def progress(self) -> SFTProgress:
        return SFTProgress(
            optimizer_steps=self._optimizer_steps,
            sequences=self._sequences,
            real_tokens=self._real_tokens,
            aligned_tokens=self._aligned_tokens,
            effective_loss_tokens=self._effective_loss_tokens,
        )

    def compute_loss(self, batch: dict[str, Any]) -> WeightedTokenLoss:
        validate_packed_tensor_batch(batch)
        hidden = self.network._forward_features(
            batch["input_ids"],
            batch["sequence_start_mask"],
        )
        if hidden.ndim != 3 or hidden.shape[:2] != batch["input_ids"].shape:
            raise ValueError("network features must have shape [1, T, C]")
        return weighted_action_cross_entropy_from_hidden(
            hidden,
            self.network.head,
            batch["targets"],
            batch["loss_weights"],
            chunk_tokens=self.head_chunk_tokens,
        )

    def train_step(self, batch: dict[str, Any]) -> SFTStepMetrics:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            result = self.compute_loss(batch)
            if not torch.isfinite(result.total):
                raise FloatingPointError("non-finite packed SFT loss")
            result.total.backward()
            parameters = [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ]
            if not parameters or not any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("packed SFT step produced no optimizer gradients")
            max_norm = self.gradient_clip_norm if self.gradient_clip_norm is not None else math.inf
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm,
                error_if_nonfinite=True,
            )
            gradient_norm_value = float(gradient_norm.detach())
            self.optimizer.step()
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        real_tokens = int(batch["cu_seqlens"][-1])
        aligned_tokens = int(batch["input_ids"].numel())
        sequences = len(batch["sample_ids"])
        self._optimizer_steps += 1
        self._sequences += sequences
        self._real_tokens += real_tokens
        self._aligned_tokens += aligned_tokens
        self._effective_loss_tokens += result.effective_tokens
        return SFTStepMetrics(
            loss=float(result.total.detach()),
            gradient_norm=gradient_norm_value,
            effective_loss_tokens=result.effective_tokens,
            loss_weight_sum=result.weight_sum,
            sequences=sequences,
            real_tokens=real_tokens,
            aligned_tokens=aligned_tokens,
            alignment_tokens=aligned_tokens - real_tokens,
            optimizer_step=self._optimizer_steps,
        )
