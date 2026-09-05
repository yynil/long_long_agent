"""Action projection and minimal environment-grounded RWKV-7 value readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

BINARY_VALUE_TASKS = (
    "parse_valid",
    "exec_valid",
    "new_information",
    "immediate_progress",
    "regression",
    "next_milestone_within_4_turns",
    "next_milestone_within_16_turns",
    "terminal_success",
)
REMAINING_TURNS_TASK = "remaining_turns"
VALUE_TASKS = (*BINARY_VALUE_TASKS, REMAINING_TURNS_TASK)


@dataclass(frozen=True)
class ValuePredictions:
    outcome_logits: Any
    remaining_turns_log: Any

    def probabilities(self):
        return torch.sigmoid(self.outcome_logits)

    def remaining_turns(self):
        return torch.expm1(self.remaining_turns_log.float().clamp_min(0))


def project_action_logits(network: Any, hidden: Any):
    if hidden.ndim < 2:
        raise ValueError("action hidden must include a feature dimension")
    in_features = getattr(network.head, "in_features", None)
    if in_features is not None and hidden.shape[-1] != in_features:
        raise ValueError("action hidden width does not match the pretrained LM head")
    return network.head(hidden)


class RWKV7ValueReadout(nn.Module):
    """Linear V0 critic over top-layer hidden without flattening recurrent state."""

    def __init__(self, n_embd: int, *, init_gain: float = 0.01) -> None:
        super().__init__()
        if n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if init_gain <= 0:
            raise ValueError("init_gain must be positive")
        self.n_embd = n_embd
        self.init_gain = init_gain
        self.outcomes = nn.Linear(n_embd, len(BINARY_VALUE_TASKS))
        self.remaining_turns_head = nn.Linear(n_embd, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.outcomes.weight, gain=self.init_gain)
        nn.init.zeros_(self.outcomes.bias)
        nn.init.orthogonal_(self.remaining_turns_head.weight, gain=self.init_gain)
        nn.init.zeros_(self.remaining_turns_head.bias)

    def forward(self, hidden: Any) -> ValuePredictions:
        if hidden.ndim < 2 or hidden.shape[-1] != self.n_embd:
            raise ValueError("value hidden shape does not match readout width")
        features = hidden.to(dtype=self.outcomes.weight.dtype)
        return ValuePredictions(
            outcome_logits=self.outcomes(features),
            remaining_turns_log=self.remaining_turns_head(features).squeeze(-1),
        )
