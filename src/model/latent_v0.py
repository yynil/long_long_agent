"""V0 fixed latent-control embeddings and decision-local recurrent rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .rwkv7_stateful import stateful_forward_embeddings
from .slow_fast_state import SlowFastRWKVState


class RWKV7V0LatentControl(nn.Module):
    """Learned ``e_latent + e_depth(k)`` inputs without continuous feedback."""

    def __init__(self, n_embd: int, max_depth: int = 16) -> None:
        super().__init__()
        if n_embd <= 0 or max_depth <= 0:
            raise ValueError("n_embd and max_depth must be positive")
        self.n_embd = n_embd
        self.max_depth = max_depth
        self.latent_embedding = nn.Parameter(torch.zeros(n_embd))
        self.depth_embedding = nn.Parameter(torch.zeros(max_depth, n_embd))

    @torch.no_grad()
    def initialize_from_token_embeddings(self, token_embedding_weight: Any) -> None:
        if tuple(token_embedding_weight.shape[1:]) != (self.n_embd,):
            raise ValueError("token embedding width does not match latent control")
        self.latent_embedding.copy_(token_embedding_weight.float().mean(dim=0))
        self.depth_embedding.zero_()

    def embeddings(
        self,
        *,
        batch_size: int,
        start_depth: int,
        steps: int,
        device: Any,
        dtype: Any,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(start_depth, bool) or isinstance(steps, bool):
            raise TypeError("depth values must be integers")
        if start_depth < 0 or steps < 0 or start_depth + steps > self.max_depth:
            raise ValueError("requested latent depth is outside the configured range")
        depth = self.depth_embedding[start_depth : start_depth + steps]
        inputs = self.latent_embedding.unsqueeze(0) + depth
        return inputs.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)


@dataclass(frozen=True)
class V0LatentRollout:
    decision: SlowFastRWKVState
    hidden: Any

    @property
    def total_depth(self) -> int:
        return self.decision.fast_steps


def run_v0_latent_steps(
    network: Any,
    control: RWKV7V0LatentControl,
    decision: SlowFastRWKVState,
    *,
    steps: int,
) -> V0LatentRollout:
    """Advance fast state only; the LM head is intentionally never called."""
    if control.n_embd != decision.fast.spec.n_embd:
        raise ValueError("latent control width does not match RWKV state")
    inputs = control.embeddings(
        batch_size=decision.fast.spec.batch_size,
        start_depth=decision.fast_steps,
        steps=steps,
        device=decision.fast.device,
        dtype=decision.fast.previous_x_dtype,
    )
    if steps == 0:
        return V0LatentRollout(decision=decision, hidden=inputs)

    starts = torch.zeros(
        decision.fast.spec.batch_size,
        steps,
        device=decision.fast.device,
        dtype=torch.uint8,
    )
    hidden, next_fast = stateful_forward_embeddings(
        network,
        inputs,
        state=decision.fast,
        sequence_start_mask=starts,
        detach_state=False,
    )
    return V0LatentRollout(
        decision=decision.with_fast(next_fast, steps=steps),
        hidden=hidden,
    )
