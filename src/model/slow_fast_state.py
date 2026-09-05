"""Decision-scoped ownership for RWKV-7 slow and fast recurrent states."""

from __future__ import annotations

from dataclasses import dataclass

from .state import RWKVState


@dataclass(frozen=True)
class SlowFastRWKVState:
    """Keep confirmed history separate from decision-local latent hypotheses.

    A fast state is always deep-cloned from a slow state at a decision boundary.
    The class deliberately exposes no fast-to-slow commit operation. A caller may
    advance slow state only with a distinct state produced by consuming confirmed
    action/observation tokens through the normal RWKV path.
    """

    slow: RWKVState
    fast: RWKVState
    decision_id: str
    slow_revision: int = 0
    fast_steps: int = 0

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id must be non-empty")
        if self.slow_revision < 0:
            raise ValueError("slow_revision must be non-negative")
        if self.fast_steps < 0:
            raise ValueError("fast_steps must be non-negative")
        if self.slow.spec != self.fast.spec:
            raise ValueError("slow and fast state specs must match")
        if self.slow.shares_storage_with(self.fast):
            raise ValueError("slow and fast states must not share tensor storage")

    @classmethod
    def begin_decision(
        cls,
        slow: RWKVState,
        *,
        decision_id: str,
        slow_revision: int = 0,
        detach_fast: bool = False,
    ) -> SlowFastRWKVState:
        return cls(
            slow=slow,
            fast=slow.clone(detach=detach_fast),
            decision_id=decision_id,
            slow_revision=slow_revision,
        )

    def with_fast(self, next_fast: RWKVState, *, steps: int = 1) -> SlowFastRWKVState:
        """Replace only the decision-local state after one or more latent steps."""
        if steps <= 0:
            raise ValueError("fast state advancement must consume at least one step")
        return SlowFastRWKVState(
            slow=self.slow,
            fast=next_fast,
            decision_id=self.decision_id,
            slow_revision=self.slow_revision,
            fast_steps=self.fast_steps + steps,
        )

    def advance_slow(
        self,
        next_slow: RWKVState,
        *,
        next_decision_id: str,
        confirmed_token_count: int,
        detach_fast: bool = False,
    ) -> SlowFastRWKVState:
        """Start the next decision from a confirmed normal-token state update."""
        if confirmed_token_count <= 0:
            raise ValueError("slow state advancement requires confirmed tokens")
        if next_slow.spec != self.slow.spec:
            raise ValueError("next slow state spec does not match the current state")
        if next_slow.shares_storage_with(self.fast):
            raise ValueError("fast state cannot be committed as the next slow state")
        return self.begin_decision(
            next_slow,
            decision_id=next_decision_id,
            slow_revision=self.slow_revision + 1,
            detach_fast=detach_fast,
        )
