from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from src.model.slow_fast_state import SlowFastRWKVState
from src.model.state import RWKVState, RWKVStateSpec


def make_state(*, batch_size: int = 1, requires_grad: bool = False) -> RWKVState:
    spec = RWKVStateSpec(
        n_layer=2,
        n_embd=8,
        n_head=2,
        head_size=4,
        batch_size=batch_size,
    )
    state = RWKVState.zeros(spec, device="cpu", dtype=torch.bfloat16)
    if requires_grad:
        state.layers[0].time_mix_previous_x.requires_grad_(True)
    return state


def test_begin_decision_clones_fast_without_storage_alias() -> None:
    slow = make_state(requires_grad=True)
    decision = SlowFastRWKVState.begin_decision(slow, decision_id="episode-1:decision-3")

    assert decision.slow is slow
    assert not decision.slow.shares_storage_with(decision.fast)
    assert decision.fast.layers[0].time_mix_previous_x.grad_fn is not None

    decision.fast.layers[0].time_mix_previous_x.data.add_(1)
    assert torch.count_nonzero(decision.slow.layers[0].time_mix_previous_x) == 0


def test_begin_decision_detach_policy_is_explicit() -> None:
    slow = make_state(requires_grad=True)
    decision = SlowFastRWKVState.begin_decision(
        slow,
        decision_id="episode-1:decision-3",
        detach_fast=True,
    )
    assert decision.fast.layers[0].time_mix_previous_x.grad_fn is None


def test_fast_evolution_preserves_slow_and_tracks_depth() -> None:
    decision = SlowFastRWKVState.begin_decision(
        make_state(), decision_id="episode-1:decision-3", slow_revision=7
    )
    next_fast = decision.fast.clone()
    next_fast.layers[0].wkv_matrix.add_(1)
    evolved = decision.with_fast(next_fast, steps=4)

    assert evolved.slow is decision.slow
    assert evolved.slow_revision == 7
    assert evolved.fast_steps == 4
    assert torch.count_nonzero(evolved.slow.layers[0].wkv_matrix) == 0
    assert torch.count_nonzero(evolved.fast.layers[0].wkv_matrix) > 0


def test_advance_slow_discards_old_fast_and_starts_new_decision() -> None:
    decision = SlowFastRWKVState.begin_decision(
        make_state(), decision_id="episode-1:decision-3", slow_revision=7
    )
    next_slow = make_state()
    next_slow.layers[1].channel_mix_previous_x.add_(2)
    advanced = decision.advance_slow(
        next_slow,
        next_decision_id="episode-1:decision-4",
        confirmed_token_count=23,
    )

    assert advanced.slow is next_slow
    assert advanced.slow_revision == 8
    assert advanced.fast_steps == 0
    assert advanced.decision_id == "episode-1:decision-4"
    assert not advanced.slow.shares_storage_with(advanced.fast)
    advanced.fast.layers[1].channel_mix_previous_x.zero_()
    assert torch.count_nonzero(advanced.slow.layers[1].channel_mix_previous_x) > 0


def test_slow_fast_container_rejects_aliases_and_invalid_transitions() -> None:
    slow = make_state()
    with pytest.raises(ValueError, match="share"):
        SlowFastRWKVState(slow, slow, decision_id="decision")

    decision = SlowFastRWKVState.begin_decision(slow, decision_id="decision")
    with pytest.raises(ValueError, match="committed"):
        decision.advance_slow(
            decision.fast,
            next_decision_id="next",
            confirmed_token_count=1,
        )
    with pytest.raises(ValueError, match="confirmed"):
        decision.advance_slow(
            make_state(),
            next_decision_id="next",
            confirmed_token_count=0,
        )
    with pytest.raises(ValueError, match="at least one"):
        decision.with_fast(decision.fast.clone(), steps=0)

    mismatched = make_state(batch_size=2)
    with pytest.raises(ValueError, match="spec"):
        decision.with_fast(mismatched)

    aliased_layer = replace(
        decision.fast.layers[0],
        time_mix_previous_x=slow.layers[0].time_mix_previous_x,
    )
    partially_aliased = RWKVState(
        spec=slow.spec,
        layers=(aliased_layer, decision.fast.layers[1]),
    )
    with pytest.raises(ValueError, match="share"):
        decision.with_fast(partially_aliased)
