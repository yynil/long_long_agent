from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model.latent_v0 import RWKV7V0LatentControl, run_v0_latent_steps
from src.model.slow_fast_state import SlowFastRWKVState
from src.model.state import RWKVState, RWKVStateSpec


def make_decision() -> SlowFastRWKVState:
    spec = RWKVStateSpec(n_layer=1, n_embd=8, n_head=2, head_size=4, batch_size=2)
    slow = RWKVState.zeros(spec, device="cpu", dtype=torch.float32)
    return SlowFastRWKVState.begin_decision(slow, decision_id="episode:decision")


def test_latent_control_builds_fixed_inputs_and_initializes_from_token_mean() -> None:
    control = RWKV7V0LatentControl(n_embd=8, max_depth=4)
    token_weights = torch.arange(40, dtype=torch.float32).view(5, 8)
    control.initialize_from_token_embeddings(token_weights)
    embeddings = control.embeddings(
        batch_size=2,
        start_depth=1,
        steps=2,
        device="cpu",
        dtype=torch.bfloat16,
    )
    assert embeddings.shape == (2, 2, 8)
    assert embeddings.dtype == torch.bfloat16
    torch.testing.assert_close(embeddings[0], embeddings[1])
    torch.testing.assert_close(
        control.latent_embedding,
        token_weights.mean(dim=0),
    )
    assert torch.count_nonzero(control.depth_embedding) == 0


def test_k_zero_is_an_exact_anchor_and_does_not_call_network() -> None:
    class ForbiddenNetwork:
        def __getattribute__(self, name):
            raise AssertionError(f"network was accessed for K=0: {name}")

    decision = make_decision()
    control = RWKV7V0LatentControl(n_embd=8, max_depth=4)
    rollout = run_v0_latent_steps(ForbiddenNetwork(), control, decision, steps=0)
    assert rollout.decision is decision
    assert rollout.total_depth == 0
    assert rollout.hidden.shape == (2, 0, 8)


def test_latent_control_rejects_depth_and_width_errors() -> None:
    decision = make_decision()
    control = RWKV7V0LatentControl(n_embd=8, max_depth=4)
    with pytest.raises(ValueError, match="range"):
        control.embeddings(
            batch_size=2,
            start_depth=3,
            steps=2,
            device="cpu",
            dtype=torch.float32,
        )
    with pytest.raises(TypeError, match="integers"):
        control.embeddings(
            batch_size=2,
            start_depth=0,
            steps=True,
            device="cpu",
            dtype=torch.float32,
        )
    wrong_width = RWKV7V0LatentControl(n_embd=16, max_depth=4)
    with pytest.raises(ValueError, match="width"):
        run_v0_latent_steps(object(), wrong_width, decision, steps=0)
