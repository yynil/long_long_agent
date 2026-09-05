from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model.rwkv7_stateful import state_passing, state_passing_reference


def make_inputs(*, timesteps: int = 5, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(71)
    state = torch.randn(2, 2, 4, 4, generator=generator) * 0.1
    values = [torch.randn(2, timesteps, 8, generator=generator) * 0.1 for _ in range(6)]
    if requires_grad:
        state.requires_grad_(True)
        for value in values:
            value.requires_grad_(True)
    return state, values


def test_reference_short_windows_compose_with_identical_gradients() -> None:
    starts = torch.zeros(2, 5, dtype=torch.uint8)
    starts[0, 3] = 1
    base_state, base_values = make_inputs(requires_grad=False)
    whole_state = base_state.clone().requires_grad_(True)
    whole_values = [value.clone().requires_grad_(True) for value in base_values]
    whole_y, whole_final = state_passing_reference(*whole_values, whole_state, starts)
    (whole_y.square().sum() + whole_final.square().sum()).backward()

    split_state = base_state.clone().requires_grad_(True)
    split_values = [value.clone().requires_grad_(True) for value in base_values]
    first_y, first_final = state_passing_reference(
        *(value[:, :2] for value in split_values),
        split_state,
        starts[:, :2],
    )
    second_y, split_final = state_passing_reference(
        *(value[:, 2:] for value in split_values),
        first_final,
        starts[:, 2:],
    )
    split_y = torch.cat((first_y, second_y), dim=1)
    (split_y.square().sum() + split_final.square().sum()).backward()

    torch.testing.assert_close(whole_y, split_y, rtol=0, atol=0)
    torch.testing.assert_close(whole_final, split_final, rtol=0, atol=0)
    for whole, split in zip(
        [whole_state, *whole_values],
        [split_state, *split_values],
    ):
        torch.testing.assert_close(whole.grad, split.grad, rtol=0, atol=0)


def test_reference_reset_cuts_state_and_token_gradients() -> None:
    state, values = make_inputs(timesteps=4, requires_grad=True)
    starts = torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=torch.uint8)
    output, final_state = state_passing_reference(*values, state, starts)
    (output[:, 2:].square().sum() + final_state.square().sum()).backward()

    assert torch.count_nonzero(state.grad) == 0
    for value in values:
        assert torch.count_nonzero(value.grad[:, :2]) == 0
        assert torch.count_nonzero(value.grad[:, 2:]) > 0


def test_public_state_passing_uses_reference_for_cpu_short_window() -> None:
    state, values = make_inputs(timesteps=1)
    starts = torch.zeros(2, 1, dtype=torch.uint8)
    expected_y, expected_state = state_passing_reference(*values, state, starts)
    actual_y, actual_state = state_passing(*values, state, starts)
    torch.testing.assert_close(actual_y, expected_y)
    torch.testing.assert_close(actual_state, expected_state)
