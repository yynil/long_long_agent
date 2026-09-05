from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model.readout import (
    BINARY_VALUE_TASKS,
    RWKV7ValueReadout,
    project_action_logits,
)
from src.training.losses import (
    multitask_value_loss,
    weighted_action_cross_entropy,
    weighted_action_cross_entropy_from_hidden,
)


def test_action_projection_reuses_pretrained_head() -> None:
    class Network:
        def __init__(self):
            self.head = torch.nn.Linear(4, 7, bias=False)

    network = Network()
    hidden = torch.randn(2, 3, 4)
    calls = 0

    def count_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    hook = network.head.register_forward_hook(count_calls)
    logits = project_action_logits(network, hidden)
    hook.remove()
    assert logits.shape == (2, 3, 7)
    assert calls == 1
    with pytest.raises(ValueError, match="width"):
        project_action_logits(network, torch.randn(2, 3, 5))


def test_weighted_action_loss_matches_manual_and_masks_gradients() -> None:
    logits = torch.tensor(
        [[[3.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 4.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 1, 2]], dtype=torch.long)
    weights = torch.tensor([[0.0, 1.0, 2.0]])
    output = weighted_action_cross_entropy(logits, targets, weights)
    manual = (
        torch.nn.functional.cross_entropy(logits[:, 1].float(), targets[:, 1])
        + 2 * torch.nn.functional.cross_entropy(logits[:, 2].float(), targets[:, 2])
    ) / 3
    torch.testing.assert_close(output.total, manual)
    assert output.effective_tokens == 2
    assert output.weight_sum == 3
    output.total.backward()
    assert torch.count_nonzero(logits.grad[:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, 1:]) > 0


def test_hidden_action_loss_matches_full_projection_and_chunks_head_calls() -> None:
    torch.manual_seed(9)
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    head = torch.nn.Linear(4, 7, bias=False)
    targets = torch.tensor([[1, 2, -100, 4, 5], [0, 1, 2, 3, 4]])
    weights = torch.tensor([[1.0, 2.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0, 0.0]])
    full = weighted_action_cross_entropy(head(hidden), targets, weights)

    calls = []

    def counted_head(features):
        calls.append(features.shape[0])
        return head(features)

    chunked = weighted_action_cross_entropy_from_hidden(
        hidden,
        counted_head,
        targets,
        weights,
        chunk_tokens=2,
    )

    assert chunked.effective_tokens == 5
    assert calls == [2, 2, 1]
    assert torch.allclose(chunked.total, full.total, atol=1e-7, rtol=1e-6)
    chunked.total.backward()
    assert hidden.grad is not None
    assert hidden.grad[0, 2].abs().sum() == 0


def test_weighted_action_loss_rejects_empty_or_inconsistent_masks() -> None:
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="no effective"):
        weighted_action_cross_entropy(
            logits,
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.zeros(1, 2),
        )
    with pytest.raises(ValueError, match="ignored"):
        weighted_action_cross_entropy(
            logits,
            torch.tensor([[-100, 1]], dtype=torch.long),
            torch.ones(1, 2),
        )
    with pytest.raises(ValueError, match="finite"):
        weighted_action_cross_entropy(
            logits,
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[float("nan"), 1.0]]),
        )


def test_value_readout_shapes_masked_loss_and_gradients() -> None:
    torch.manual_seed(17)
    readout = RWKV7ValueReadout(4)
    hidden = torch.randn(3, 4, requires_grad=True)
    predictions = readout(hidden)
    assert predictions.outcome_logits.shape == (3, len(BINARY_VALUE_TASKS))
    assert predictions.remaining_turns_log.shape == (3,)

    targets = torch.zeros_like(predictions.outcome_logits)
    targets[0] = 1
    targets[2] = float("nan")
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, 4:] = True
    remaining = torch.tensor([3.0, 8.0, float("nan")])
    remaining_mask = torch.tensor([True, True, False])
    output = multitask_value_loss(
        predictions,
        targets,
        mask,
        remaining,
        remaining_mask,
    )
    assert torch.isfinite(output.total)
    output.total.backward()
    assert torch.count_nonzero(readout.outcomes.weight.grad) > 0
    assert torch.count_nonzero(readout.remaining_turns_head.weight.grad) > 0
    assert torch.count_nonzero(hidden.grad[:2]) > 0
    assert torch.count_nonzero(hidden.grad[2]) == 0


def test_value_loss_rejects_invalid_active_labels_and_empty_batch() -> None:
    readout = RWKV7ValueReadout(4)
    predictions = readout(torch.zeros(2, 4))
    targets = torch.zeros_like(predictions.outcome_logits)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    remaining = torch.zeros(2)
    remaining_mask = torch.zeros(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="no positively weighted"):
        multitask_value_loss(predictions, targets, mask, remaining, remaining_mask)

    mask[0, 0] = True
    targets[0, 0] = 2
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        multitask_value_loss(predictions, targets, mask, remaining, remaining_mask)
