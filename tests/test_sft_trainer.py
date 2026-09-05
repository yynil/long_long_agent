from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.training.episode_collator import PackedEpisodeCollator
from src.training.packing import CausalSequence
from src.training.sft_trainer import (
    PackedAgentSFTTrainer,
    move_packed_batch,
    validate_packed_tensor_batch,
)


class TinyPackedModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, width: int = 12) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.projection = torch.nn.Linear(width, width)
        self.head = torch.nn.Linear(width, vocab_size, bias=False)
        self.last_sequence_start_count = 0

    def _forward_features(self, input_ids, sequence_start_mask):
        if sequence_start_mask.dtype != torch.uint8:
            raise TypeError("expected uint8 starts")
        self.last_sequence_start_count = int(sequence_start_mask.sum())
        return torch.tanh(self.projection(self.embedding(input_ids)))


def make_batch(count: int = 8):
    sequences = [
        CausalSequence(
            sample_id=f"sample-{index}",
            input_ids=(1, index % 4 + 2),
            targets=(index % 4 + 2, index % 4 + 6),
            loss_weights=(0.0, 1.0),
        )
        for index in range(count)
    ]
    collator = PackedEpisodeCollator(object(), align_to=16, max_pack_tokens=64)
    return collator.collate_tokenized(sequences)


def test_packed_trainer_reduces_loss_and_tracks_exact_counters() -> None:
    torch.manual_seed(4)
    model = TinyPackedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0)
    trainer = PackedAgentSFTTrainer(
        model,
        optimizer,
        head_chunk_tokens=3,
        gradient_clip_norm=5.0,
    )
    batch = make_batch()
    with torch.no_grad():
        initial = float(trainer.compute_loss(batch).total)
    metrics = None
    for _ in range(30):
        metrics = trainer.train_step(batch)
    with torch.no_grad():
        final = float(trainer.compute_loss(batch).total)

    assert metrics is not None
    assert final < initial * 0.05
    assert model.last_sequence_start_count == 8
    assert metrics.sequences == 8
    assert metrics.real_tokens == 16
    assert metrics.aligned_tokens == 16
    assert metrics.effective_loss_tokens == 8
    assert trainer.progress.optimizer_steps == 30
    assert trainer.progress.sequences == 240
    assert trainer.progress.effective_loss_tokens == 240


def test_batch_validation_and_failed_step_do_not_advance_progress() -> None:
    model = TinyPackedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    trainer = PackedAgentSFTTrainer(model, optimizer)
    batch = make_batch(2)
    batch["loss_weights"].zero_()

    with pytest.raises(ValueError, match="no effective loss tokens"):
        trainer.train_step(batch)
    assert trainer.progress.optimizer_steps == 0
    assert all(parameter.grad is None for parameter in model.parameters())

    invalid = make_batch(2)
    invalid["sequence_start_mask"].zero_()
    with pytest.raises(ValueError, match="disagrees"):
        validate_packed_tensor_batch(invalid)


def test_move_packed_batch_preserves_metadata() -> None:
    batch = make_batch(2)
    moved = move_packed_batch(batch, "cpu")
    assert moved["sample_ids"] == batch["sample_ids"]
    assert moved["utilization"] == batch["utilization"]
