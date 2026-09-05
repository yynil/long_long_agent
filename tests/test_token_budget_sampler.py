from __future__ import annotations

import pytest

from src.training.packing import CausalSequence, pack_sequences
from src.training.token_budget_sampler import TokenBudgetPackSampler


def make_sequences(lengths: list[int]) -> list[CausalSequence]:
    return [
        CausalSequence(
            sample_id=f"sample-{index}",
            input_ids=tuple(range(length)),
            targets=tuple(range(length)),
            loss_weights=(1.0,) * length,
        )
        for index, length in enumerate(lengths)
    ]


def test_sampler_covers_every_sample_once_and_integrates_with_packer() -> None:
    lengths = [15, 9, 8, 7, 6, 5, 4, 3]
    sequences = make_sequences(lengths)
    sampler = TokenBudgetPackSampler(
        lengths,
        [sequence.sample_id for sequence in sequences],
        max_tokens=32,
        alignment=8,
        seed=7,
        bucket_size=4,
    )
    rows = list(sampler)
    flattened = [index for row in rows for index in row]
    assert sorted(flattened) == list(range(len(lengths)))
    assert len(flattened) == len(set(flattened))
    for row in rows:
        packed = pack_sequences((sequences[index] for index in row), align_to=8)
        assert packed.aligned_token_count <= 32
        assert packed.alignment_token_count <= 7
        assert packed.sample_ids == tuple(sequences[index].sample_id for index in row)


def test_sampler_is_deterministic_per_epoch_and_changes_across_epochs() -> None:
    lengths = [(index * 7) % 29 + 1 for index in range(80)]
    sample_ids = [f"sample-{index}" for index in range(len(lengths))]
    first = TokenBudgetPackSampler(
        lengths,
        sample_ids,
        max_tokens=64,
        seed=19,
        bucket_size=16,
    )
    second = TokenBudgetPackSampler(
        lengths,
        sample_ids,
        max_tokens=64,
        seed=19,
        bucket_size=16,
    )
    assert list(first) == list(second)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    assert list(first) != list(TokenBudgetPackSampler(lengths, sample_ids, max_tokens=64, seed=19))


def test_distributed_ranks_are_disjoint_and_have_equal_step_counts() -> None:
    lengths = [17] * 17
    sample_ids = [f"sample-{index}" for index in range(len(lengths))]
    samplers = [
        TokenBudgetPackSampler(
            lengths,
            sample_ids,
            max_tokens=32,
            seed=23,
            world_size=3,
            rank=rank,
        )
        for rank in range(3)
    ]
    plans = [sampler.plan() for sampler in samplers]
    assert len({len(plan.rank_rows) for plan in plans}) == 1
    rank_indices = [{index for row in plan.rank_rows for index in row.indices} for plan in plans]
    assert all(
        rank_indices[left].isdisjoint(rank_indices[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    kept = set().union(*rank_indices)
    dropped = {index for row in plans[0].dropped_rows for index in row.indices}
    assert kept.isdisjoint(dropped)
    assert kept | dropped == set(range(len(lengths)))
    assert plans[0].dropped_sample_count == 2


@pytest.mark.parametrize(
    ("lengths", "sample_ids", "kwargs"),
    [
        ([0], ["a"], {}),
        ([33], ["a"], {}),
        ([1, 1], ["a", "a"], {}),
        ([1], ["a"], {"world_size": 2, "rank": 2}),
        ([1], ["a"], {"world_size": 2, "rank": 0, "drop_distributed_tail": False}),
    ],
)
def test_sampler_rejects_invalid_contract(lengths, sample_ids, kwargs) -> None:
    with pytest.raises(ValueError):
        TokenBudgetPackSampler(
            lengths,
            sample_ids,
            max_tokens=32,
            **kwargs,
        )
