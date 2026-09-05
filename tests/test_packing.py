from __future__ import annotations

from dataclasses import replace

import pytest

from src.training.packing import IGNORE_INDEX, CausalSequence, pack_sequences


def test_targets_are_shifted_before_packing() -> None:
    first = CausalSequence.from_tokens("a", [10, 11, 12], [0, 1, 2])
    second = CausalSequence.from_tokens("b", [20, 21, 22, 23], [0, 3, 4, 5])
    batch = pack_sequences([first, second], align_to=4, pad_token_id=99)

    assert batch.input_ids == (10, 11, 20, 21, 22, 99, 99, 99)
    assert batch.targets == (11, 12, 21, 22, 23, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX)
    assert batch.loss_weights == (1, 2, 3, 4, 5, 0, 0, 0)
    assert batch.cu_seqlens == (0, 2, 5)
    assert batch.sequence_start_mask == (True, False, True, False, False, True, False, False)
    assert batch.valid_token_mask == (True, True, True, True, True, False, False, False)
    assert batch.segment_ids == (0, 0, 1, 1, 1, -1, -1, -1)


def test_packing_uses_only_global_tail_alignment() -> None:
    sequences = [
        CausalSequence.from_tokens("short", [1, 2, 3]),
        CausalSequence.from_tokens("long", list(range(100, 111))),
    ]
    batch = pack_sequences(sequences, align_to=4)
    stats = batch.utilization()

    assert batch.real_token_count == 12
    assert batch.aligned_token_count == 12
    assert stats["padded_baseline_tokens"] == 20
    assert stats["padding_tokens_avoided"] == 8
    assert stats["packed_storage_efficiency"] == 1.0


def test_boundary_metadata_fails_closed() -> None:
    batch = pack_sequences(
        [
            CausalSequence.from_tokens("a", [1, 2, 3]),
            CausalSequence.from_tokens("b", [4, 5, 6]),
        ],
        align_to=1,
    )
    broken = replace(
        batch,
        sequence_start_mask=(True, False, False, False),
    )
    with pytest.raises(ValueError, match="disagrees"):
        broken.validate()


def test_invalid_sequences_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        CausalSequence.from_tokens("too-short", [1])
    with pytest.raises(ValueError, match="unique"):
        item = CausalSequence.from_tokens("same", [1, 2])
        pack_sequences([item, item])
