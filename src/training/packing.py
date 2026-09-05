"""Framework-neutral packed-varlen batch contract.

The CUDA path consumes ``sequence_start_mask`` while ``cu_seqlens`` remains the
canonical segment index. Targets are shifted inside each source sequence before
packing, so a target can never point at the first token of the next sequence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

IGNORE_INDEX = -100


@dataclass(frozen=True)
class CausalSequence:
    """One independently recurrent causal sequence, already shifted."""

    sample_id: str
    input_ids: tuple[int, ...]
    targets: tuple[int, ...]
    loss_weights: tuple[float, ...]

    def __post_init__(self) -> None:
        length = len(self.input_ids)
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if length == 0:
            raise ValueError("a packed sequence must contain at least one input token")
        if len(self.targets) != length or len(self.loss_weights) != length:
            raise ValueError("input_ids, targets, and loss_weights must have equal length")
        if any(weight < 0 for weight in self.loss_weights):
            raise ValueError("loss_weights must be non-negative")

    @classmethod
    def from_tokens(
        cls,
        sample_id: str,
        token_ids: Sequence[int],
        token_loss_weights: Sequence[float] | None = None,
    ) -> CausalSequence:
        """Build input/target pairs before concatenation.

        ``token_loss_weights[i]`` controls loss when token ``i`` is the target.
        This lets callers apply role/action masking without ever constructing a
        target across a packed sequence boundary.
        """

        tokens = tuple(int(token) for token in token_ids)
        if len(tokens) < 2:
            raise ValueError("a causal sequence needs at least two tokens")
        if token_loss_weights is None:
            weights = (1.0,) * len(tokens)
        else:
            weights = tuple(float(weight) for weight in token_loss_weights)
            if len(weights) != len(tokens):
                raise ValueError("token_loss_weights must match token_ids")
        return cls(
            sample_id=sample_id,
            input_ids=tokens[:-1],
            targets=tokens[1:],
            loss_weights=weights[1:],
        )


@dataclass(frozen=True)
class PackedBatch:
    """A flat packed stream plus redundant, validated boundary metadata."""

    input_ids: tuple[int, ...]
    targets: tuple[int, ...]
    loss_weights: tuple[float, ...]
    cu_seqlens: tuple[int, ...]
    sequence_start_mask: tuple[bool, ...]
    valid_token_mask: tuple[bool, ...]
    segment_ids: tuple[int, ...]
    sample_ids: tuple[str, ...]
    alignment: int

    @property
    def real_token_count(self) -> int:
        return self.cu_seqlens[-1]

    @property
    def aligned_token_count(self) -> int:
        return len(self.input_ids)

    @property
    def alignment_token_count(self) -> int:
        return self.aligned_token_count - self.real_token_count

    def segment_slice(self, index: int) -> slice:
        if not 0 <= index < len(self.sample_ids):
            raise IndexError(index)
        return slice(self.cu_seqlens[index], self.cu_seqlens[index + 1])

    def utilization(self) -> dict[str, float | int]:
        lengths = [
            self.cu_seqlens[index + 1] - self.cu_seqlens[index]
            for index in range(len(self.sample_ids))
        ]
        padded_baseline = len(lengths) * max(lengths)
        return {
            "sequence_count": len(lengths),
            "real_tokens": self.real_token_count,
            "aligned_tokens": self.aligned_token_count,
            "alignment_tokens": self.alignment_token_count,
            "effective_loss_tokens": sum(weight > 0 for weight in self.loss_weights),
            "packed_storage_efficiency": self.real_token_count / self.aligned_token_count,
            "padded_baseline_tokens": padded_baseline,
            "padded_baseline_efficiency": self.real_token_count / padded_baseline,
            "padding_tokens_avoided": padded_baseline - self.aligned_token_count,
        }

    def validate(self) -> None:
        size = len(self.input_ids)
        parallel = (
            self.targets,
            self.loss_weights,
            self.sequence_start_mask,
            self.valid_token_mask,
            self.segment_ids,
        )
        if any(len(values) != size for values in parallel):
            raise ValueError("all packed token fields must have equal length")
        if self.alignment <= 0 or size % self.alignment != 0:
            raise ValueError("packed storage length must satisfy alignment")
        if len(self.cu_seqlens) != len(self.sample_ids) + 1:
            raise ValueError("cu_seqlens must contain one offset per sample plus zero")
        if not self.cu_seqlens or self.cu_seqlens[0] != 0:
            raise ValueError("cu_seqlens must start at zero")
        if any(right <= left for left, right in zip(self.cu_seqlens, self.cu_seqlens[1:])):
            raise ValueError("cu_seqlens must be strictly increasing")
        if self.real_token_count > size:
            raise ValueError("cu_seqlens exceeds packed storage")

        expected_starts = set(self.cu_seqlens[:-1])
        if self.alignment_token_count:
            expected_starts.add(self.real_token_count)
        actual_starts = {index for index, value in enumerate(self.sequence_start_mask) if value}
        if actual_starts != expected_starts:
            raise ValueError("sequence_start_mask disagrees with cu_seqlens")
        for index in range(size):
            is_real = index < self.real_token_count
            if self.valid_token_mask[index] != is_real:
                raise ValueError("valid_token_mask is not a contiguous valid prefix")
            if not is_real:
                if self.segment_ids[index] != -1:
                    raise ValueError("alignment tokens must use segment_id=-1")
                if self.targets[index] != IGNORE_INDEX or self.loss_weights[index] != 0:
                    raise ValueError("alignment tokens must not contribute target loss")


def pack_sequences(
    sequences: Iterable[CausalSequence],
    *,
    align_to: int = 16,
    pad_token_id: int = 0,
) -> PackedBatch:
    """Concatenate independent sequences and add only global tail alignment."""

    items = tuple(sequences)
    if not items:
        raise ValueError("cannot pack an empty sequence collection")
    if align_to <= 0:
        raise ValueError("align_to must be positive")
    if len({item.sample_id for item in items}) != len(items):
        raise ValueError("sample_id values must be unique within a packed batch")

    input_ids: list[int] = []
    targets: list[int] = []
    loss_weights: list[float] = []
    starts: list[bool] = []
    valid: list[bool] = []
    segment_ids: list[int] = []
    cu_seqlens = [0]

    for segment_id, item in enumerate(items):
        length = len(item.input_ids)
        input_ids.extend(item.input_ids)
        targets.extend(item.targets)
        loss_weights.extend(item.loss_weights)
        starts.extend([True, *([False] * (length - 1))])
        valid.extend([True] * length)
        segment_ids.extend([segment_id] * length)
        cu_seqlens.append(len(input_ids))

    tail = (-len(input_ids)) % align_to
    if tail:
        input_ids.extend([pad_token_id] * tail)
        targets.extend([IGNORE_INDEX] * tail)
        loss_weights.extend([0.0] * tail)
        starts.extend([True, *([False] * (tail - 1))])
        valid.extend([False] * tail)
        segment_ids.extend([-1] * tail)

    batch = PackedBatch(
        input_ids=tuple(input_ids),
        targets=tuple(targets),
        loss_weights=tuple(loss_weights),
        cu_seqlens=tuple(cu_seqlens),
        sequence_start_mask=tuple(starts),
        valid_token_mask=tuple(valid),
        segment_ids=tuple(segment_ids),
        sample_ids=tuple(item.sample_id for item in items),
        alignment=align_to,
    )
    batch.validate()
    return batch
