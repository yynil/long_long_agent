"""Deterministic token-budget pack rows for independent Agent decisions."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class PackRow:
    indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    real_tokens: int
    aligned_tokens: int


@dataclass(frozen=True)
class TokenBudgetPlan:
    epoch: int
    all_rows: tuple[PackRow, ...]
    kept_rows: tuple[PackRow, ...]
    dropped_rows: tuple[PackRow, ...]
    rank_rows: tuple[PackRow, ...]

    @property
    def dropped_sample_count(self) -> int:
        return sum(len(row.indices) for row in self.dropped_rows)


def _seed_for_epoch(seed: int, epoch: int) -> int:
    payload = f"long-long-agent-token-budget-v1:{seed}:{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class TokenBudgetPackSampler:
    """Yield one list of sample indices per packed CUDA row.

    Samples are independently reset decisions. Stateful episode-continuation
    windows intentionally use a different sampler and ownership contract.
    """

    def __init__(
        self,
        lengths: list[int] | tuple[int, ...],
        sample_ids: list[str] | tuple[str, ...],
        *,
        max_tokens: int,
        alignment: int = 16,
        seed: int = 20260904,
        bucket_size: int = 2048,
        shuffle: bool = True,
        world_size: int = 1,
        rank: int = 0,
        drop_distributed_tail: bool = True,
    ) -> None:
        self.lengths = tuple(int(length) for length in lengths)
        self.sample_ids = tuple(sample_ids)
        self.max_tokens = int(max_tokens)
        self.alignment = int(alignment)
        self.seed = int(seed)
        self.bucket_size = int(bucket_size)
        self.shuffle = bool(shuffle)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.drop_distributed_tail = bool(drop_distributed_tail)
        self.epoch = 0
        self._validate()

    def _validate(self) -> None:
        if not self.lengths or len(self.lengths) != len(self.sample_ids):
            raise ValueError("lengths and sample_ids must be non-empty and equal length")
        if len(set(self.sample_ids)) != len(self.sample_ids) or any(
            not sample_id for sample_id in self.sample_ids
        ):
            raise ValueError("sample_ids must be non-empty and globally unique")
        if self.alignment <= 0 or self.max_tokens <= 0 or self.max_tokens % self.alignment:
            raise ValueError("max_tokens must be a positive multiple of alignment")
        if any(length <= 0 or length > self.max_tokens for length in self.lengths):
            raise ValueError("every sequence length must be in (0, max_tokens]")
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.world_size > 1 and not self.drop_distributed_tail:
            raise ValueError("distributed sampling must drop tail rows to keep step counts equal")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _row(self, indices: list[int]) -> PackRow:
        real_tokens = sum(self.lengths[index] for index in indices)
        aligned_tokens = ((real_tokens + self.alignment - 1) // self.alignment) * self.alignment
        if aligned_tokens > self.max_tokens:
            raise AssertionError("sampler produced a row above the token budget")
        return PackRow(
            indices=tuple(indices),
            sample_ids=tuple(self.sample_ids[index] for index in indices),
            real_tokens=real_tokens,
            aligned_tokens=aligned_tokens,
        )

    def _all_rows(self, rng: random.Random) -> list[PackRow]:
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        rows: list[PackRow] = []
        for offset in range(0, len(order), self.bucket_size):
            bucket = order[offset : offset + self.bucket_size]
            bucket.sort(key=lambda index: (-self.lengths[index], self.sample_ids[index]))
            bins: list[tuple[int, list[int]]] = []
            for index in bucket:
                length = self.lengths[index]
                candidates = [
                    (self.max_tokens - used - length, bin_index)
                    for bin_index, (used, _) in enumerate(bins)
                    if used + length <= self.max_tokens
                ]
                if candidates:
                    _, selected = min(candidates)
                    used, members = bins[selected]
                    members.append(index)
                    bins[selected] = (used + length, members)
                else:
                    bins.append((length, [index]))
            for _, members in bins:
                if self.shuffle:
                    rng.shuffle(members)
                rows.append(self._row(members))
        if self.shuffle:
            rng.shuffle(rows)
        return rows

    def plan(self) -> TokenBudgetPlan:
        rng = random.Random(_seed_for_epoch(self.seed, self.epoch))
        all_rows = self._all_rows(rng)
        tail = len(all_rows) % self.world_size
        if tail and self.world_size > 1:
            kept_rows = all_rows[:-tail]
            dropped_rows = all_rows[-tail:]
        else:
            kept_rows = all_rows
            dropped_rows = []
        rank_rows = kept_rows[self.rank :: self.world_size]
        return TokenBudgetPlan(
            epoch=self.epoch,
            all_rows=tuple(all_rows),
            kept_rows=tuple(kept_rows),
            dropped_rows=tuple(dropped_rows),
            rank_rows=tuple(rank_rows),
        )

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return (row.indices for row in self.plan().rank_rows)

    def __len__(self) -> int:
        return len(self.plan().rank_rows)
