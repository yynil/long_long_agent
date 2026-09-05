"""Serialize normalized Agent episodes and build reset-aware packed batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from src.data.canonical import stable_id
from src.data.types import NormalizedEpisode, NormalizedMessage
from src.data.util import canonical_json

from .packing import CausalSequence, PackedBatch, pack_sequences

EOD_TOKEN_ID = 0


class ByteTokenizer(Protocol):
    def encode_bytes(self, source: bytes) -> list[int]: ...

    def token_bytes(self, token_id: int) -> bytes: ...


@dataclass(frozen=True)
class EpisodeEncodingConfig:
    assistant_weight: float = 1.0
    reasoning_weight: float = 1.0
    action_weight: float = 2.0
    include_tool_schema: bool = True
    supervise_assistant_turn_end: bool = True
    supervise_terminal_eod: bool = True

    def __post_init__(self) -> None:
        weights = (self.assistant_weight, self.reasoning_weight, self.action_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True)
class AnnotatedText:
    text: str
    region: str
    loss_weight: float


@dataclass(frozen=True)
class TokenizedEpisode:
    sample_id: str
    rendered_text: str
    token_ids: tuple[int, ...]
    token_loss_weights: tuple[float, ...]
    token_regions: tuple[str, ...]
    mixed_boundary_token_count: int

    def __post_init__(self) -> None:
        size = len(self.token_ids)
        if size < 2:
            raise ValueError("tokenized episode must contain at least two tokens")
        if len(self.token_loss_weights) != size or len(self.token_regions) != size:
            raise ValueError("tokenized episode fields must have equal length")

    def to_causal_sequence(self) -> CausalSequence:
        return CausalSequence.from_tokens(
            self.sample_id,
            self.token_ids,
            self.token_loss_weights,
        )

    def loss_token_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for region, weight in zip(self.token_regions, self.token_loss_weights):
            if weight > 0:
                counts[region] = counts.get(region, 0) + 1
        return counts


@dataclass(frozen=True)
class TokenizedDecision:
    turn_id: int
    message_index: int
    dropped_message_count: int
    episode: TokenizedEpisode

    def to_causal_sequence(self) -> CausalSequence:
        return self.episode.to_causal_sequence()


def tokenize_annotated_text(
    tokenizer: ByteTokenizer, pieces: Sequence[AnnotatedText]
) -> tuple[list[int], list[float], list[str], int]:
    """Tokenize the complete text, conservatively masking cross-region tokens."""
    payload = b"".join(piece.text.encode("utf-8") for piece in pieces)
    byte_weights: list[float] = []
    byte_regions: list[str] = []
    for piece in pieces:
        encoded = piece.text.encode("utf-8")
        byte_weights.extend([piece.loss_weight] * len(encoded))
        byte_regions.extend([piece.region] * len(encoded))

    token_ids = tokenizer.encode_bytes(payload)
    token_weights: list[float] = []
    token_regions: list[str] = []
    mixed_count = 0
    offset = 0
    for token_id in token_ids:
        token = tokenizer.token_bytes(token_id)
        if not payload.startswith(token, offset):
            raise ValueError("tokenizer token bytes do not reconstruct the input")
        end = offset + len(token)
        weights = set(byte_weights[offset:end])
        regions = set(byte_regions[offset:end])
        if len(weights) == 1 and len(regions) == 1:
            token_weights.append(next(iter(weights)))
            token_regions.append(next(iter(regions)))
        else:
            token_weights.append(0.0)
            token_regions.append("mixed_boundary")
            mixed_count += 1
        offset = end
    if offset != len(payload):
        raise ValueError("tokenizer did not consume the complete input")
    return token_ids, token_weights, token_regions, mixed_count


def _tool_schema_piece(episode: NormalizedEpisode) -> AnnotatedText | None:
    if not episode.tools:
        return None
    text = (
        "System: Tools:\n"
        f"{canonical_json(episode.tools)}\n"
        "Use <tool_call>{...}</tool_call> for actions. Tool results use "
        "<tool_response>{...}</tool_response>.\n\n"
    )
    return AnnotatedText(text, "tool_schema", 0.0)


def _assistant_pieces(
    message: NormalizedMessage, config: EpisodeEncodingConfig
) -> list[AnnotatedText]:
    supervised = message.loss_mask is not False
    assistant_weight = config.assistant_weight if supervised else 0.0
    reasoning_weight = config.reasoning_weight if supervised else 0.0
    action_weight = config.action_weight if supervised else 0.0
    pieces = [AnnotatedText("Assistant: ", "assistant_prefix", 0.0)]
    has_output = False
    if message.reasoning:
        pieces.append(
            AnnotatedText(
                f"<think>\n{message.reasoning}\n</think>",
                "assistant_reasoning",
                reasoning_weight,
            )
        )
        has_output = True
    if message.content:
        if has_output:
            pieces.append(AnnotatedText("\n", "assistant_separator", assistant_weight))
        pieces.append(AnnotatedText(message.content, "assistant_final", assistant_weight))
        has_output = True
    if message.action is not None:
        if has_output:
            pieces.append(AnnotatedText("\n", "assistant_separator", action_weight))
        pieces.append(
            AnnotatedText(
                f"<tool_call>{canonical_json(message.action)}</tool_call>",
                "assistant_action",
                action_weight,
            )
        )
        has_output = True
    turn_end_weight = (
        assistant_weight if has_output and config.supervise_assistant_turn_end else 0.0
    )
    pieces.append(AnnotatedText("\n\n", "assistant_turn_end", turn_end_weight))
    return pieces


def render_episode(
    episode: NormalizedEpisode, config: EpisodeEncodingConfig
) -> list[AnnotatedText]:
    pieces: list[AnnotatedText] = []
    if config.include_tool_schema:
        tool_schema = _tool_schema_piece(episode)
        if tool_schema is not None:
            pieces.append(tool_schema)
    for message in episode.messages:
        if message.role in {"system", "developer"}:
            pieces.append(AnnotatedText(f"System: {message.content}\n\n", "system", 0.0))
        elif message.role == "user":
            pieces.append(AnnotatedText(f"User: {message.content}\n\n", "user", 0.0))
        elif message.role == "tool":
            payload = canonical_json(
                {"content": message.content, "tool_call_id": message.tool_call_id}
            )
            pieces.append(
                AnnotatedText(
                    f"User: <tool_response>{payload}</tool_response>\n\n",
                    "tool_response",
                    0.0,
                )
            )
        elif message.role == "assistant":
            pieces.extend(_assistant_pieces(message, config))
        else:
            raise ValueError(f"unsupported message role: {message.role!r}")
    if not pieces:
        raise ValueError("episode has no serializable messages")
    return pieces


def tokenize_episode(
    episode: NormalizedEpisode,
    tokenizer: ByteTokenizer,
    config: EpisodeEncodingConfig | None = None,
) -> TokenizedEpisode:
    config = config or EpisodeEncodingConfig()
    pieces = render_episode(episode, config)
    token_ids, token_weights, token_regions, mixed_count = tokenize_annotated_text(
        tokenizer, pieces
    )
    last_message = episode.messages[-1] if episode.messages else None
    terminal_weight = 0.0
    if (
        config.supervise_terminal_eod
        and last_message is not None
        and last_message.role == "assistant"
        and last_message.loss_mask is not False
        and (last_message.content or last_message.reasoning or last_message.action is not None)
    ):
        terminal_weight = config.assistant_weight
    return TokenizedEpisode(
        sample_id=stable_id(episode.source_dataset, episode.source_record_id),
        rendered_text="".join(piece.text for piece in pieces),
        token_ids=(EOD_TOKEN_ID, *token_ids, EOD_TOKEN_ID),
        token_loss_weights=(0.0, *token_weights, terminal_weight),
        token_regions=("document_start", *token_regions, "document_end"),
        mixed_boundary_token_count=mixed_count,
    )


def decision_message_indices(episode: NormalizedEpisode) -> tuple[int, ...]:
    return tuple(
        index
        for index, message in enumerate(episode.messages)
        if message.role == "assistant"
        and message.loss_mask is not False
        and (message.content or message.reasoning or message.action is not None)
    )


def _decision_candidate(
    episode: NormalizedEpisode,
    target_index: int,
    droppable_groups: tuple[tuple[int, ...], ...],
    drop_count: int,
) -> NormalizedEpisode:
    removed = {index for group in droppable_groups[:drop_count] for index in group}
    messages = []
    for index, message in enumerate(episode.messages[: target_index + 1]):
        if index not in removed:
            if message.role == "assistant" and index != target_index:
                message = replace(message, loss_mask=False)
            messages.append(message)
    if episode.task_text.strip() and not any(
        episode.task_text.strip() in message.content
        for message in messages
        if message.role in {"system", "developer", "user"}
    ):
        messages.insert(
            0, NormalizedMessage(role="user", content=f"Task contract:\n{episode.task_text}")
        )
    return replace(
        episode,
        source_record_id=f"{episode.source_record_id}/decision/{target_index}",
        messages=tuple(messages),
    )


def removable_history_groups(
    episode: NormalizedEpisode, target_index: int
) -> tuple[tuple[int, ...], ...]:
    """Keep task and latest observation; remove old assistant exchanges atomically."""
    prefix = episode.messages[:target_index]
    protected = {
        index for index, message in enumerate(prefix) if message.role in {"system", "developer"}
    }
    users = [index for index, message in enumerate(prefix) if message.role == "user"]
    if users:
        protected.update((users[0], users[-1]))
    assistants = [index for index, message in enumerate(prefix) if message.role == "assistant"]
    latest_start = assistants[-1] if assistants else 0
    protected.update(range(latest_start, target_index))
    groups: list[list[int]] = []
    for index, message in enumerate(prefix):
        if not groups or message.role == "assistant":
            groups.append([])
        groups[-1].append(index)
    return tuple(tuple(group) for group in groups if not protected.intersection(group))


def tokenize_decision(
    episode: NormalizedEpisode,
    message_index: int,
    tokenizer: ByteTokenizer,
    *,
    max_tokens: int,
    config: EpisodeEncodingConfig | None = None,
) -> TokenizedDecision:
    """Build a decision window with protected task and complete current observation."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    eligible = decision_message_indices(episode)
    if message_index not in eligible:
        raise ValueError("message_index is not a supervised assistant decision")
    turn_id = eligible.index(message_index)
    droppable = removable_history_groups(episode, message_index)
    encoding = config or EpisodeEncodingConfig()

    def encode(drop_count: int) -> TokenizedEpisode:
        candidate = _decision_candidate(episode, message_index, droppable, drop_count)
        return tokenize_episode(candidate, tokenizer, encoding)

    full = encode(0)
    if len(full.token_ids) - 1 <= max_tokens:
        return TokenizedDecision(turn_id, message_index, 0, full)
    minimal = encode(len(droppable))
    if len(minimal.token_ids) - 1 > max_tokens:
        raise ValueError(
            f"decision {turn_id} protected context needs {len(minimal.token_ids) - 1} tokens, "
            f"above max_tokens={max_tokens}"
        )

    lower = 0
    upper = len(droppable)
    best = minimal
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        candidate = encode(middle)
        if len(candidate.token_ids) - 1 <= max_tokens:
            upper = middle
            best = candidate
        else:
            lower = middle
    if upper != len(droppable):
        best = encode(upper)
    dropped_messages = sum(len(group) for group in droppable[:upper])
    return TokenizedDecision(turn_id, message_index, dropped_messages, best)


def tokenize_decisions(
    episode: NormalizedEpisode,
    tokenizer: ByteTokenizer,
    *,
    max_tokens: int,
    config: EpisodeEncodingConfig | None = None,
) -> tuple[TokenizedDecision, ...]:
    return tuple(
        tokenize_decision(
            episode,
            message_index,
            tokenizer,
            max_tokens=max_tokens,
            config=config,
        )
        for message_index in decision_message_indices(episode)
    )


class PackedEpisodeCollator:
    """Create one variable-length pack row; batching rows belongs to the sampler."""

    def __init__(
        self,
        tokenizer: ByteTokenizer,
        *,
        encoding: EpisodeEncodingConfig | None = None,
        align_to: int = 16,
        max_pack_tokens: int = 16384,
    ):
        if max_pack_tokens <= 0 or max_pack_tokens % align_to:
            raise ValueError("max_pack_tokens must be a positive multiple of align_to")
        self.tokenizer = tokenizer
        self.encoding = encoding or EpisodeEncodingConfig()
        self.align_to = align_to
        self.max_pack_tokens = max_pack_tokens

    def encode(self, episode: NormalizedEpisode) -> TokenizedEpisode:
        return tokenize_episode(episode, self.tokenizer, self.encoding)

    def pack_tokenized(
        self,
        items: Sequence[TokenizedEpisode | TokenizedDecision | CausalSequence],
    ) -> PackedBatch:
        sequences = []
        for item in items:
            if isinstance(item, TokenizedDecision | TokenizedEpisode):
                sequences.append(item.to_causal_sequence())
            elif isinstance(item, CausalSequence):
                sequences.append(item)
            else:
                raise TypeError(f"unsupported tokenized pack item: {type(item).__name__}")
        batch = pack_sequences(
            sequences,
            align_to=self.align_to,
            pad_token_id=EOD_TOKEN_ID,
        )
        if batch.aligned_token_count > self.max_pack_tokens:
            raise ValueError(
                f"packed row has {batch.aligned_token_count} tokens, "
                f"above max_pack_tokens={self.max_pack_tokens}"
            )
        return batch

    def pack(self, episodes: Sequence[NormalizedEpisode]) -> PackedBatch:
        encoded = [self.encode(episode) for episode in episodes]
        return self.pack_tokenized(encoded)

    @staticmethod
    def materialize(batch: PackedBatch) -> dict[str, Any]:
        """Materialize a validated packed row as the public PyTorch batch contract."""
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to materialize a training batch") from error
        batch.validate()
        return {
            "input_ids": torch.tensor(batch.input_ids, dtype=torch.long).unsqueeze(0),
            "targets": torch.tensor(batch.targets, dtype=torch.long).unsqueeze(0),
            "loss_weights": torch.tensor(batch.loss_weights, dtype=torch.float32).unsqueeze(0),
            "cu_seqlens": torch.tensor(batch.cu_seqlens, dtype=torch.int32),
            "sequence_start_mask": torch.tensor(
                batch.sequence_start_mask, dtype=torch.uint8
            ).unsqueeze(0),
            "valid_token_mask": torch.tensor(batch.valid_token_mask, dtype=torch.bool).unsqueeze(0),
            "segment_ids": torch.tensor(batch.segment_ids, dtype=torch.int32).unsqueeze(0),
            "sample_ids": batch.sample_ids,
            "utilization": batch.utilization(),
        }

    def collate_tokenized(
        self,
        items: Sequence[TokenizedEpisode | TokenizedDecision | CausalSequence],
    ) -> dict[str, Any]:
        return self.materialize(self.pack_tokenized(items))

    def __call__(self, episodes: Sequence[NormalizedEpisode]) -> dict[str, Any]:
        """Materialize the public batch contract as PyTorch tensors, lazily importing torch."""
        return self.materialize(self.pack(episodes))
