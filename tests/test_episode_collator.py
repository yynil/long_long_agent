from __future__ import annotations

from dataclasses import replace

import pytest

from src.data.types import NormalizedEpisode, NormalizedMessage
from src.training.episode_collator import (
    AnnotatedText,
    EpisodeEncodingConfig,
    PackedEpisodeCollator,
    decision_message_indices,
    tokenize_annotated_text,
    tokenize_decision,
    tokenize_decisions,
    tokenize_episode,
)


class ByteLevelTokenizer:
    def encode_bytes(self, source: bytes) -> list[int]:
        return [value + 1 for value in source]

    def token_bytes(self, token_id: int) -> bytes:
        return bytes([token_id - 1])


class CrossingTokenizer(ByteLevelTokenizer):
    def encode_bytes(self, source: bytes) -> list[int]:
        assert source == b"ab"
        return [300]

    def token_bytes(self, token_id: int) -> bytes:
        assert token_id == 300
        return b"ab"


def example_episode(source_record_id: str = "trace-1") -> NormalizedEpisode:
    return NormalizedEpisode(
        source_dataset="fixture/agent",
        source_revision="a" * 40,
        source_license="apache-2.0",
        source_record_id=source_record_id,
        task_id="task-1",
        task_text="Inspect",
        tools=[{"name": "bash"}],
        messages=(
            NormalizedMessage(role="user", content="Inspect the repository"),
            NormalizedMessage(
                role="assistant",
                reasoning="I should list files.",
                action={"name": "bash", "arguments": {"command": "ls"}},
            ),
            NormalizedMessage(role="tool", content="README.md", tool_call_id="call-1"),
            NormalizedMessage(role="assistant", content="Done."),
        ),
    )


def region_weights(tokenized, region: str) -> set[float]:
    return {
        weight
        for token_region, weight in zip(tokenized.token_regions, tokenized.token_loss_weights)
        if token_region == region
    }


def test_episode_mask_only_supervises_assistant_regions() -> None:
    tokenized = tokenize_episode(example_episode(), ByteLevelTokenizer())

    assert tokenized.token_ids[0] == 0
    assert tokenized.token_ids[-1] == 0
    assert region_weights(tokenized, "tool_schema") == {0.0}
    assert region_weights(tokenized, "user") == {0.0}
    assert region_weights(tokenized, "tool_response") == {0.0}
    assert region_weights(tokenized, "assistant_prefix") == {0.0}
    assert region_weights(tokenized, "assistant_reasoning") == {1.0}
    assert region_weights(tokenized, "assistant_action") == {2.0}
    assert region_weights(tokenized, "assistant_final") == {1.0}
    assert region_weights(tokenized, "document_end") == {1.0}
    assert "<tool_call>" in tokenized.rendered_text
    assert "<tool_response>" in tokenized.rendered_text
    assert tokenized.mixed_boundary_token_count == 0


def test_source_loss_mask_disables_assistant_supervision() -> None:
    episode = example_episode()
    messages = list(episode.messages)
    messages[1] = replace(messages[1], loss_mask=False)
    tokenized = tokenize_episode(replace(episode, messages=tuple(messages)), ByteLevelTokenizer())

    first_action_weights = [
        weight
        for region, weight in zip(tokenized.token_regions, tokenized.token_loss_weights)
        if region in {"assistant_reasoning", "assistant_action"}
    ]
    assert first_action_weights
    assert set(first_action_weights) == {0.0}
    assert region_weights(tokenized, "assistant_final") == {1.0}


def test_cross_region_token_is_conservatively_masked() -> None:
    result = tokenize_annotated_text(
        CrossingTokenizer(),
        [AnnotatedText("a", "user", 0.0), AnnotatedText("b", "assistant_final", 1.0)],
    )
    token_ids, weights, regions, mixed_count = result

    assert token_ids == [300]
    assert weights == [0.0]
    assert regions == ["mixed_boundary"]
    assert mixed_count == 1


def test_collator_packs_episode_targets_before_concatenation() -> None:
    collator = PackedEpisodeCollator(
        ByteLevelTokenizer(),
        encoding=EpisodeEncodingConfig(include_tool_schema=False),
        align_to=16,
        max_pack_tokens=4096,
    )
    first = example_episode("trace-a")
    second = replace(example_episode("trace-b"), tools=[])
    batch = collator.pack([first, second])

    assert len(batch.sample_ids) == 2
    assert batch.sequence_start_mask[0]
    assert batch.sequence_start_mask[batch.cu_seqlens[1]]
    assert batch.targets[batch.cu_seqlens[1] - 1] == 0
    assert batch.input_ids[batch.cu_seqlens[1]] == 0
    assert batch.alignment_token_count <= 15
    assert all(
        weight == 0 for target, weight in zip(batch.targets, batch.loss_weights) if target < 0
    )


def test_collator_fails_closed_on_overflow_and_unknown_role() -> None:
    collator = PackedEpisodeCollator(ByteLevelTokenizer(), align_to=16, max_pack_tokens=16)
    with pytest.raises(ValueError, match="above max_pack_tokens"):
        collator.pack([example_episode()])

    bad = replace(example_episode(), messages=(NormalizedMessage(role="unknown", content="x"),))
    with pytest.raises(ValueError, match="unsupported message role"):
        collator.encode(bad)


def test_decision_windows_only_supervise_current_assistant() -> None:
    episode = example_episode()
    windows = tokenize_decisions(
        episode,
        ByteLevelTokenizer(),
        max_tokens=4096,
        config=EpisodeEncodingConfig(include_tool_schema=False),
    )

    assert decision_message_indices(episode) == (1, 3)
    assert len(windows) == 2
    assert windows[0].episode.sample_id != windows[1].episode.sample_id
    assert windows[1].episode.loss_token_counts().get("assistant_action", 0) == 0
    assert windows[1].episode.loss_token_counts()["assistant_final"] > 0


def test_sampler_rows_can_pack_pre_tokenized_decisions_without_reencoding_episode() -> None:
    collator = PackedEpisodeCollator(
        ByteLevelTokenizer(),
        encoding=EpisodeEncodingConfig(include_tool_schema=False),
        align_to=16,
        max_pack_tokens=4096,
    )
    decisions = tokenize_decisions(
        example_episode(),
        collator.tokenizer,
        max_tokens=4096,
        config=collator.encoding,
    )
    batch = collator.pack_tokenized(decisions)

    assert batch.sample_ids == tuple(item.episode.sample_id for item in decisions)
    assert batch.real_token_count == sum(
        len(item.to_causal_sequence().input_ids) for item in decisions
    )
    assert sum(weight > 0 for weight in batch.loss_weights) == sum(
        sum(weight > 0 for weight in item.to_causal_sequence().loss_weights) for item in decisions
    )


def test_pack_tokenized_rejects_unknown_item_type() -> None:
    collator = PackedEpisodeCollator(ByteLevelTokenizer())
    with pytest.raises(TypeError, match="unsupported tokenized pack item"):
        collator.pack_tokenized([object()])


def test_decision_window_drops_old_messages_at_boundaries() -> None:
    episode = example_episode()
    target_index = decision_message_indices(episode)[-1]
    config = EpisodeEncodingConfig(include_tool_schema=False)
    minimal_episode = replace(
        episode,
        source_record_id="minimal",
        messages=(episode.messages[target_index],),
    )
    minimum = tokenize_episode(minimal_episode, ByteLevelTokenizer(), config)
    max_tokens = len(minimum.token_ids) + 8
    window = tokenize_decision(
        episode,
        target_index,
        ByteLevelTokenizer(),
        max_tokens=max_tokens,
        config=config,
    )

    assert window.dropped_message_count > 0
    assert len(window.episode.token_ids) - 1 <= max_tokens
    assert "Done." in window.episode.rendered_text


def test_decision_window_rejects_untrainable_assistant() -> None:
    episode = example_episode()
    messages = list(episode.messages)
    messages[1] = replace(messages[1], loss_mask=False)
    masked = replace(episode, messages=tuple(messages))

    with pytest.raises(ValueError, match="not a supervised"):
        tokenize_decision(masked, 1, ByteLevelTokenizer(), max_tokens=4096)
