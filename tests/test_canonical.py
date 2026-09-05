from __future__ import annotations

import json

from src.data.blob_store import BlobStore
from src.data.canonical import canonicalize_episode, decision_types
from src.data.types import NormalizedEpisode, NormalizedMessage


def test_canonical_episode_and_decision_round_trip(tmp_path) -> None:
    episode = NormalizedEpisode(
        source_dataset="example/agent",
        source_revision="a" * 40,
        source_license="apache-2.0",
        source_record_id="trace-1",
        task_id="task-1",
        task_text="Inspect the repository",
        messages=(
            NormalizedMessage(role="user", content="Inspect the repository"),
            NormalizedMessage(
                role="assistant",
                reasoning="Need file list",
                action={"name": "bash", "arguments": {"command": "ls"}},
            ),
            NormalizedMessage(role="tool", content="README.md"),
        ),
        harness="test",
        tools=[{"name": "bash"}],
        raw_record={"messages": [{"role": "user", "content": "Inspect the repository"}]},
    )
    store = BlobStore(tmp_path / "blobs")
    episode_row, decisions = canonicalize_episode(episode, store)
    assert episode_row["turn_count"] == 1
    assert episode_row["split_group"] == "task:task-1"
    assert len(decisions) == 1
    assert decisions[0]["current_observation"] == "Inspect the repository"
    assert decisions[0]["next_tool_result"] == "README.md"
    blob_ref = episode_row["raw_trace_ref"]
    trace = store.read_json(blob_ref)
    assert trace["blob_schema_version"] == 1
    assert trace["source_record_id"] == "trace-1"
    assert trace["raw_record"]["messages"][0]["role"] == "user"
    assert trace["normalized"]["messages"][1]["reasoning"] == "Need file list"
    assert decisions[0]["prefix_messages_ref"].endswith("#normalized.messages=0:1")


def test_terminus_incomplete_action_is_not_termination() -> None:
    message = NormalizedMessage(
        role="assistant",
        action={
            "format": "terminus_json",
            "payload": {
                "commands": [{"keystrokes": "pytest -q"}],
                "task_complete": False,
            },
        },
    )
    assert decision_types(message) == ["verification"]


def test_multiple_tool_results_are_not_dropped(tmp_path) -> None:
    episode = NormalizedEpisode(
        source_dataset="example/agent",
        source_revision="a" * 40,
        source_license="apache-2.0",
        source_record_id="trace-multi-tool",
        task_id="task-multi-tool",
        task_text="Inspect two files",
        messages=(
            NormalizedMessage(role="user", content="Inspect two files"),
            NormalizedMessage(
                role="assistant",
                action=[{"name": "read", "id": "a"}, {"name": "read", "id": "b"}],
            ),
            NormalizedMessage(role="tool", content="first", tool_call_id="a"),
            NormalizedMessage(role="tool", content="second", tool_call_id="b"),
        ),
        harness="test",
    )
    _, decisions = canonicalize_episode(episode, BlobStore(tmp_path / "blobs"))
    results = json.loads(decisions[0]["next_tool_result"])
    assert [item["tool_call_id"] for item in results] == ["a", "b"]
    assert [item["content"] for item in results] == ["first", "second"]
