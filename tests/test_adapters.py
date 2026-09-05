from __future__ import annotations

import json

from src.data.adapters.common import source_row_digest
from src.data.adapters.nebius import adapt_nebius_openhands, adapt_nebius_swe_agent
from src.data.adapters.open_swe import adapt_open_swe_traces
from src.data.adapters.openthoughts import adapt_openthoughts_agent
from src.data.adapters.orchard import adapt_orchard_swe

REVISION = "a" * 40


def test_openthoughts_splits_think_and_terminus_json() -> None:
    episode = adapt_openthoughts_agent(
        {
            "run_id": "run",
            "trial_name": "trial",
            "episode": "episode-1",
            "task": "task-1",
            "agent": "terminus-2",
            "model": "teacher",
            "result": "success",
            "conversations": [
                {"role": "user", "content": "Solve the task"},
                {
                    "role": "assistant",
                    "content": '<think>inspect first</think>{"commands":[],"task_complete":false}',
                },
                {"role": "user", "content": "terminal output"},
            ],
        },
        REVISION,
        "apache-2.0",
    )
    assert episode.success is True
    assert episode.messages[1].reasoning == "inspect first"
    assert episode.messages[1].action["format"] == "terminus_json"
    assert episode.messages[1].action["payload"]["task_complete"] is False
    assert episode.teacher_model == "GLM-4.7-AWQ"
    assert len(episode.source_record_id.rsplit("/", 1)[-1]) == 64


def test_source_row_digest_ignores_adapter_fields_but_tracks_content() -> None:
    source = {
        "run_id": "run",
        "conversations": [{"role": "assistant", "content": "first"}],
    }
    with_path = {**source, "__source_file__": "data/a.parquet"}
    changed = {
        **source,
        "conversations": [{"role": "assistant", "content": "second"}],
    }
    assert source_row_digest(source) == source_row_digest(with_path)
    assert source_row_digest(source) != source_row_digest(changed)


def test_open_swe_preserves_reasoning_and_structured_tool_call() -> None:
    episode = adapt_open_swe_traces(
        {
            "instance_id": "org__repo-1",
            "trajectory_id": "trace-1",
            "repo": "org/repo",
            "license": "mit",
            "language": "Python",
            "hf_dataset_name": "openhands/minimax_m25",
            "__source_file__": (
                "data/openhands/minimax_m25/swe-rebench-v2/train-00000-of-00018.parquet"
            ),
            "resolved": 1,
            "tools": ['{"type":"function","function":{"name":"bash"}}'],
            "metadata": {"category": "bug-fix"},
            "messages": [
                {"role": "user", "content": "Fix issue"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "inspect files",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
            ],
        },
        REVISION,
        "cc-by-4.0",
    )
    assert episode.harness == "openhands"
    assert episode.teacher_model == "MiniMax-M2.5"
    assert episode.success is True
    assert episode.messages[1].reasoning == "inspect files"
    assert episode.messages[1].action[0]["function"]["name"] == "bash"
    assert episode.messages[1].action[0]["function"]["arguments"] == {"command": "ls"}


def test_invalid_tool_arguments_are_rejected() -> None:
    try:
        adapt_nebius_openhands(
            {
                "trajectory_id": "bad-trace",
                "instance_id": "bad-task",
                "trajectory": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "bad-call",
                                "type": "function",
                                "function": {"name": "bash", "arguments": "{bad json"},
                            }
                        ],
                    }
                ],
            },
            REVISION,
            "cc-by-4.0",
        )
    except ValueError as error:
        assert "not valid JSON" in str(error)
    else:
        raise AssertionError("Invalid tool arguments must fail closed")


def test_orchard_decodes_json_fields() -> None:
    episode = adapt_orchard_swe(
        {
            "tools": '[{"name":"bash"}]',
            "metadata": json.dumps(
                {
                    "instance_id": "org__repo-2",
                    "sample_idx": 3,
                    "source": "rebench-M2.5",
                    "model": "MiniMax-M2.5",
                    "repo": "org__repo",
                    "verify_status": "unresolved",
                }
            ),
            "messages": [{"role": "user", "content": "Fix issue", "tool_calls": []}],
        },
        REVISION,
        "mit",
    )
    assert episode.source_record_id.startswith("org__repo-2/rebench-M2.5/3/")
    assert episode.success is False
    assert episode.failure_type == "unresolved"


def test_nebius_swe_agent_extracts_last_fenced_action() -> None:
    episode = adapt_nebius_swe_agent(
        {
            "instance_id": "repo-3",
            "model_name": "teacher",
            "target": False,
            "exit_status": "failed",
            "trajectory": [
                {"role": "system", "system_prompt": "System", "text": "", "mask": False},
                {"role": "user", "text": "Fix issue", "mask": False},
                {"role": "ai", "text": "Inspect.\n```\nls -la\n```", "mask": True},
            ],
        },
        REVISION,
        "cc-by-4.0",
    )
    assert episode.messages[2].reasoning == "Inspect."
    assert episode.messages[2].action["content"] == "ls -la"
    assert episode.success is False
    assert len(episode.source_record_id.rsplit("/", 1)[-1]) == 64


def test_nebius_swe_agent_rollouts_get_distinct_source_ids() -> None:
    base = {
        "instance_id": "repo-3",
        "model_name": "teacher",
        "target": False,
        "exit_status": "failed",
        "trajectory": [{"role": "ai", "text": "Inspect.\n```\nls\n```", "mask": True}],
    }
    first = adapt_nebius_swe_agent(base, REVISION, "cc-by-4.0")
    second = adapt_nebius_swe_agent(
        {
            **base,
            "trajectory": [{"role": "ai", "text": "Inspect.\n```\npwd\n```", "mask": True}],
        },
        REVISION,
        "cc-by-4.0",
    )
    assert first.source_record_id != second.source_record_id


def test_nebius_openhands_maps_trajectory() -> None:
    episode = adapt_nebius_openhands(
        {
            "trajectory_id": "trace-4",
            "instance_id": "repo-4",
            "repo": "org/repo",
            "resolved": 0,
            "exit_status": "failed",
            "trajectory": [
                {"role": "user", "content": "Fix issue"},
                {"role": "assistant", "content": "final answer", "tool_calls": []},
            ],
            "tools": [],
        },
        REVISION,
        "cc-by-4.0",
    )
    assert episode.harness == "openhands"
    assert episode.messages[1].content == "final answer"
    assert episode.success is False


def test_openai_tool_call_content_becomes_reasoning() -> None:
    episode = adapt_nebius_openhands(
        {
            "trajectory_id": "trace-5",
            "instance_id": "repo-5",
            "trajectory": [
                {"role": "user", "content": "Inspect"},
                {
                    "role": "assistant",
                    "content": "I should inspect the repository first.",
                    "tool_calls": [
                        {
                            "id": "call-5",
                            "type": "function",
                            "function": {"name": "execute_bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
            ],
            "tools": [],
        },
        REVISION,
        "cc-by-4.0",
    )
    assistant = episode.messages[1]
    assert assistant.content == ""
    assert assistant.reasoning == "I should inspect the repository first."
    assert assistant.action[0]["function"]["name"] == "execute_bash"
    assert episode.teacher_model == "Qwen3-Coder-480B-A35B-Instruct"
