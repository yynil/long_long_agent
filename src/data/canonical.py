"""Convert source-neutral episodes into canonical episode and decision rows."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from .blob_store import BlobStore
from .governance import canonical_group_identity
from .types import NormalizedEpisode, NormalizedMessage
from .util import canonical_json


def stable_id(namespace: str, *parts: str) -> str:
    payload = "\x1f".join((namespace, *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialize_message(message: NormalizedMessage) -> dict[str, Any]:
    return asdict(message)


def serialize_observations(messages: list[NormalizedMessage]) -> str:
    if not messages:
        return ""
    if len(messages) == 1:
        return messages[0].content
    return canonical_json(
        [
            {
                "role": message.role,
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }
            for message in messages
        ]
    )


def previous_observation(messages: tuple[NormalizedMessage, ...], index: int) -> str:
    observations: list[NormalizedMessage] = []
    for message in reversed(messages[:index]):
        if message.role == "assistant":
            break
        if message.role in {"user", "tool"} and message.content:
            observations.append(message)
    observations.reverse()
    return serialize_observations(observations)


def next_observation(messages: tuple[NormalizedMessage, ...], index: int) -> str:
    observations: list[NormalizedMessage] = []
    for message in messages[index + 1 :]:
        if message.role == "assistant":
            break
        if message.role in {"user", "tool"} and message.content:
            observations.append(message)
    return serialize_observations(observations)


def decision_types(message: NormalizedMessage) -> list[str]:
    action = message.action
    is_termination = False
    if isinstance(action, dict) and action.get("format") == "terminus_json":
        terminus_payload = action.get("payload") or {}
        commands = terminus_payload.get("commands") or []
        payload = canonical_json(commands).lower()
        is_termination = terminus_payload.get("task_complete") is True
    elif isinstance(action, dict) and action.get("format") == "assistant_final":
        payload = str(action.get("content") or "").lower()
        is_termination = True
    else:
        payload = canonical_json(action).lower()
    labels: list[str] = []
    if any(token in payload for token in ("grep", "find", "search", "open", "read", "cat")):
        labels.append("information_gathering")
    if any(token in payload for token in ("edit", "write", "patch", "sed", "create")):
        labels.append("editing")
    if any(token in payload for token in ("test", "pytest", "verify", "lint")):
        labels.append("verification")
    if is_termination or any(token in payload for token in ("finish", "submit")):
        labels.append("termination")
    return labels or ["mechanical_action"]


def canonicalize_episode(
    episode: NormalizedEpisode, blob_store: BlobStore
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_id = stable_id(episode.source_dataset, episode.source_record_id)
    trace = {
        "blob_schema_version": 1,
        "source_record_id": episode.source_record_id,
        "source_dataset": episode.source_dataset,
        "source_revision": episode.source_revision,
        "source_file": episode.metadata.get("source_file"),
        "raw_record": episode.raw_record,
        "normalized": {
            "messages": [serialize_message(message) for message in episode.messages],
            "metadata": episode.metadata,
        },
    }
    raw_trace_ref = blob_store.put_json(trace)
    tools_schema = canonical_json(episode.tools)
    episode_row = {
        "episode_id": episode_id,
        "task_id": episode.task_id,
        "source_record_id": episode.source_record_id,
        "split_group": canonical_group_identity(episode.repo, episode.task_id),
        "source_dataset": episode.source_dataset,
        "source_revision": episode.source_revision,
        "source_license": episode.source_license,
        "harness": episode.harness,
        "tools_schema": tools_schema,
        "task_text": episode.task_text,
        "acceptance_criteria": None,
        "repo": episode.repo,
        "base_commit": episode.base_commit,
        "env_image_digest": None,
        "verifier_id": None,
        "verifier_version": None,
        "teacher_model": episode.teacher_model,
        "teacher_prompt_version": None,
        "success": episode.success,
        "terminal_reward": episode.terminal_reward,
        "turn_count": sum(message.role == "assistant" for message in episode.messages),
        "failure_type": episode.failure_type,
        "raw_trace_ref": raw_trace_ref,
    }

    decisions = []
    turn_id = 0
    for index, message in enumerate(episode.messages):
        if message.role != "assistant":
            continue
        if message.action is None and not message.content:
            continue
        decision_id = stable_id("decision", episode_id, str(turn_id))
        action = message.action
        if action is None:
            action = {"format": "assistant_final", "content": message.content}
        decisions.append(
            {
                "episode_id": episode_id,
                "turn_id": turn_id,
                "decision_id": decision_id,
                "prefix_messages_ref": f"{raw_trace_ref}#normalized.messages=0:{index}",
                "current_observation": previous_observation(episode.messages, index),
                "tools_schema": tools_schema,
                "teacher_think_raw": message.reasoning,
                "teacher_think_tokens": None,
                "macro_thoughts": None,
                "teacher_action": canonical_json(action),
                "next_tool_result": next_observation(episode.messages, index),
                "env_delta": None,
                "verifier_delta": None,
                "progress_label": None,
                "regression_label": None,
                "recovery_label": None,
                "decision_type": decision_types(message),
            }
        )
        turn_id += 1
    return episode_row, decisions
