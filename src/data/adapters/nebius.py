"""Adapters for the two Nebius SWE trajectory formats."""

from __future__ import annotations

import hashlib
from typing import Any

from ..types import NormalizedEpisode, NormalizedMessage
from ..util import canonical_json, first_task_message, normalize_success, split_last_fenced_action
from .common import normalize_openai_messages, source_row


def adapt_nebius_swe_agent(
    row: dict[str, Any], source_revision: str, source_license: str
) -> NormalizedEpisode:
    messages: list[NormalizedMessage] = []
    for item in row.get("trajectory") or []:
        source_role = str(item.get("role") or "unknown")
        role = {"ai": "assistant"}.get(source_role, source_role)
        content = str(item.get("text") or item.get("system_prompt") or "")
        reasoning = ""
        action = None
        if role == "assistant":
            reasoning, action = split_last_fenced_action(content)
            content = ""
        messages.append(
            NormalizedMessage(
                role=role,
                content=content,
                reasoning=reasoning,
                action=action,
                loss_mask=item.get("mask"),
                metadata={"cutoff_date": item.get("cutoff_date")},
            )
        )
    success = normalize_success(row.get("target"))
    instance_id = str(row.get("instance_id") or "unknown")
    trace_digest = hashlib.sha256(
        canonical_json(
            {
                "trajectory": row.get("trajectory"),
                "target": row.get("target"),
                "exit_status": row.get("exit_status"),
                "generated_patch": row.get("generated_patch"),
            }
        ).encode("utf-8")
    ).hexdigest()
    return NormalizedEpisode(
        source_dataset="nebius/SWE-agent-trajectories",
        source_revision=source_revision,
        source_license=source_license,
        source_record_id=f"{instance_id}/{row.get('model_name', 'unknown')}/{trace_digest}",
        task_id=instance_id,
        task_text=first_task_message(messages),
        messages=tuple(messages),
        harness="swe-agent",
        teacher_model=str(row.get("model_name") or "unknown"),
        success=success,
        failure_type=str(row.get("exit_status")) if success is False else None,
        metadata={
            "source_file": row.get("__source_file__"),
            "trace_digest": trace_digest,
            "exit_status": row.get("exit_status"),
            "generated_patch": row.get("generated_patch"),
            "eval_logs": row.get("eval_logs"),
        },
        raw_record=source_row(row),
    )


def adapt_nebius_openhands(
    row: dict[str, Any], source_revision: str, source_license: str
) -> NormalizedEpisode:
    messages = normalize_openai_messages(row.get("trajectory"))
    success = normalize_success(row.get("resolved"))
    return NormalizedEpisode(
        source_dataset="nebius/SWE-rebench-openhands-trajectories",
        source_revision=source_revision,
        source_license=source_license,
        source_record_id=str(row.get("trajectory_id") or row.get("instance_id")),
        task_id=str(row.get("instance_id") or "unknown"),
        task_text=first_task_message(messages),
        messages=tuple(messages),
        harness="openhands",
        tools=row.get("tools") or [],
        repo=row.get("repo"),
        teacher_model="Qwen3-Coder-480B-A35B-Instruct",
        success=success,
        failure_type=str(row.get("exit_status")) if success is False else None,
        metadata={
            "source_file": row.get("__source_file__"),
            "agent_version": "OpenHands-0.54.0",
            "model_patch": row.get("model_patch"),
            "exit_status": row.get("exit_status"),
            "gen_tests_correct": row.get("gen_tests_correct"),
            "pred_passes_gen_tests": row.get("pred_passes_gen_tests"),
        },
        raw_record=source_row(row),
    )
