"""Adapter for OpenThoughts-Agent-SFT trajectories."""

from __future__ import annotations

from typing import Any

from ..types import NormalizedEpisode, NormalizedMessage
from ..util import (
    first_json_object,
    first_task_message,
    normalize_success,
    split_think,
)
from .common import source_row, source_row_digest

TERMINUS_TOOL_SCHEMA = {
    "format": "terminus_json",
    "fields": ["analysis", "plan", "commands", "task_complete"],
    "command_fields": ["keystrokes", "duration"],
}


def adapt_openthoughts_agent(
    row: dict[str, Any], source_revision: str, source_license: str
) -> NormalizedEpisode:
    messages: list[NormalizedMessage] = []
    for item in row.get("conversations") or []:
        role = str(item.get("role") or "unknown").lower()
        content = str(item.get("content") or "")
        reasoning = ""
        action = None
        if role == "assistant":
            reasoning, visible = split_think(content)
            parsed_action = first_json_object(visible)
            if parsed_action is not None:
                action = {"format": "terminus_json", "payload": parsed_action}
                content = ""
            else:
                content = visible
        messages.append(
            NormalizedMessage(role=role, content=content, reasoning=reasoning, action=action)
        )

    source_prefix = "/".join(
        str(row.get(key) or "unknown") for key in ("run_id", "trial_name", "episode")
    )
    trace_digest = source_row_digest(row)
    source_record_id = f"{source_prefix}/{trace_digest}"
    return NormalizedEpisode(
        source_dataset="open-thoughts/OpenThoughts-Agent-SFT-100K",
        source_revision=source_revision,
        source_license=source_license,
        source_record_id=source_record_id,
        task_id=str(row.get("task") or source_prefix),
        task_text=first_task_message(messages),
        messages=tuple(messages),
        harness=str(row.get("agent") or "terminus-2"),
        tools=TERMINUS_TOOL_SCHEMA,
        teacher_model="GLM-4.7-AWQ",
        success=normalize_success(row.get("result")),
        failure_type=None
        if normalize_success(row.get("result")) is not False
        else str(row.get("result")),
        metadata={
            **{
                key: row.get(key)
                for key in ("date", "model_provider", "trace_source")
                if row.get(key) is not None
            },
            "source_file": row.get("__source_file__"),
            "source_model": row.get("model"),
            "trace_digest": trace_digest,
        },
        raw_record=source_row(row),
    )
