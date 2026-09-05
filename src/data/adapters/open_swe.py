"""Adapter for NVIDIA Open-SWE-Traces."""

from __future__ import annotations

from typing import Any

from ..types import NormalizedEpisode
from ..util import first_task_message, normalize_success, parse_json_maybe
from .common import normalize_openai_messages, source_row


def provenance(source_file: str) -> tuple[str, str]:
    parts = source_file.split("/")
    harness = parts[1] if len(parts) > 2 and parts[0] == "data" else "unknown"
    teacher_key = parts[2] if len(parts) > 3 and parts[0] == "data" else "unknown"
    teacher_model = {
        "minimax_m25": "MiniMax-M2.5",
        "qwen35_122b": "Qwen3.5-122B",
    }.get(teacher_key, teacher_key)
    return harness, teacher_model


def adapt_open_swe_traces(
    row: dict[str, Any], source_revision: str, source_license: str
) -> NormalizedEpisode:
    messages = normalize_openai_messages(row.get("messages"))
    tools = [parse_json_maybe(item) for item in (row.get("tools") or [])]
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    dataset_name = str(row.get("hf_dataset_name") or "unknown")
    source_file = str(row.get("__source_file__") or "unknown")
    harness, teacher_model = provenance(source_file)
    success = normalize_success(row.get("resolved"))
    return NormalizedEpisode(
        source_dataset="nvidia/Open-SWE-Traces",
        source_revision=source_revision,
        source_license=str(row.get("license") or source_license),
        source_record_id=str(row.get("trajectory_id") or row.get("instance_id")),
        task_id=str(row.get("instance_id")),
        task_text=first_task_message(messages),
        messages=tuple(messages),
        harness=harness,
        tools=tools,
        repo=row.get("repo"),
        teacher_model=teacher_model,
        success=success,
        failure_type="unresolved" if success is False else None,
        metadata={
            "language": row.get("language"),
            "dataset_name": dataset_name,
            "source_file": source_file,
            **metadata,
        },
        raw_record=source_row(row),
    )
