"""Adapter for the Microsoft Orchard SWE subset."""

from __future__ import annotations

from typing import Any

from ..types import NormalizedEpisode
from ..util import first_task_message, normalize_success, parse_json_maybe
from .common import normalize_openai_messages, source_row, source_row_digest


def adapt_orchard_swe(
    row: dict[str, Any], source_revision: str, source_license: str
) -> NormalizedEpisode:
    metadata = parse_json_maybe(row.get("metadata"), default={})
    if not isinstance(metadata, dict):
        raise TypeError("Orchard metadata must decode to an object")
    messages = normalize_openai_messages(row.get("messages"))
    instance_id = str(metadata.get("instance_id") or "unknown")
    trace_digest = source_row_digest(row)
    success = normalize_success(metadata.get("verify_status"))
    return NormalizedEpisode(
        source_dataset="microsoft/Orchard:swe",
        source_revision=source_revision,
        source_license=source_license,
        source_record_id=(
            f"{instance_id}/{metadata.get('source', 'unknown')}/"
            f"{metadata.get('sample_idx', 0)}/{trace_digest}"
        ),
        task_id=instance_id,
        task_text=first_task_message(messages),
        messages=tuple(messages),
        harness="openhands"
        if str(metadata.get("source", "")).startswith("oh-")
        else "mini-swe-agent",
        tools=parse_json_maybe(row.get("tools"), default=[]),
        repo=metadata.get("repo"),
        teacher_model=metadata.get("model"),
        success=success,
        failure_type="unresolved" if success is False else None,
        metadata={
            "source_file": row.get("__source_file__"),
            "trace_digest": trace_digest,
            **metadata,
        },
        raw_record=source_row(row),
    )
