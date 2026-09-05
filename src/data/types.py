"""Source-neutral Agent trajectory types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedMessage:
    role: str
    content: str = ""
    reasoning: str = ""
    action: Any | None = None
    tool_call_id: str | None = None
    loss_mask: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedEpisode:
    source_dataset: str
    source_revision: str
    source_license: str
    source_record_id: str
    task_id: str
    task_text: str
    messages: tuple[NormalizedMessage, ...]
    harness: str = "unknown"
    tools: Any = field(default_factory=list)
    repo: str | None = None
    base_commit: str | None = None
    teacher_model: str | None = None
    success: bool | None = None
    terminal_reward: float | None = None
    failure_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_record: dict[str, Any] = field(default_factory=dict)
