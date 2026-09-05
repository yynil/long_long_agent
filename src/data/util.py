"""Deterministic parsing and serialization helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from .types import NormalizedMessage

THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)
FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n?(.*?)```", re.DOTALL)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_json_maybe(value: Any, default: Any = None) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else value


def split_think(content: str | None) -> tuple[str, str]:
    text = content or ""
    match = THINK_RE.match(text)
    if not match:
        return "", text.strip()
    return match.group(1).strip(), match.group(2).strip()


def first_json_object(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def split_last_fenced_action(text: str | None) -> tuple[str, Any | None]:
    value = (text or "").strip()
    matches = list(FENCE_RE.finditer(value))
    if not matches:
        return value, None
    match = matches[-1]
    action_text = match.group(1).strip()
    reasoning = (value[: match.start()] + value[match.end() :]).strip()
    action = {
        "format": "fenced_tool_action",
        "content": action_text,
    }
    return reasoning, action


def normalize_success(value: Any) -> bool | None:
    if value is None or value == -1:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "success", "succeeded", "resolved", "passed", "pass"}:
        return True
    if normalized in {"0", "false", "failure", "failed", "unresolved", "error"}:
        return False
    return None


def first_task_message(messages: list[NormalizedMessage]) -> str:
    for message in messages:
        if message.role == "user" and message.content:
            return message.content
    for message in messages:
        if message.content:
            return message.content
    return ""
