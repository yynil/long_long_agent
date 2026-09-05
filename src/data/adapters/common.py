"""Shared OpenAI-style message normalization."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from ..types import NormalizedMessage
from ..util import canonical_json, parse_json_maybe, split_think


def source_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return the original source row without adapter-only injected fields."""
    return {key: value for key, value in row.items() if not key.startswith("__")}


def source_row_digest(row: dict[str, Any]) -> str:
    payload = canonical_json(source_row(row)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(value, default=[])
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise TypeError(f"Expected tool call mapping, got {type(item).__name__}")
        call = copy.deepcopy(item)
        function = call.get("function")
        if not isinstance(function, dict):
            raise TypeError("Tool call must contain a function mapping")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ValueError("Tool call arguments are not valid JSON") from error
        if not isinstance(arguments, dict):
            raise TypeError("Tool call function.arguments must decode to an object")
        function["arguments"] = arguments
        normalized.append(call)
    return normalized


def normalize_openai_messages(value: Any) -> list[NormalizedMessage]:
    parsed = parse_json_maybe(value, default=[])
    if not isinstance(parsed, list):
        raise TypeError("Expected a list of messages")
    normalized: list[NormalizedMessage] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise TypeError(f"Expected message mapping, got {type(item).__name__}")
        source_role = str(item.get("role") or "unknown").lower()
        role = {"ai": "assistant", "function": "tool"}.get(source_role, source_role)
        content = item.get("content")
        if content is None:
            content = item.get("text") or ""
        explicit_reasoning = item.get("reasoning_content") or ""
        content_text = str(content)
        had_inline_think = content_text.lstrip().startswith("<think>")
        inline_reasoning, visible = split_think(content_text)
        reasoning = str(explicit_reasoning or inline_reasoning)
        tool_calls = normalize_tool_calls(item.get("tool_calls"))
        action: Any | None = tool_calls or None
        visible_content = visible if had_inline_think else content_text
        if role == "assistant" and tool_calls and visible_content.strip():
            reasoning = "\n\n".join(part for part in (reasoning, visible_content.strip()) if part)
            visible_content = ""
        normalized.append(
            NormalizedMessage(
                role=role,
                content=visible_content,
                reasoning=reasoning,
                action=action,
                tool_call_id=item.get("tool_call_id"),
                loss_mask=item.get("mask"),
                metadata={"source_role": source_role, "think": item.get("think")},
            )
        )
    return normalized
