"""Content-only fingerprints and fail-closed Agent release checks.

Findings contain rule identifiers, never matched private text. Repository license
evidence is taken from a pinned publisher's explicit per-repository SPDX column.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict
from typing import Any

import jsonschema

from .types import NormalizedEpisode
from .util import canonical_json

QUALITY_VERSION = "agent_quality_v1"
ALLOWED_REPO_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})
SENSITIVE_RULES = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "credential_url": re.compile(r"https?://[^\s/:@]+:[^\s/@]{4,}@"),
    "credential_assignment": re.compile(
        r"(?i)\b(?:password|api[_-]?key|access[_-]?token|client[_-]?secret)\b"
        r"\s*[=:]\s*[\"']([A-Za-z0-9_+/=-]{16,})[\"']"
    ),
    "email": re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def sensitive_findings(text: str) -> tuple[str, ...]:
    findings = []
    for name, pattern in SENSITIVE_RULES.items():
        matches = pattern.finditer(text)
        if name == "email":
            matches = (
                m
                for m in matches
                if m.group().lower() != "openhands@all-hands.dev"
                and not _reserved_email_domain(m.group().rsplit("@", 1)[1].lower())
            )
        if next(matches, None) is not None:
            findings.append(name)
    return tuple(findings)


def _reserved_email_domain(domain: str) -> bool:
    return domain.endswith((".invalid", ".test")) or any(
        domain == reserved or domain.endswith("." + reserved)
        for reserved in ("example.com", "example.org", "example.net")
    )


def normalized_task(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_digest(text: str) -> str:
    return hashlib.sha256(normalized_task(text).encode()).hexdigest()


def task_shingles(text: str) -> frozenset[str]:
    words = re.findall(r"\w+", normalized_task(text))
    if len(words) < 5:
        return frozenset({text_digest(text)})
    return frozenset(
        hashlib.sha256(" ".join(words[i : i + 5]).encode()).hexdigest()[:16]
        for i in range(len(words) - 4)
    )


def minhash_bands(shingles: frozenset[str]) -> tuple[str, ...]:
    # Deterministic 32-permutation MinHash, eight four-value LSH bands.
    prime = (1 << 61) - 1
    values = [int(value[:15], 16) for value in shingles]
    signature = []
    for index in range(32):
        seed = hashlib.sha256(f"task-minhash-v1/{index}".encode()).digest()
        a = int.from_bytes(seed[:8], "big") % (prime - 1) + 1
        b = int.from_bytes(seed[8:16], "big") % prime
        signature.append(min((a * value + b) % prime for value in values))
    return tuple(
        hashlib.sha256(canonical_json(signature[i : i + 4]).encode()).hexdigest()
        for i in range(0, 32, 4)
    )


def episode_content_digest(episode: NormalizedEpisode) -> str:
    messages = [
        {"role": m.role, "content": m.content, "reasoning": m.reasoning, "action": m.action}
        for m in episode.messages
    ]
    return hashlib.sha256(canonical_json(messages).encode()).hexdigest()


def validate_tool_interactions(episode: NormalizedEpisode) -> tuple[str, ...]:
    if not isinstance(episode.tools, list) or not episode.tools:
        return ("missing_tool_schema",)
    definitions: dict[str, Any] = {}
    try:
        for definition in episode.tools:
            function = definition["function"]
            name, parameters = function["name"], function["parameters"]
            if not name or name in definitions or parameters.get("type") != "object":
                return ("invalid_tool_schema",)
            jsonschema.Draft202012Validator.check_schema(parameters)
            if any(not value.startswith("#") for value in _schema_refs(parameters)):
                return ("external_tool_schema_reference",)
            definitions[name] = jsonschema.Draft202012Validator(parameters)
    except (KeyError, TypeError, AttributeError, jsonschema.SchemaError):
        return ("invalid_tool_schema",)
    pending: set[str] = set()
    seen: set[str] = set()
    action_count = 0
    for message_index, message in enumerate(episode.messages):
        if message.role not in {"system", "developer", "user", "assistant", "tool"}:
            return ("unknown_role",)
        if message.role == "assistant":
            if pending:
                return ("missing_tool_response",)
            if message.action is None:
                continue
            if not isinstance(message.action, list) or not message.action:
                return ("unsupported_action_grammar",)
            for call in message.action:
                try:
                    identifier = call["id"]
                    function = call["function"]
                    name, arguments = function["name"], function["arguments"]
                    if not isinstance(identifier, str) or not identifier or identifier in seen:
                        return ("invalid_tool_call_id",)
                    if name not in definitions or not isinstance(arguments, dict):
                        return ("unknown_tool_or_arguments",)
                    definitions[name].validate(arguments)
                except (KeyError, TypeError, jsonschema.ValidationError):
                    return ("invalid_tool_arguments",)
                seen.add(identifier)
                if not (
                    name == "finish"
                    and message_index == len(episode.messages) - 1
                    and len(message.action) == 1
                ):
                    pending.add(identifier)
                action_count += 1
        elif message.role == "tool":
            if message.tool_call_id not in pending:
                return ("orphan_tool_response",)
            pending.remove(message.tool_call_id)
        elif pending and message.role == "user":
            return ("missing_tool_response",)
    if pending:
        return ("missing_tool_response",)
    return () if action_count else ("no_tool_actions",)


def _schema_refs(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str):
                yield item
            yield from _schema_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _schema_refs(item)


def audit_episode(episode: NormalizedEpisode) -> tuple[str, ...]:
    findings = []
    if not re.fullmatch(r"[0-9a-f]{40}", episode.source_revision):
        findings.append("unpinned_source")
    if episode.source_license != "cc-by-4.0":
        findings.append("unverified_dataset_license")
    if episode.metadata.get("repo_license") not in ALLOWED_REPO_LICENSES:
        findings.append("unverified_repo_license")
    if not episode.repo or not episode.task_text.strip():
        findings.append("missing_task_identity")
    findings.extend(validate_tool_interactions(episode))
    # Include unrendered raw metadata/patches so an accepted CAS blob is also scanned.
    findings.extend(sensitive_findings(canonical_json(asdict(episode))))
    return tuple(sorted(set(findings)))
