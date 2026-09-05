from dataclasses import replace

import pytest

from src.data.adapters.common import normalize_openai_messages
from src.data.quality import audit_episode, minhash_bands, sensitive_findings, task_shingles
from src.data.types import NormalizedEpisode


def episode():
    rows = [
        {"role": "user", "content": "Inspect the project."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "read", "arguments": {"path": "README"}},
                }
            ],
        },
        {"role": "tool", "content": "instructions"},
        {"role": "assistant", "content": "Verified."},
    ]
    return NormalizedEpisode(
        "fixture",
        "a" * 40,
        "cc-by-4.0",
        "one",
        "repo-1",
        "Inspect the project.",
        tuple(normalize_openai_messages(rows)),
        repo="owner/repo",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        metadata={"repo_license": "MIT"},
    )


def test_unambiguous_missing_call_id_is_recorded_and_valid():
    sample = episode()
    assert sample.messages[2].tool_call_id == "one"
    assert sample.messages[2].metadata["tool_pairing"] == "inferred_single_pending_call_v1"
    assert audit_episode(sample) == ()


def test_missing_result_invalid_arguments_and_unknown_license_rejected():
    sample = episode()
    assert "missing_tool_response" in audit_episode(replace(sample, messages=sample.messages[:2]))
    assert "unverified_repo_license" in audit_episode(replace(sample, metadata={}))
    messages = list(sample.messages)
    messages[1] = replace(
        messages[1], action=[{"id": "one", "function": {"name": "read", "arguments": {}}}]
    )
    assert "invalid_tool_arguments" in audit_episode(replace(sample, messages=tuple(messages)))


def test_ambiguous_parallel_call_ids_not_guessed():
    rows = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": name, "function": {"name": "read", "arguments": {}}} for name in ("a", "b")
            ],
        },
        {"role": "tool", "content": "ambiguous"},
    ]
    assert normalize_openai_messages(rows)[1].tool_call_id is None


@pytest.mark.parametrize(
    "text,rule",
    [
        ("test.user@private.testdomain", "email"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private_key"),
        ("ghp_" + "x" * 36, "github_token"),
        ('api_key="' + "x" * 32 + '"', "credential_assignment"),
    ],
)
def test_sensitive_findings_only_return_rule_names(text, rule):
    findings = sensitive_findings(text)
    assert rule in findings
    assert text not in str(findings)


def test_package_versions_and_documented_service_account_are_not_personal_email():
    assert sensitive_findings("pkg@3.2.1 openhands@all-hands.dev demo@example.com") == ()


def test_minhash_is_stable_and_uses_normalized_content():
    first = task_shingles("Please inspect the source and fix the failing test.")
    second = task_shingles("PLEASE  inspect the source and fix the failing test.")
    assert first == second and minhash_bands(first) == minhash_bands(second)
