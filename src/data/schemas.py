"""Versioned PyArrow schemas for canonical Agent data."""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = "1.1.0"
SCHEMA_METADATA = {b"schema_version": SCHEMA_VERSION.encode("ascii")}


EPISODES = pa.schema(
    [
        ("episode_id", pa.string()),
        ("task_id", pa.string()),
        ("source_record_id", pa.string()),
        ("split_group", pa.string()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_license", pa.string()),
        ("harness", pa.string()),
        ("tools_schema", pa.large_string()),
        ("task_text", pa.large_string()),
        ("acceptance_criteria", pa.large_string()),
        ("repo", pa.string()),
        ("base_commit", pa.string()),
        ("env_image_digest", pa.string()),
        ("verifier_id", pa.string()),
        ("verifier_version", pa.string()),
        ("teacher_model", pa.string()),
        ("teacher_prompt_version", pa.string()),
        ("success", pa.bool_()),
        ("terminal_reward", pa.float32()),
        ("turn_count", pa.int32()),
        ("failure_type", pa.string()),
        ("raw_trace_ref", pa.string()),
    ],
    metadata=SCHEMA_METADATA,
)


DECISIONS = pa.schema(
    [
        ("episode_id", pa.string()),
        ("turn_id", pa.int32()),
        ("decision_id", pa.string()),
        ("prefix_messages_ref", pa.string()),
        ("current_observation", pa.large_string()),
        ("tools_schema", pa.large_string()),
        ("teacher_think_raw", pa.large_string()),
        ("teacher_think_tokens", pa.int32()),
        ("macro_thoughts", pa.large_string()),
        ("teacher_action", pa.large_string()),
        ("next_tool_result", pa.large_string()),
        ("env_delta", pa.large_string()),
        ("verifier_delta", pa.float32()),
        ("progress_label", pa.float32()),
        ("regression_label", pa.bool_()),
        ("recovery_label", pa.bool_()),
        ("decision_type", pa.list_(pa.string())),
    ],
    metadata=SCHEMA_METADATA,
)


SNAPSHOTS = pa.schema(
    [
        ("snapshot_id", pa.string()),
        ("episode_id", pa.string()),
        ("turn_id", pa.int32()),
        ("env_snapshot_ref", pa.string()),
        ("prefix_messages_ref", pa.string()),
        ("task_contract", pa.large_string()),
        ("current_observation", pa.large_string()),
        ("milestones", pa.large_string()),
        ("ledger", pa.large_string()),
        ("known_constraints", pa.large_string()),
    ],
    metadata=SCHEMA_METADATA,
)


FORKS = pa.schema(
    [
        ("snapshot_id", pa.string()),
        ("policy_checkpoint", pa.string()),
        ("candidate_id", pa.string()),
        ("recurrent_depth", pa.int16()),
        ("latent_mode", pa.string()),
        ("sampling_seed", pa.int64()),
        ("action", pa.large_string()),
        ("parse_valid", pa.bool_()),
        ("exec_valid", pa.bool_()),
        ("new_information", pa.bool_()),
        ("immediate_progress", pa.float32()),
        ("regression", pa.bool_()),
        ("h4_progress", pa.float32()),
        ("h16_progress", pa.float32()),
        ("terminal_success", pa.bool_()),
        ("model_latency_ms", pa.float32()),
        ("model_flops_estimate", pa.float64()),
        ("tool_latency_ms", pa.float32()),
    ],
    metadata=SCHEMA_METADATA,
)


TABLE_SCHEMAS = {
    "episodes": EPISODES,
    "decisions": DECISIONS,
    "snapshots": SNAPSHOTS,
    "forks": FORKS,
}
