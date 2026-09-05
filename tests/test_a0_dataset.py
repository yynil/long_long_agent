from dataclasses import asdict

import pytest

from src.data.blob_store import BlobStore
from src.data.canonical import canonicalize_episode
from src.data.types import NormalizedEpisode, NormalizedMessage
from src.training.a0_dataset import prepare_overfit_inputs, restore_episode


def test_restore_canonical_episode_preserves_messages_and_source(tmp_path):
    episode = NormalizedEpisode(
        "fixture",
        "a" * 40,
        "synthetic",
        "one",
        "task-one",
        "Inspect configuration.",
        (
            NormalizedMessage("user", "Inspect configuration."),
            NormalizedMessage("assistant", "Verified."),
        ),
        repo="fixture/project",
        base_commit="b" * 40,
        success=True,
    )
    blobs = BlobStore(tmp_path)
    row, _ = canonicalize_episode(episode, blobs)
    blob = blobs.read_json(row["raw_trace_ref"])
    assert asdict(restore_episode(row, blob)) == asdict(episode)
    blob["source_revision"] = "c" * 40
    with pytest.raises(ValueError, match="identity"):
        restore_episode(row, blob)


def test_real_input_selection_validates_budgets_before_reading(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        prepare_overfit_inputs(tmp_path, None, count=3, max_tokens=8192, seed=1)
    with pytest.raises(ValueError, match="16-aligned"):
        prepare_overfit_inputs(tmp_path, None, count=32, max_tokens=8191, seed=1)
