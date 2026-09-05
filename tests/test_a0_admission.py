import json
import subprocess
from pathlib import Path

import pytest
import yaml

from src.data.a0_release import load_config, verify_release
from src.data.source_io import verified_inventory


def test_old_production_entrypoint_fails_before_data_access():
    result = subprocess.run(
        [
            str(Path.cwd() / ".venv/bin/python"),
            "scripts/build_canonical.py",
            "--source",
            "unknown",
            "--release-id",
            "should-not-exist",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "audited admission" in result.stderr


def test_a0_config_rejects_unknown_fields_and_weakened_gates(tmp_path):
    config = yaml.safe_load(Path("configs/a0_release.yaml").read_text())
    for mutation in ({"extra": True}, {"require_task_join": False}, {"release_id": "../escape"}):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**config, **mutation}))
        with pytest.raises(ValueError):
            load_config(path)


def test_unadmitted_or_incomplete_release_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": "1.1.0", "preview": False})
    )
    with pytest.raises(ValueError, match="not admitted"):
        verify_release(tmp_path)


def test_source_manifest_existence_does_not_substitute_for_content_hash(tmp_path):
    root = tmp_path / "raw/example" / ("a" * 40)
    root.mkdir(parents=True)
    (root / "data.parquet").write_bytes(b"changed content")
    (root / "download_manifest.json").write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "repo_id": "x/y",
                "files": [{"path": "data.parquet", "lfs_sha256": "b" * 64}],
            }
        )
    )
    with pytest.raises(ValueError, match="content hash"):
        verified_inventory(
            {
                "source_id": "example",
                "revision": "a" * 40,
                "repo_id": "x/y",
                "allow_patterns": ["*.parquet"],
            },
            tmp_path,
        )
