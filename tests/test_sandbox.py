from pathlib import Path

import pytest

from src.evaluation.sandbox import CommandBudget, OfflineSandbox


def test_sandbox_commands_have_no_host_network_credentials_or_unbounded_resources(tmp_path):
    rootfs, project = tmp_path / "rootfs", tmp_path / "project"
    rootfs.mkdir()
    project.mkdir()
    sandbox = OfflineSandbox(rootfs, project, "/project")
    command = sandbox.command("printf ok", CommandBudget(), "fixture")
    for required in (
        "--unshare-all",
        "--clearenv",
        "--disable-userns",
        "--property=KillMode=control-group",
        "--property=MemorySwapMax=0",
    ):
        assert required in command
    assert "--share-net" not in command
    assert "--dev-bind" not in command
    assert str(Path.home()) not in command
    assert any(item.startswith("--property=TasksMax=") for item in command)
    assert any(item.startswith("--property=RuntimeMaxSec=") for item in command)


def test_sandbox_rejects_unsafe_mounts_and_budgets(tmp_path):
    with pytest.raises(ValueError):
        OfflineSandbox(Path("/"), tmp_path, "/project")
    for name in ("/usr", "/tmp", "/a/../b", "relative"):
        with pytest.raises(ValueError):
            OfflineSandbox(tmp_path, tmp_path, name)
    with pytest.raises(ValueError):
        CommandBudget(tasks_max=True)
    with pytest.raises(ValueError):
        CommandBudget(wall_seconds=0)
