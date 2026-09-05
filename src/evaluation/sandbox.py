"""Offline OCI-rootfs commands with namespaces and aggregate cgroup budgets."""

from __future__ import annotations

import hashlib
import os
import selectors
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from src.data.quality import sensitive_findings


@dataclass(frozen=True)
class CommandBudget:
    wall_seconds: int = 60
    memory_bytes: int = 4 * 1024**3
    tasks_max: int = 64
    cpu_percent: int = 200
    output_bytes: int = 256 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("all command budgets must be positive integers")


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stop_reason: str
    text: str
    output_sha256: str
    elapsed_seconds: float
    sensitive_rules: tuple[str, ...]


class OfflineSandbox:
    def __init__(
        self,
        rootfs: Path,
        project: Path,
        project_path: str,
        *,
        readonly_project=False,
        verification_root: Path | None = None,
    ):
        self.rootfs, self.project = rootfs.resolve(strict=True), project.resolve(strict=True)
        if not self.rootfs.is_dir() or not self.project.is_dir():
            raise ValueError("rootfs and project must be directories")
        container_path = PurePosixPath(project_path)
        if (
            not container_path.is_absolute()
            or len(container_path.parts) != 2
            or ".." in container_path.parts
        ):
            raise ValueError("project must use one explicit root-level container path")
        if container_path.name in {
            "etc",
            "usr",
            "bin",
            "lib",
            "lib64",
            "dev",
            "proc",
            "sys",
            "tmp",
            "run",
            "root",
            "home",
        }:
            raise ValueError("project overlaps a protected container path")
        if self.project == Path("/") or self.rootfs == Path("/"):
            raise ValueError("host root cannot be a sandbox mount")
        self.project_path = str(container_path)
        self.readonly_project = readonly_project
        self.verification_root = (
            None if verification_root is None else verification_root.resolve(strict=True)
        )

    def command(self, script: str, budget: CommandBudget, unit: str) -> list[str]:
        if not isinstance(script, str) or not script or "\0" in script:
            raise ValueError("invalid shell command")
        arguments = [
            "systemd-run",
            "--user",
            "--quiet",
            "--wait",
            "--pipe",
            "--collect",
            f"--unit={unit}",
            f"--property=MemoryMax={budget.memory_bytes}",
            "--property=MemorySwapMax=0",
            f"--property=TasksMax={budget.tasks_max}",
            f"--property=CPUQuota={budget.cpu_percent}%",
            f"--property=RuntimeMaxSec={budget.wall_seconds}",
            "--property=KillMode=control-group",
            "bwrap",
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--assert-userns-disabled",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--ro-bind",
            str(self.rootfs),
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--size",
            str(256 * 1024**2),
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--tmpfs",
            "/root",
            "--tmpfs",
            "/home",
            "--ro-bind" if self.readonly_project else "--bind",
            str(self.project),
            self.project_path,
            "--setenv",
            "PATH",
            "/opt/conda/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--chdir",
            self.project_path,
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            script,
        ]
        if self.verification_root is not None:
            offset = arguments.index("--chdir")
            arguments[offset:offset] = [
                "--ro-bind",
                str(self.verification_root),
                "/tmp/verification",
            ]
        return arguments

    def run(self, script: str, budget: CommandBudget | None = None) -> CommandResult:
        budget = budget or CommandBudget()
        unit = "lla-sandbox-" + uuid.uuid4().hex
        arguments = self.command(script, budget, unit)
        started = time.monotonic()
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        output, digest, reason = bytearray(), hashlib.sha256(), "exited"
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    if time.monotonic() - started > budget.wall_seconds + 10:
                        reason = "wall_budget"
                        break
                    for key, _ in selector.select(timeout=0.1):
                        chunk = os.read(key.fileobj.fileno(), 16384)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        digest.update(chunk)
                        output.extend(chunk)
                        if len(output) > budget.output_bytes:
                            reason = "output_budget"
                            break
                    if reason != "exited":
                        break
            if reason != "exited":
                subprocess.run(
                    ["systemctl", "--user", "stop", unit],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
            code = process.wait(timeout=15)
        finally:
            if process.poll() is None:
                subprocess.run(
                    ["systemctl", "--user", "stop", unit],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                process.kill()
                process.wait()
            process.stdout.close()
        text = bytes(output[: budget.output_bytes]).decode("utf-8", errors="replace")
        rules = sensitive_findings(text)
        if rules:
            text = "[observation quarantined: sensitive content detected]"
            reason = "sensitive_observation"
        return CommandResult(
            code, reason, text, digest.hexdigest(), time.monotonic() - started, rules
        )
