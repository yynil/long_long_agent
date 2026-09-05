"""Reproduce a pinned dev task's buggy/gold verifier without running an Agent."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pyarrow.parquet as pq
import yaml

from src.data.governance import load_heldout_policy, sha256_file

from .sandbox import CommandBudget, OfflineSandbox
from .verifier import test_outcome

ROOT = Path(__file__).resolve().parents[2]


def verify_fixture(config_path: Path, rootfs: Path, harness: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite environment evidence")
    config = yaml.safe_load(config_path.read_text())
    jsonschema.validate(
        config, json.loads((ROOT / "schemas/environment_fixture.schema.json").read_text())
    )
    storage = Path(
        yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    )
    if not rootfs.is_relative_to(storage / "environments") or rootfs.name != "rootfs":
        raise ValueError("rootfs must be an unpacked image in the controlled environment store")
    unpacked = json.loads((rootfs.parent / "umoci.json").read_text())
    if (
        unpacked["from_descriptor_path"]["descriptor_walk"][-1]["digest"]
        != config["normalized_image_digest"]
    ):
        raise ValueError("unpacked image descriptor mismatch")
    taskfile = (
        storage
        / "raw/swe_rebench_v2_tasks"
        / config["task_source_revision"]
        / "data/train-00000-of-00001.parquet"
    )
    if sha256_file(taskfile) != config["task_file_sha256"]:
        raise ValueError("fixture metadata hash mismatch")
    matches = pq.read_table(
        taskfile, filters=[("instance_id", "=", config["instance_id"])]
    ).to_pylist()
    if len(matches) != 1:
        raise ValueError("fixture task ID is not unique")
    task = matches[0]
    for source, key in (
        ("base_commit", "base_commit"),
        ("repo", "repo"),
        ("license", "repo_license"),
    ):
        if task[source] != config[key]:
            raise ValueError("fixture source metadata mismatch")
    policy = load_heldout_policy(ROOT / "data/heldout/manifest.yaml")
    if policy.split_for_episode(task["repo"], task["instance_id"]) != config["split"]:
        raise ValueError("fixture split mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=harness, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=harness
    )
    if revision != config["harness_revision"] or dirty:
        raise ValueError("harness is not the pinned clean revision")
    spec = importlib.util.spec_from_file_location(
        "swe_rebench_fixture_oracle", harness / "scripts/eval.py"
    )
    oracle = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(harness))
    spec.loader.exec_module(oracle)
    parser = oracle.get_parser(task["install_config"]["log_parser"])
    project_path = "/" + task["repo"].split("/")[1]
    source = rootfs / project_path.lstrip("/")
    base = OfflineSandbox(rootfs, source, project_path, readonly_project=True)
    budget = CommandBudget(
        config["command_wall_seconds"],
        config["memory_bytes"],
        config["tasks_max"],
        config["cpu_percent"],
        config["output_bytes"],
    )
    identity = base.run(f"git -c safe.directory={project_path} rev-parse HEAD", budget)
    if identity.exit_code or identity.text.strip() != task["base_commit"]:
        raise ValueError("image project is not the expected base commit")
    run_root = storage / "artifacts/environment_fixture"
    run_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="flake8-179-", dir=run_root))
    results = {}
    for mode in ("buggy", "gold"):
        project, verification = directory / mode, directory / (mode + "-verification")
        shutil.copytree(source, project, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        verification.mkdir()
        (verification / "test.patch").write_text(task["test_patch"])
        if mode == "gold":
            (verification / "gold.patch").write_text(task["patch"])
        sandbox = OfflineSandbox(rootfs, project, project_path, verification_root=verification)
        setup = [
            "set -eu",
            "git init -q",
            "git config user.name fixture",
            "git config user.email fixture@example.invalid",
            "git add .",
            "git -c core.hooksPath=/dev/null commit -qm immutable-base",
        ]
        if mode == "gold":
            setup.append("git apply --recount --whitespace=nowarn /tmp/verification/gold.patch")
        setup.append("git apply --recount --whitespace=nowarn /tmp/verification/test.patch")
        prepared = sandbox.run("\n".join(setup), budget)
        if prepared.exit_code or prepared.stop_reason != "exited":
            results[mode] = {"status": "setup_failed", "command": asdict(prepared)}
            continue
        test_commands = task["install_config"]["test_cmd"]
        if isinstance(test_commands, str):
            test_commands = [test_commands]
        run = sandbox.run("\n".join(test_commands), budget)
        parsed = {oracle._normalize_test_name(k): v for k, v in parser(run.text).items()}
        metrics = test_outcome(
            parsed,
            [oracle._normalize_test_name(n) for n in task["FAIL_TO_PASS"]],
            [oracle._normalize_test_name(n) for n in task["PASS_TO_PASS"]],
        )
        # No raw log is persisted. A bounded/redacted observation is retained only
        # in memory for the parser; the report keeps counts, stop reason and hash.
        results[mode] = {
            "status": "evaluated",
            "exit_code": run.exit_code,
            "stop_reason": run.stop_reason,
            "output_sha256": run.output_sha256,
            "seconds": run.elapsed_seconds,
            **metrics,
        }
    passed = (
        results.get("buggy", {}).get("status") == "evaluated"
        and results["buggy"]["fail_to_pass_passed"] < results["buggy"]["fail_to_pass_total"]
        and results["buggy"]["fail_to_pass_failed"] > 0
        and results["buggy"]["stop_reason"] == "exited"
        and results["buggy"]["exit_code"] != 0
        and results["buggy"]["regressions"] == 0
        and results["buggy"]["missing_tests"] == 0
        and results.get("gold", {}).get("official_passed_match") is True
        and results["gold"]["exit_code"] == 0
        and results["gold"]["stop_reason"] == "exited"
    )
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "purpose": config["purpose"],
        "config_sha256": sha256_file(config_path),
        "image_digest": config["image_digest"],
        "normalized_image_digest": config["normalized_image_digest"],
        "unpack_metadata_sha256": sha256_file(rootfs.parent / "umoci.json"),
        "base_commit": config["base_commit"],
        "harness_revision": revision,
        "heldout_sha256": policy.manifest_sha256,
        "split": config["split"],
        "network": "none",
        "agent_executed": False,
        "future_git_history_exposed": False,
        "run_root": str(directory),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
