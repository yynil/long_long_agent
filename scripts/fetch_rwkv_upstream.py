#!/usr/bin/env python3
"""Clone or update the pinned RWKV-LM and RWKV-CUDA checkouts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/base_model.yaml")
    parser.add_argument(
        "--component",
        choices=("all", "rwkv_lm", "rwkv_cuda"),
        default="all",
    )
    args = parser.parse_args()

    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    components = {
        "rwkv_lm": ("upstream", REPO_ROOT / "external/RWKV-LM"),
        "rwkv_cuda": ("state_passing_upstream", REPO_ROOT / "external/RWKV-CUDA"),
    }
    selected = (
        components.items()
        if args.component == "all"
        else [(args.component, components[args.component])]
    )
    for name, (config_key, target_path) in selected:
        repo = config[config_key]["repo"]
        revision = config[config_key]["revision"]
        if len(revision) != 40:
            raise SystemExit(f"Refusing non-pinned revision for {name}: {revision!r}")

        target = target_path.resolve()
        if not (target / ".git").is_dir():
            if target.exists() and any(target.iterdir()):
                raise SystemExit(f"Target exists and is not an empty Git repository: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--filter=blob:none", "--no-checkout", repo, str(target))
        remote = run("git", "remote", "get-url", "origin", cwd=target)
        if remote.rstrip("/") != repo.rstrip("/"):
            raise SystemExit(f"Unexpected origin for {target}: {remote}")
        run("git", "fetch", "--depth", "1", "origin", revision, cwd=target)
        run("git", "checkout", "--detach", revision, cwd=target)
        actual = run("git", "rev-parse", "HEAD", cwd=target)
        if actual != revision:
            raise SystemExit(f"Checkout mismatch: {actual} != {revision}")
        print(f"{name} pinned at {actual} in {target}")


if __name__ == "__main__":
    main()
