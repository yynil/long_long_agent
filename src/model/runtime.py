"""Pinned official training runtime shared by generation and training entries."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.cpp_extension import load

from src.data.governance import sha256_file
from src.data.util import canonical_json
from src.training.tokenizer import RWKVByteTokenizer

ROOT = Path(__file__).resolve().parents[2]


def verify_patched_tree(tree: Path, revision: str, patches: list[dict], temporary: Path) -> dict:
    """Reconstruct the expected patched Git index without altering the worktree."""

    def git(*args, env=None):
        return subprocess.check_output(["git", *args], cwd=tree, env=env)

    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise ValueError("upstream revision mismatch")
    temporary.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "expected-index")}
        git("read-tree", revision, env=env)
        for item in patches:
            patch = ROOT / item["path"]
            if sha256_file(patch) != item["sha256"]:
                raise ValueError("compatibility patch hash mismatch")
            git("apply", "--cached", str(patch), env=env)
        paths = git("diff", "--cached", "--name-only", revision, env=env).decode().splitlines()
        changed = git("diff", "HEAD", "--name-only").decode().splitlines()
        if set(changed) != set(paths):
            raise ValueError("worktree changes differ from the fixed patch series")
        hashes = {}
        for name in paths:
            expected = hashlib.sha256(git("show", ":" + name, env=env)).hexdigest()
            if sha256_file(tree / name) != expected:
                raise ValueError("patched worktree content mismatch")
            hashes[name] = expected
        untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
        if any(Path(name).suffix in {".py", ".cpp", ".cu", ".h", ".so"} for name in untracked):
            raise ValueError("untracked executable source in upstream worktree")
    return {"revision": revision, "patched_files": hashes}


def compile_state_kernel(tree: Path, build_root: Path) -> None:
    source = tree / "rwkv7_fast_fused/cuda"
    destination = build_root / "state_passing_bf16_n64_c16"
    destination.mkdir(parents=True, exist_ok=True)
    load(
        name="rwkv7_statepassing_clampw",
        sources=[str(source / ("rwkv7_statepassing_clampw" + ext)) for ext in (".cu", ".cpp")],
        build_directory=str(destination),
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=[
            "-res-usage",
            "-D_N_=64",
            "-D_CHUNK_LEN_=16",
            "--use_fast_math",
            "-O3",
            "-Xptxas=-O3",
            "--extra-device-vectorization",
        ],
    )


def load_runtime(role: str, lm_tree: Path, cuda_tree: Path, build_root: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = yaml.safe_load((ROOT / "configs/base_model.yaml").read_text())
    spec = config["checkpoints"][role]
    storage = yaml.safe_load((ROOT / "configs/storage.yaml").read_text())["storage"]["local_root"]
    checkpoint = (
        Path(storage) / "models/blinkdl__rwkv7-g1" / config["model_repo"]["revision"] / spec["file"]
    )
    if sha256_file(checkpoint) != spec["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    trees = []
    for tree, upstream, series in (
        (lm_tree, "upstream", "compatibility_patch_series"),
        (cuda_tree, "state_passing_upstream", "state_passing_compatibility_patch_series"),
    ):
        trees.append(
            verify_patched_tree(
                tree, config[upstream]["revision"], config[series]["patches"], build_root
            )
        )
    vocabulary = lm_tree / config["tokenizer"]["source"]
    if sha256_file(vocabulary) != config["tokenizer"]["sha256"]:
        raise ValueError("tokenizer hash mismatch")
    environment = {
        "RWKV_JIT_ON": "0",
        "RWKV_HEAD_SIZE": "64",
        "RWKV_MY_TESTING": "x070",
        "RWKV_KERNEL": "",
        "RWKV_HEAD_L2WRAP_CE_CHUNK": "0",
        "TORCH_CUDA_ARCH_LIST": ".".join(map(str, torch.cuda.get_device_capability())),
    }
    for key, value in environment.items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError(f"runtime environment conflicts with fixed setting: {key}")
        os.environ[key] = value
    compile_state_kernel(cuda_tree, build_root)
    train_root = lm_tree / config["upstream"]["reference_dir"]
    previous = Path.cwd()
    try:
        os.chdir(train_root)
        sys.path.insert(0, str(train_root / "src"))
        module = importlib.import_module("model")
    finally:
        os.chdir(previous)
    if Path(module.__file__).resolve() != train_root / "src/model.py":
        raise ValueError("unexpected imported RWKV implementation")
    architecture = spec["architecture"]
    width = architecture["embedding_dim"]
    args = SimpleNamespace(
        n_layer=architecture["layers"],
        n_embd=width,
        vocab_size=65536,
        ctx_len=architecture["context_length"],
        head_size=architecture["head_size"],
        dim_att=width,
        dim_ffn=width * 7 // 2,
        grad_cp=0,
        my_testing="x070",
        **{
            key: architecture[key]
            for key in ("decay_lora_rank", "aaa_lora_rank", "mv_lora_rank", "gate_lora_rank")
        },
    )
    network = module.RWKV(args)
    network.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    network = network.to(device="cuda", dtype=torch.bfloat16).eval()
    network.requires_grad_(False)
    provenance = {
        "checkpoint_sha256": spec["sha256"],
        "tokenizer_sha256": config["tokenizer"]["sha256"],
        "upstream_trees": trees,
        "model_code_sha256": hashlib.sha256(
            canonical_json(
                {
                    "trees": trees,
                    "local": {
                        p.name: sha256_file(p) for p in sorted((ROOT / "src/model").glob("*.py"))
                    },
                }
            ).encode()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "runtime_environment": environment,
        "short_inference_bf16_matmul_rows": "native; alignment is diagnostic-only",
        "matmul_precision": {
            "fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        },
    }
    return network, RWKVByteTokenizer(vocabulary), provenance
