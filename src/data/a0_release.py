"""Audit candidate locators, then build and verify an immutable A0 release."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.training.episode_collator import decision_message_indices, tokenize_decision
from src.training.tokenizer import RWKVByteTokenizer

from .adapters import get_adapter
from .adapters.common import source_row_digest
from .blob_store import BlobStore
from .canonical import canonicalize_episode, stable_id
from .contamination import ContaminationIndex
from .governance import canonical_group_identity, load_heldout_policy, normalize_repo, sha256_file
from .quality import QUALITY_VERSION, audit_episode, episode_content_digest, text_digest
from .schemas import SCHEMA_VERSION, TABLE_SCHEMAS
from .source_io import iter_rows, load_sources, verified_inventory
from .util import canonical_json

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEYS = {
    "schema_version",
    "release_id",
    "source_id",
    "source_registry",
    "task_registry",
    "heldout_manifest",
    "training_config",
    "seed",
    "episodes_per_split",
    "teachers",
    "decisions_per_episode",
    "require_known_outcome",
    "require_task_join",
    "maximum_scan_episodes",
}


def implementation_digest() -> str:
    paths = sorted(
        [
            *(REPO_ROOT / "src/data").rglob("*.py"),
            REPO_ROOT / "src/training/episode_collator.py",
            REPO_ROOT / "src/training/tokenizer.py",
            REPO_ROOT / "schemas/a0_admission.schema.json",
        ]
    )
    return hashlib.sha256(
        canonical_json(
            {p.relative_to(REPO_ROOT).as_posix(): sha256_file(p) for p in paths}
        ).encode()
    ).hexdigest()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS or config["schema_version"] != 1:
        raise ValueError("unknown or missing A0 config fields/version")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", config["release_id"]):
        raise ValueError("unsafe release ID")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if set(config["episodes_per_split"]) != {"train", "dev", "test"}:
        raise ValueError("all three splits are required")
    numbers = [
        *config["episodes_per_split"].values(),
        config["decisions_per_episode"],
        config["maximum_scan_episodes"],
    ]
    if any(type(value) is not int or value <= 0 for value in numbers):
        raise ValueError("A0 counts must be positive integers")
    if config["require_known_outcome"] is not True or config["require_task_join"] is not True:
        raise ValueError("A0 requires verified task join and known outcome")
    if config["teachers"] != ["MiniMax-M2.5", "Qwen3.5-122B"] or any(
        count % 2 for count in config["episodes_per_split"].values()
    ):
        raise ValueError("A0 requires equal per-split quotas for its two teachers")
    return config


def load_tasks(config: dict, data_root: Path):
    registry, _ = load_sources(REPO_ROOT / config["task_registry"])
    source = registry["sources"][0]
    root, files = verified_inventory(source, data_root)
    tasks = {}
    columns = ["instance_id", "repo", "base_commit", "license", "image_name"]
    for _, _, row in iter_rows(root, files, columns=columns):
        if row["instance_id"] in tasks:
            raise ValueError("task join key is not unique")
        tasks[row["instance_id"]] = row
    return tasks, {"source_id": source["source_id"], "revision": source["revision"], "files": files}


def audit_candidates(config_path: Path, index_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("refusing to overwrite A0 admission")
    config = load_config(config_path)
    registry, data_root = load_sources(REPO_ROOT / config["source_registry"])
    source = next(s for s in registry["sources"] if s["source_id"] == config["source_id"])
    policy = load_heldout_policy(REPO_ROOT / config["heldout_manifest"])
    policy.verify_source(source["source_id"], source["revision"])
    raw, files = verified_inventory(source, data_root)
    tasks, task_manifest = load_tasks(config, data_root)
    training_path = REPO_ROOT / config["training_config"]
    training = yaml.safe_load(training_path.read_text())
    vocabulary = REPO_ROOT / training["tokenizer"]["vocabulary"]
    if sha256_file(vocabulary) != training["tokenizer"]["sha256"]:
        raise ValueError("tokenizer content hash mismatch")
    tokenizer = RWKVByteTokenizer(vocabulary)
    contamination = ContaminationIndex(index_path, policy.manifest_sha256)
    expected_source = next(
        s for s in contamination.manifest["sources"] if s["source_id"] == source["source_id"]
    )
    if expected_source["revision"] != source["revision"] or expected_source["files"] != files:
        raise ValueError("contamination index does not match source contents")
    adapter = get_adapter(source["adapter"])
    rejection_counts, split_counts, teacher_counts, outcome_counts = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    quarantined = []
    accepted, tasks_seen, content_seen = [], set(), set()
    ordered_files = sorted(
        files, key=lambda x: hashlib.sha256(f"{config['seed']}/{x['path']}".encode()).hexdigest()
    )
    scanned = 0
    teacher_split_counts = Counter()
    try:
        for source_file, ordinal, row in iter_rows(raw, ordered_files):
            if scanned >= config["maximum_scan_episodes"]:
                break
            scanned += 1
            digest = source_row_digest(row)
            try:
                episode = adapter(row, source["revision"], source["declared_license"])
                identity = canonical_group_identity(episode.repo, episode.task_id)
                split = policy.split_for_identity(identity)
                findings = list(audit_episode(episode))
                task = tasks.get(episode.task_id)
                if task is None:
                    findings.append("task_join_missing")
                elif (
                    normalize_repo(task["repo"]) != normalize_repo(episode.repo or "")
                    or task["license"] != episode.metadata.get("repo_license")
                    or not re.fullmatch(r"[a-f0-9]{40}", task["base_commit"])
                ):
                    findings.append("task_join_conflict")
                if episode.success is None:
                    findings.append("unknown_outcome")
                if not findings:
                    findings.extend(
                        contamination.findings(episode.task_text, row["messages"], split)
                    )
                content = episode_content_digest(episode)
                task_digest = text_digest(episode.task_text)
                if (
                    episode.task_id in tasks_seen
                    or task_digest in tasks_seen
                    or content in content_seen
                ):
                    findings.append("duplicate_selected_task_or_trace")
                if split_counts[split] >= config["episodes_per_split"][split]:
                    findings.append("split_quota_full")
                if (
                    teacher_split_counts[(split, episode.teacher_model)]
                    >= config["episodes_per_split"][split] // 2
                ):
                    findings.append("teacher_quota_full")
                if findings:
                    rejection_counts.update(set(findings))
                    quarantined.append({"record_sha256": digest, "reasons": sorted(set(findings))})
                    continue
                episode = replace(episode, base_commit=task["base_commit"])
                eligible = decision_message_indices(episode)
                n = config["decisions_per_episode"]
                if len(eligible) < n:
                    rejection_counts["insufficient_decisions"] += 1
                    continue
                # Evenly spread over the whole trajectory; never pick only the shortest actions.
                chosen = (
                    [eligible[i * (len(eligible) - 1) // (n - 1)] for i in range(n)]
                    if n > 1
                    else [eligible[-1]]
                )
                windows = [
                    tokenize_decision(
                        episode, i, tokenizer, max_tokens=training["decision_windows"]["max_tokens"]
                    )
                    for i in chosen
                ]
            except (TypeError, ValueError, KeyError, jsonschema.ValidationError) as error:
                reason = (
                    "protected_context_overflow"
                    if "protected context" in str(error)
                    else "normalization_or_schema_error"
                )
                rejection_counts[reason] += 1
                quarantined.append({"record_sha256": digest, "reasons": [reason]})
                continue
            accepted.append(
                {
                    "source_file": source_file,
                    "row_index": ordinal,
                    "record_sha256": digest,
                    "episode_id": stable_id(episode.source_dataset, episode.source_record_id),
                    "split": split,
                    "message_indices": chosen,
                    "base_commit": episode.base_commit,
                    "repo_license": episode.metadata["repo_license"],
                    "window_tokens": [len(w.episode.token_ids) - 1 for w in windows],
                    "dropped_messages": [w.dropped_message_count for w in windows],
                }
            )
            tasks_seen.update((episode.task_id, task_digest))
            content_seen.add(content)
            split_counts[split] += 1
            teacher_counts[episode.teacher_model] += 1
            teacher_split_counts[(split, episode.teacher_model)] += 1
            outcome_counts[str(episode.success)] += 1
            if len(accepted) % 25 == 0:
                print(
                    json.dumps(
                        {"scanned": scanned, "accepted": len(accepted), "split": dict(split_counts)}
                    ),
                    flush=True,
                )
            if all(split_counts[s] == count for s, count in config["episodes_per_split"].items()):
                break
            if scanned >= config["maximum_scan_episodes"]:
                break
    finally:
        contamination.close()
    complete = all(split_counts[s] == count for s, count in config["episodes_per_split"].items())
    report = {
        "schema_version": 1,
        "status": "passed" if complete else "insufficient_eligible_data",
        "quality_version": QUALITY_VERSION,
        "implementation_sha256": implementation_digest(),
        "config_sha256": sha256_file(config_path),
        "training_config_sha256": sha256_file(training_path),
        "source_id": source["source_id"],
        "source_revision": source["revision"],
        "input_files": files,
        "task_source": task_manifest,
        "heldout_sha256": policy.manifest_sha256,
        "index_sha256": sha256_file(index_path),
        "source_readme_sha256": sha256_file(raw / "README.md"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "statistics": {
            "scanned": scanned,
            "accepted": len(accepted),
            "splits": dict(split_counts),
            "teachers": dict(teacher_counts),
            "outcomes": dict(outcome_counts),
            "rejections": dict(rejection_counts),
        },
        "accepted": accepted,
    }
    jsonschema.validate(
        report, json.loads((REPO_ROOT / "schemas/a0_admission.schema.json").read_text())
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".quarantine.jsonl").write_text(
        "".join(canonical_json(r) + "\n" for r in quarantined)
    )
    return report


def load_admission(
    config_path: Path, admission: Path, index_path: Path
) -> tuple[dict, dict, Path, dict]:
    config = load_config(config_path)
    report = json.loads(admission.read_text())
    jsonschema.validate(
        report, json.loads((REPO_ROOT / "schemas/a0_admission.schema.json").read_text())
    )
    if report["status"] != "passed" or report["quality_version"] != QUALITY_VERSION:
        raise ValueError("A0 admission has not passed")
    for actual, expected in (
        (sha256_file(config_path), report["config_sha256"]),
        (implementation_digest(), report["implementation_sha256"]),
        (sha256_file(REPO_ROOT / config["training_config"]), report["training_config_sha256"]),
        (sha256_file(REPO_ROOT / config["heldout_manifest"]), report["heldout_sha256"]),
        (sha256_file(index_path), report["index_sha256"]),
    ):
        if actual != expected:
            raise ValueError("admission provenance changed; re-audit required")
    registry, data_root = load_sources(REPO_ROOT / config["source_registry"])
    source = next(s for s in registry["sources"] if s["source_id"] == report["source_id"])
    raw, files = verified_inventory(source, data_root)
    if files != report["input_files"] or source["revision"] != report["source_revision"]:
        raise ValueError("admitted source changed")
    if sha256_file(raw / "README.md") != report["source_readme_sha256"]:
        raise ValueError("license evidence changed")
    _, task_source = load_tasks(config, data_root)
    if task_source != report["task_source"]:
        raise ValueError("task provenance changed")
    if Counter(e["split"] for e in report["accepted"]) != config["episodes_per_split"]:
        raise ValueError("admission split quotas mismatch")
    if len({e["episode_id"] for e in report["accepted"]}) != len(report["accepted"]):
        raise ValueError("duplicate admitted episode")
    for entry in report["accepted"]:
        if entry["source_file"] not in {item["path"] for item in files}:
            raise ValueError("admitted locator is outside verified sources")
        if len(entry["message_indices"]) != config["decisions_per_episode"]:
            raise ValueError("admitted decision count mismatch")
    return config, report, data_root, source


def build_release(config_path: Path, admission_path: Path, index_path: Path) -> Path:
    config, report, data_root, source = load_admission(config_path, admission_path, index_path)
    destination = data_root / "releases" / config["release_id"]
    temporary = destination.with_name(f".{destination.name}.building")
    if destination.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite release or interrupted build")
    temporary.mkdir(parents=True)
    raw = data_root / "raw" / source["source_id"] / source["revision"]
    adapter = get_adapter(source["adapter"])
    tasks, _ = load_tasks(config, data_root)
    policy = load_heldout_policy(REPO_ROOT / config["heldout_manifest"])
    blobs = BlobStore(data_root / "blobs")
    wanted = defaultdict(dict)
    for entry in report["accepted"]:
        wanted[entry["source_file"]][entry["row_index"]] = entry
    if sum(len(x) for x in wanted.values()) != len(report["accepted"]):
        raise ValueError("duplicate admitted locator")
    rows = {"episodes": [], "decisions": [], "snapshots": [], "forks": []}
    splits = {"train": [], "dev": [], "test": []}
    for filename, ordinals in wanted.items():
        for _, ordinal, row in iter_rows(raw, [{"path": filename}]):
            if ordinal not in ordinals:
                continue
            entry = ordinals[ordinal]
            if source_row_digest(row) != entry["record_sha256"]:
                raise ValueError("admitted row changed")
            episode = adapter(row, source["revision"], source["declared_license"])
            if audit_episode(episode):
                raise ValueError("admitted episode failed repeat quality check")
            task = tasks[episode.task_id]
            if (
                task["base_commit"] != entry["base_commit"]
                or task["license"] != entry["repo_license"]
            ):
                raise ValueError("admitted task/commit/license mismatch")
            if policy.split_for_episode(episode.repo, episode.task_id) != entry["split"]:
                raise ValueError("admitted split conflicts with frozen policy")
            episode = replace(episode, base_commit=entry["base_commit"])
            episode_row, decisions = canonicalize_episode(episode, blobs)
            if episode_row["episode_id"] != entry["episode_id"]:
                raise ValueError("admitted episode identity changed")
            # canonical turn IDs count action/final messages. Selected tokenizer indices
            # refer to the same nonempty assistant sequence in these validated sources.
            canonical_indices = [
                i
                for i, m in enumerate(episode.messages)
                if m.role == "assistant" and (m.action is not None or m.content)
            ]
            selected = {canonical_indices.index(i) for i in entry["message_indices"]}
            selected_decisions = [d for d in decisions if d["turn_id"] in selected]
            if len(selected_decisions) != config["decisions_per_episode"]:
                raise ValueError("admitted decisions do not map to canonical rows")
            rows["episodes"].append(episode_row)
            rows["decisions"].extend(selected_decisions)
            splits[entry["split"]].append(episode_row["episode_id"])
    if len(rows["episodes"]) != len(report["accepted"]):
        raise ValueError("missing admitted episodes")
    files = {}
    for table, schema in TABLE_SCHEMAS.items():
        filename = f"{table}.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows[table], schema=schema),
            temporary / filename,
            compression="zstd",
        )
        files[filename] = sha256_file(temporary / filename)
    (temporary / "splits").mkdir()
    for split, ids in splits.items():
        path = temporary / "splits" / f"{split}.txt"
        path.write_text("".join(value + "\n" for value in sorted(ids)))
        files[path.relative_to(temporary).as_posix()] = sha256_file(path)
    (temporary / "reports").mkdir()
    summaries = {
        "quality": {
            "schema_version": 1,
            "status": "passed",
            "statistics": report["statistics"],
            "decisions": len(rows["decisions"]),
            "quality_version": QUALITY_VERSION,
        },
        "contamination": {
            "schema_version": 1,
            "status": "passed",
            "index_sha256": report["index_sha256"],
            "heldout_sha256": report["heldout_sha256"],
            "accepted_cross_split_matches": 0,
            "limitation": "MinHash task near-dedup is approximate; pretrained contamination is not measured.",
        },
        "licenses": {
            "schema_version": 1,
            "status": "passed",
            "dataset_license": source["declared_license"],
            "source_revision": source["revision"],
            "publisher_readme_sha256": report["source_readme_sha256"],
            "repo_license_evidence": "pinned publisher per-row SPDX, independently matched to pinned task metadata",
            "repo_license_counts": dict(Counter(e["repo_license"] for e in report["accepted"])),
        },
    }
    for name, payload in summaries.items():
        path = temporary / "reports" / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        files[path.relative_to(temporary).as_posix()] = sha256_file(path)
    (temporary / "admission.json").write_bytes(admission_path.read_bytes())
    files["admission.json"] = sha256_file(temporary / "admission.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": config["release_id"],
        "status": "passed",
        "preview": False,
        "source_revision": source["revision"],
        "files": files,
        "episode_count": len(rows["episodes"]),
        "decision_count": len(rows["decisions"]),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "blob_root": str(blobs.root),
        "admission_sha256": files["admission.json"],
        "implementation_sha256": implementation_digest(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify_release(temporary)
    temporary.rename(destination)
    return destination


def verify_release(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    expected_keys = {
        "schema_version",
        "release_id",
        "status",
        "preview",
        "source_revision",
        "files",
        "episode_count",
        "decision_count",
        "split_counts",
        "blob_root",
        "admission_sha256",
        "implementation_sha256",
        "code_commit",
    }
    if (
        set(manifest) != expected_keys
        or manifest["status"] != "passed"
        or manifest["preview"] is not False
    ):
        raise ValueError("release is not admitted")
    required = {
        *(f"{name}.parquet" for name in TABLE_SCHEMAS),
        "admission.json",
        *(f"reports/{name}.json" for name in ("quality", "contamination", "licenses")),
        *(f"splits/{name}.txt" for name in ("train", "dev", "test")),
    }
    if set(manifest["files"]) != required:
        raise ValueError("release file inventory mismatch")
    for filename, digest in manifest["files"].items():
        if sha256_file(root / filename) != digest:
            raise ValueError("release file hash mismatch")
    episodes = pq.read_table(root / "episodes.parquet")
    decisions = pq.read_table(root / "decisions.parquet")
    for name, schema in TABLE_SCHEMAS.items():
        if not pq.read_schema(root / f"{name}.parquet").equals(schema, check_metadata=True):
            raise ValueError("release table schema mismatch")
    episode_ids = episodes["episode_id"].to_pylist()
    if len(set(episode_ids)) != len(episode_ids) or len(episode_ids) != manifest["episode_count"]:
        raise ValueError("episode identity/count mismatch")
    if len(decisions) != manifest["decision_count"] or not set(
        decisions["episode_id"].to_pylist()
    ) <= set(episode_ids):
        raise ValueError("decision referential integrity mismatch")
    seen = set()
    for split in ("train", "dev", "test"):
        ids = (root / "splits" / f"{split}.txt").read_text().splitlines()
        if (
            len(ids) != len(set(ids))
            or seen.intersection(ids)
            or len(ids) != manifest["split_counts"][split]
        ):
            raise ValueError("release splits overlap or counts mismatch")
        seen.update(ids)
    if seen != set(episode_ids):
        raise ValueError("release splits do not cover episodes")
    blobs = BlobStore(Path(manifest["blob_root"]))
    for ref in episodes["raw_trace_ref"].to_pylist():
        blobs.read_json(ref)
    return manifest
