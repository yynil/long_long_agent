"""Hash-only, complete-pool task and trace contamination index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import yaml

from .governance import group_digest, load_heldout_policy, sha256_file
from .quality import minhash_bands, task_shingles, text_digest
from .source_io import iter_rows, load_sources, task_identity_and_text, verified_inventory
from .util import canonical_json


def build_index(config: Path, heldout: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError("refusing to overwrite contamination index")
    sources, root = load_sources(config)
    policy = load_heldout_policy(heldout)
    expected = {
        x["source_id"]: x["episode_count"] for x in yaml.safe_load(heldout.read_text())["sources"]
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(destination)
    db.executescript("""
        CREATE TABLE tasks (digest TEXT PRIMARY KEY, shingles TEXT NOT NULL);
        CREATE TABLE task_groups (digest TEXT, group_hash TEXT, split TEXT,
                                  PRIMARY KEY(digest, group_hash));
        CREATE TABLE bands (band INTEGER, value TEXT, digest TEXT,
                            PRIMARY KEY(band, value, digest));
        CREATE TABLE traces (digest TEXT, split TEXT, PRIMARY KEY(digest, split));
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    summary = {
        "schema_version": 1,
        "status": "building",
        "sources": [],
        "heldout_sha256": policy.manifest_sha256,
        "near_algorithm": "task_5word_minhash32_bands8_exact_jaccard_v1",
        "near_jaccard_threshold": 0.8,
        "missing_task_text": 0,
    }
    known: set[str] = set()
    try:
        for source in sources["sources"]:
            policy.verify_source(source["source_id"], source["revision"])
            raw, files = verified_inventory(source, root)
            count = 0
            for _, _, row in iter_rows(raw, files):
                identity, task = task_identity_and_text(source["adapter"], row)
                split = policy.split_for_identity(identity)
                digest = text_digest(task)
                if not task.strip():
                    summary["missing_task_text"] += 1
                else:
                    if digest not in known:
                        shingles = task_shingles(task)
                        db.execute(
                            "INSERT INTO tasks VALUES (?,?)",
                            (digest, canonical_json(sorted(shingles))),
                        )
                        db.executemany(
                            "INSERT INTO bands VALUES (?,?,?)",
                            [(i, band, digest) for i, band in enumerate(minhash_bands(shingles))],
                        )
                        known.add(digest)
                    db.execute(
                        "INSERT OR IGNORE INTO task_groups VALUES (?,?,?)",
                        (digest, group_digest(identity, policy.salt), split),
                    )
                messages = row.get("messages") or row.get("conversations") or row.get("trajectory")
                trace = hashlib.sha256(canonical_json(messages).encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO traces VALUES (?,?)", (trace, split))
                count += 1
                if count % 5000 == 0:
                    db.commit()
                    print(
                        json.dumps({"source": source["source_id"], "indexed_rows": count}),
                        flush=True,
                    )
            if count != expected[source["source_id"]]:
                raise ValueError("contamination source row count differs from frozen inventory")
            summary["sources"].append(
                {
                    "source_id": source["source_id"],
                    "revision": source["revision"],
                    "episodes": count,
                    "files": files,
                }
            )
            db.commit()
        summary["status"] = "complete"
        summary["task_fingerprints"] = len(known)
        summary["exact_cross_split_tasks"] = db.execute(
            "SELECT count(*) FROM (SELECT digest FROM task_groups GROUP BY digest HAVING count(DISTINCT split)>1)"
        ).fetchone()[0]
        db.execute("INSERT INTO metadata VALUES ('summary', ?)", (canonical_json(summary),))
        db.commit()
    finally:
        db.close()
    summary["index_sha256"] = sha256_file(destination)
    destination.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class ContaminationIndex:
    def __init__(self, path: Path, heldout_sha256: str):
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        if manifest["status"] != "complete" or sha256_file(path) != manifest["index_sha256"]:
            raise ValueError("incomplete or modified contamination index")
        if manifest["heldout_sha256"] != heldout_sha256:
            raise ValueError("contamination index held-out mismatch")
        self.manifest = manifest
        self.connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        self._cache: dict[tuple[str, str], tuple[str, ...]] = {}

    def findings(self, task: str, raw_messages, split: str) -> tuple[str, ...]:
        trace = hashlib.sha256(canonical_json(raw_messages).encode()).hexdigest()
        trace_splits = {
            r[0]
            for r in self.connection.execute("SELECT split FROM traces WHERE digest=?", (trace,))
        }
        findings = {"cross_split_exact_trace"} if trace_splits - {split} else set()
        digest = text_digest(task)
        key = (digest, split)
        if key not in self._cache:
            task_findings = set()
            shingles = task_shingles(task)
            candidates = {digest}
            for index, band in enumerate(minhash_bands(shingles)):
                candidates.update(
                    r[0]
                    for r in self.connection.execute(
                        "SELECT digest FROM bands WHERE band=? AND value=?", (index, band)
                    )
                )
            for candidate in candidates:
                splits = {
                    r[0]
                    for r in self.connection.execute(
                        "SELECT split FROM task_groups WHERE digest=?", (candidate,)
                    )
                }
                if not splits - {split}:
                    continue
                if candidate == digest:
                    task_findings.add("cross_split_exact_task")
                else:
                    other = set(
                        json.loads(
                            self.connection.execute(
                                "SELECT shingles FROM tasks WHERE digest=?", (candidate,)
                            ).fetchone()[0]
                        )
                    )
                    if len(shingles & other) / len(shingles | other) >= 0.8:
                        task_findings.add("cross_split_near_task")
            self._cache[key] = tuple(sorted(task_findings))
        return tuple(sorted(findings | set(self._cache[key])))

    def close(self) -> None:
        self.connection.close()
