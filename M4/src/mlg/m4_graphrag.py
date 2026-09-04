from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mlg.config import PROJECT_ROOT
from mlg.m4_data import M4_DIR
from mlg.m4_model import M4Model
from mlg.m4_retrieval import read_jsonl


GRAPHRAG_ROOT = Path(os.getenv("GRAPHRAG_ROOT", PROJECT_ROOT / "third_party" / "graphrag"))
GRAPHRAG_COMMIT = "96a2460375fe579f4714e837dc581829ca459fd2"
GRAPHRAG_PYTHON = PROJECT_ROOT / ".venv-graphrag" / "Scripts" / "python.exe"
TOP_K = 10
CONTEXT_TOKENS = 8192


def build_graphrag(dataset: str, artifact_dir: Path, model_config: Path) -> dict[str, Any]:
    if not GRAPHRAG_PYTHON.exists():
        raise RuntimeError("GraphRAG Python 3.12 environment is missing")
    command = [
        str(GRAPHRAG_PYTHON), str(PROJECT_ROOT / "scripts" / "graphrag_worker.py"),
        "--dataset", dataset, "--data-dir", str(M4_DIR), "--output", str(artifact_dir),
        "--model-config", str(model_config),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"GraphRAG worker exited with code {completed.returncode}")
    roots = list((artifact_dir / dataset).glob("*/complete.json"))
    empty = list((artifact_dir / dataset).glob("*/empty.json"))
    expected = 115 if dataset == "quality" else 1
    if len(roots) + len(empty) != expected:
        raise RuntimeError(f"GraphRAG {dataset} has {len(roots) + len(empty)}/{expected} terminal indexes")
    return {
        "indexes": len(roots), "empty_indexes": len(empty), "path": str(artifact_dir / dataset),
        "official_commit": GRAPHRAG_COMMIT, "python": str(GRAPHRAG_PYTHON),
        "compatibility_correction": (
            "resolved chunk overlap forced to zero; DeepSeek thinking disabled; "
            "embedding responses requested as float; document metadata retained; "
            "gleaning disabled after source-grounding audit; official query levels 0-2 retained; "
            "zero-edge document graphs recorded as empty retrieval indexes"
        ),
    }


def retrieve_graphrag(
    dataset: str, queries: list[dict], query_vectors: Any, artifact_dir: Path, model: M4Model,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    del model
    vectors = np.asarray(query_vectors, dtype=np.float32)
    if len(queries) != len(vectors):
        raise ValueError("GraphRAG query and vector counts differ")
    base = artifact_dir / dataset
    manifest_path = base / "local_query_manifest.jsonl"
    vectors_path = base / "local_query_vectors.npy"
    output_path = base / "local_search_contexts.jsonl"
    meta_path = base / "local_search_contexts.meta.json"
    protocol = "official_graphrag_0.3.5_local_search_v1"
    payload = [
        {"query_id": row["query_id"], "question": row["question"], "doc_id": row.get("doc_id", "")}
        for row in queries
    ]
    fingerprint = hashlib.sha256(
        (protocol + json.dumps(payload, ensure_ascii=False, sort_keys=True)).encode("utf-8")
        + vectors.tobytes()
    ).hexdigest()
    cached = False
    if output_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cached = meta.get("fingerprint") == fingerprint and meta.get("rows") == len(queries)
    if not cached:
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in payload),
            encoding="utf-8",
        )
        np.save(vectors_path, vectors)
        command = [
            str(GRAPHRAG_PYTHON), str(PROJECT_ROOT / "scripts" / "graphrag_local_context.py"),
            "--dataset", dataset, "--artifact-dir", str(artifact_dir),
            "--queries", str(manifest_path), "--vectors", str(vectors_path),
            "--output", str(output_path),
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"GraphRAG local context worker exited with code {completed.returncode}")
        rows = read_jsonl(output_path)
        if len(rows) != len(queries):
            raise RuntimeError(f"GraphRAG local context output has {len(rows)}/{len(queries)} rows")
        meta_path.write_text(
            json.dumps({"protocol": protocol, "fingerprint": fingerprint, "rows": len(rows)}, indent=2) + "\n",
            encoding="utf-8",
        )
    rows = read_jsonl(output_path)
    by_id = {row["query_id"]: row["retrieval"] for row in rows}
    expected_ids = {row["query_id"] for row in queries}
    if set(by_id) != expected_ids:
        raise RuntimeError("GraphRAG local context query IDs do not match")
    return [by_id[row["query_id"]] for row in queries], {
        "contexts": len(rows), "protocol": protocol, "cache_hit": cached,
        "official_commit": GRAPHRAG_COMMIT,
    }
