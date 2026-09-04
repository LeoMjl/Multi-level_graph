from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlg.config import RuntimeConfig
from mlg.graph import TaskGraph


TASKGRAPH_CACHE_SCHEMA = "mlg-taskgraph-cache-v1"


@dataclass(frozen=True)
class TaskGraphCacheSpec:
    enabled: bool
    cache_id: str
    path: Path
    descriptor: dict[str, Any]


def build_taskgraph_cache_spec(
    runtime: RuntimeConfig,
    *,
    graph_kind: str,
    haystack_key: str,
) -> TaskGraphCacheSpec:
    descriptor = {
        "schema": TASKGRAPH_CACHE_SCHEMA,
        "graph_kind": graph_kind,
        "haystack_key": haystack_key,
    }
    encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    safe_kind = re.sub(r"[^a-zA-Z0-9_.-]+", "-", graph_kind).strip("-.") or "taskgraph"
    path = Path(runtime.graph_cache_dir).resolve() / f"{safe_kind}-{cache_id[:24]}.json"
    return TaskGraphCacheSpec(
        enabled=bool(runtime.graph_cache_enabled),
        cache_id=cache_id,
        path=path,
        descriptor=descriptor,
    )


def load_taskgraph_cache(
    spec: TaskGraphCacheSpec,
) -> tuple[TaskGraph | None, dict[str, Any]]:
    metadata = _status_metadata(spec, "disabled" if not spec.enabled else "miss", False)
    if not spec.enabled or not spec.path.is_file():
        return None, metadata
    try:
        stored = json.loads(spec.path.read_text(encoding="utf-8"))
        if stored.get("schema") != TASKGRAPH_CACHE_SCHEMA:
            raise ValueError("schema mismatch")
        if stored.get("descriptor") != spec.descriptor:
            raise ValueError("descriptor mismatch")
        graph_payload = stored.get("graph")
        if not isinstance(graph_payload, dict):
            raise ValueError("missing graph payload")
        graph = TaskGraph.from_dict(graph_payload)
        metadata = _status_metadata(spec, "hit", True)
        metadata["graph_cache_node_count"] = len(graph.nodes)
        metadata["graph_cache_edge_count"] = len(graph.edges)
        return graph, metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        metadata.update(
            {
                "graph_cache_status": "invalid",
                "graph_cache_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return None, metadata


def persist_taskgraph_cache(
    spec: TaskGraphCacheSpec,
    graph: TaskGraph,
) -> dict[str, Any]:
    if not spec.enabled:
        return _status_metadata(spec, "disabled", False)
    payload = {
        "schema": TASKGRAPH_CACHE_SCHEMA,
        "descriptor": spec.descriptor,
        "graph": graph.to_dict(),
    }
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = spec.path.with_name(f".{spec.path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, spec.path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    metadata = _status_metadata(spec, "built", False)
    metadata["graph_cache_node_count"] = len(graph.nodes)
    metadata["graph_cache_edge_count"] = len(graph.edges)
    return metadata


def _status_metadata(
    spec: TaskGraphCacheSpec,
    status: str,
    hit: bool,
) -> dict[str, Any]:
    return {
        "graph_cache_enabled": spec.enabled,
        "graph_cache_id": spec.cache_id,
        "graph_cache_path": str(spec.path),
        "graph_cache_status": status,
        "graph_cache_hit": hit,
        "graph_cache_descriptor": dict(spec.descriptor),
    }
