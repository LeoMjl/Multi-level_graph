from __future__ import annotations

from typing import Any

from mlg.graph import TaskGraph
from mlg.m5.rag_memory import _pack_vectors, _unpack_vectors
from mlg.m5.taskgraph_config import PaperDependencyConfig
from mlg.m5.taskgraph_dependency import PaperDependencyBuilder


def dump_taskgraph_memory(memory: Any, base: dict[str, Any]) -> dict[str, Any]:
    vector_ids = list(memory.node_vectors)
    return {
        **base,
        "schema": memory.state_schema,
        "graph": memory.graph.to_dict(),
        "root_id": memory.root_id,
        "volume_by_id": memory.volume_by_id,
        "chapter_by_id": memory.chapter_by_id,
        "fact_by_key": memory.fact_by_key,
        "vector_node_ids": vector_ids,
        "vectors": _pack_vectors([memory.node_vectors[node_id] for node_id in vector_ids]),
        "embedding_model": memory.embeddings.model,
        "total_volumes": memory.total_volumes,
        "chapters_per_volume": memory.chapters_per_volume,
        "volume_briefs": memory.volume_briefs,
        "dependency_config": memory.dependency_builder.config.__dict__,
    }


def load_taskgraph_memory(memory: Any, state: dict[str, Any]) -> None:
    if state.get("schema") != memory.state_schema:
        raise RuntimeError("Legacy M5 TaskGraph state is incompatible; start a new run directory")
    if state.get("embedding_model") != memory.embeddings.model:
        raise RuntimeError("M5 TaskGraph embedding model changed inside a resumable run")
    memory.total_volumes = int(state["total_volumes"])
    memory.chapters_per_volume = int(state["chapters_per_volume"])
    memory.volume_briefs = {
        int(key): str(value) for key, value in state.get("volume_briefs", {}).items()
    } or memory.volume_briefs
    memory.dependency_builder = PaperDependencyBuilder(
        PaperDependencyConfig(**state["dependency_config"]),
    )
    memory.graph = TaskGraph.from_dict(state["graph"])
    memory.root_id = str(state["root_id"])
    memory.volume_by_id = {int(k): str(v) for k, v in state["volume_by_id"].items()}
    memory.chapter_by_id = {int(k): str(v) for k, v in state["chapter_by_id"].items()}
    memory.fact_by_key = {str(k): str(v) for k, v in state["fact_by_key"].items()}
    vector_ids = list(state["vector_node_ids"])
    vectors = _unpack_vectors(state["vectors"])
    if len(vector_ids) != len(vectors):
        raise RuntimeError("M5 TaskGraph vector checkpoint is incomplete")
    memory.node_vectors = dict(zip(vector_ids, vectors))
