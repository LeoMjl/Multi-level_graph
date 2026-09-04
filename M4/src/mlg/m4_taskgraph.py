from __future__ import annotations

from pathlib import Path
from typing import Any

from mlg.m4_model import M4Model
from mlg.m4_schema_index import build_schema_index
from mlg.m4_schema_retrieve import SchemaGraphRetriever


def build_taskgraph_index(
    dataset: str,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    chunk_vectors: Any,
    model: M4Model,
    artifact_dir: Path,
) -> tuple[SchemaGraphRetriever, dict[str, Any]]:
    del chunks, chunk_vectors
    return build_schema_index(dataset, documents, model, artifact_dir)
