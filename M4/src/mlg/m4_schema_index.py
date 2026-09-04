from __future__ import annotations

from pathlib import Path
from typing import Any

from mlg.m4_model import M4Model
from mlg.m4_schema_cluster import build_schema_specs
from mlg.m4_schema_graph import assemble_schema_graph
from mlg.m4_schema_retrieve import SchemaGraphRetriever
from mlg.m4_schema_summary import QUALITY_NARRATIVE_PROTOCOL, summarize_quality_hierarchy
from mlg.m4_schema_units import SCHEMA_PROTOCOL, build_atomic_units, build_report_events


def build_schema_index(
    dataset: str,
    documents: list[dict[str, Any]],
    model: M4Model,
    artifact_dir: Path,
) -> tuple[SchemaGraphRetriever, dict[str, Any]]:
    if dataset not in {"quality", "multihop_rag"}:
        raise ValueError(f"Unsupported schema dataset: {dataset}")
    relation_scope = "within_document" if dataset == "quality" else "cross_document"
    protocol = QUALITY_NARRATIVE_PROTOCOL if dataset == "quality" else SCHEMA_PROTOCOL
    units = build_atomic_units(documents)
    unit_vectors, unit_vector_meta = model.cached_embeddings(
        [str(unit["text"]) for unit in units],
        [str(unit["unit_id"]) for unit in units],
        dataset=dataset,
        kind=f"taskgraph_atomic_evidence_{protocol}",
    )
    events, event_vectors = build_report_events(units, unit_vectors)
    schemas, schema_vectors, relations, cluster_meta = build_schema_specs(
        events, event_vectors, relation_scope=relation_scope,
    )
    document_summaries = None
    document_vectors = None
    hierarchy_meta: dict[str, Any] = {}
    if dataset == "quality":
        schemas, schema_vectors, document_summaries, document_vectors, hierarchy_meta = summarize_quality_hierarchy(
            documents, events, schemas, model, artifact_dir,
        )
    graph, records, node_vectors, graph_meta = assemble_schema_graph(
        dataset, units, unit_vectors, events, event_vectors, schemas, schema_vectors, relations,
        relation_scope=relation_scope, documents=documents,
        document_summaries=document_summaries, document_vectors=document_vectors,
    )
    metadata = {
        "protocol": protocol,
        "dataset": dataset,
        "schema_scope": relation_scope,
        "construction_input": "documents_only",
        "questions_used_for_construction": 0,
        "labels_used_for_construction": 0,
        "atomic_units": len(units),
        "report_events": len(events),
        "atomic_vector_cache": unit_vector_meta,
        **hierarchy_meta,
        **cluster_meta,
        **graph_meta,
    }
    retriever = SchemaGraphRetriever(
        graph,
        records,
        node_vectors,
        per_doc_limit=None if dataset == "quality" else 4,
        direct_limit=6 if dataset == "quality" else None,
        include_hierarchy=dataset == "quality",
        policy=("taskgraph_quality_narrative_budget_v13" if dataset == "quality"
                else f"taskgraph_unified_schema_{relation_scope}_v12"),
    )
    return retriever, metadata


def build_multihop_schema_index(
    documents: list[dict[str, Any]],
    model: M4Model,
    artifact_dir: Path,
) -> tuple[SchemaGraphRetriever, dict[str, Any]]:
    return build_schema_index("multihop_rag", documents, model, artifact_dir)
