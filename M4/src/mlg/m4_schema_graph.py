from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np

from mlg.graph.task_graph import EdgeType, NodeLevel, NodeStatus, TaskGraph
from mlg.m4_retrieval import normalized


def _date_key(value: str) -> str:
    match = re.search(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", str(value))
    return "" if match is None else "-".join(match.groups())


def assemble_schema_graph(
    dataset: str,
    units: list[dict[str, Any]],
    unit_vectors: Any,
    events: list[dict[str, Any]],
    event_vectors: Any,
    schemas: list[dict[str, Any]],
    schema_vectors: Any,
    relations: list[dict[str, Any]],
    *,
    relation_scope: str,
    documents: list[dict[str, Any]] | None = None,
    document_summaries: dict[str, str] | None = None,
    document_vectors: Any | None = None,
) -> tuple[TaskGraph, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    graph = TaskGraph()
    unit_array = normalized(unit_vectors)
    event_array = normalized(event_vectors)
    schema_array = normalized(schema_vectors)
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    document_nodes: dict[str, str] = {}
    schema_nodes: dict[int, str] = {}
    event_nodes: dict[int, str] = {}
    unit_nodes: dict[int, str] = {}

    if dataset == "quality":
        if not documents or document_summaries is None or document_vectors is None:
            raise ValueError("QuALITY narrative graph requires document overviews and vectors")
        document_array = normalized(document_vectors)
        for document_index, document in enumerate(documents):
            doc_id = str(document["doc_id"])
            summary = str(document_summaries[doc_id])
            node = graph.add_node(
                NodeLevel.L1, summary, document_index, f"{dataset}/{doc_id}",
                hint="Document", status=NodeStatus.ACTIVE,
                metadata={
                    "retrieval_id": f"taskgraph:{doc_id}:document",
                    "node_kind": "document_overview", "doc_id": doc_id,
                    "source": str(document.get("source", "")),
                    "published_at": str(document.get("published_at", "")),
                    "query_independent": True,
                },
            )
            document_nodes[doc_id] = node
            records.append({
                "node_id": node, "retrieval_id": f"taskgraph:{doc_id}:document",
                "doc_id": doc_id, "level": NodeLevel.L1.value,
                "node_kind": "document_overview", "text": summary, "order": document_index,
                "source": str(document.get("source", "")),
                "published_at": str(document.get("published_at", "")),
            })
            vectors.append(document_array[document_index])
        root = ""
    else:
        root = graph.add_node(
            NodeLevel.L1, f"{dataset} query-independent evidence corpus", 0, dataset,
            hint="Corpus", status=NodeStatus.ACTIVE,
            metadata={"dataset": dataset, "query_independent": True, "relation_scope": relation_scope},
        )

    for schema_index, schema in enumerate(schemas):
        schema_doc_id = str(schema["doc_ids"][0]) if len(schema["doc_ids"]) == 1 else ""
        node = graph.add_node(
            NodeLevel.L2, str(schema["text"]), schema_index,
            f"{dataset}/schema/{schema_index}", hint="Schema",
            metadata={
                "retrieval_id": schema["schema_id"], "node_kind": "event_schema",
                "doc_id": schema_doc_id, "doc_ids": schema["doc_ids"],
                "anchors": schema["anchors"], "scene_index": schema.get("scene_index"),
            },
        )
        schema_nodes[schema_index] = node
        parent = document_nodes.get(schema_doc_id, root)
        graph.add_edge(parent, node, EdgeType.INCLUSION, {
            "relation": "document_scene" if dataset == "quality" else "schema_membership",
        })
        records.append({
            "node_id": node, "retrieval_id": schema["schema_id"], "doc_id": schema_doc_id,
            "level": NodeLevel.L2.value,
            "node_kind": "narrative_scene" if dataset == "quality" else "event_schema",
            "text": schema["text"], "order": int(schema.get("scene_index", schema_index)),
            "schema_id": schema["schema_id"],
        })
        vectors.append(schema_array[schema_index])

        for event_global in schema["event_indices"]:
            event = events[event_global]
            event_node = graph.add_node(
                NodeLevel.L3, str(event["text"]), int(event["event_index"]),
                f"{dataset}/{event['doc_id']}/event/{event['event_index']}",
                hint="Event", metadata={
                    "retrieval_id": event["event_id"], "node_kind": "report_event",
                    "doc_id": event["doc_id"], "source": event["source"],
                    "published_at": event["published_at"],
                },
            )
            event_nodes[event_global] = event_node
            graph.add_edge(node, event_node, EdgeType.INCLUSION, {"relation": "schema_event"})
            records.append({
                "node_id": event_node, "retrieval_id": event["event_id"],
                "doc_id": event["doc_id"], "level": NodeLevel.L3.value,
                "node_kind": "report_event", "text": event["text"],
                "order": int(event["event_index"]), "source": event["source"],
                "published_at": event["published_at"], "schema_id": schema["schema_id"],
            })
            vectors.append(event_array[event_global])

            previous_unit = ""
            for unit_global in event["unit_indices"]:
                unit = units[unit_global]
                unit_node = graph.add_node(
                    NodeLevel.L4, str(unit["fact_text"]), int(unit["unit_index"]),
                    f"{dataset}/{unit['doc_id']}/fact/{unit['unit_index']}",
                    hint="Fact", metadata={
                        "retrieval_id": unit["unit_id"], "node_kind": "atomic_evidence",
                        "doc_id": unit["doc_id"], "source": unit["source"],
                        "published_at": unit["published_at"],
                        "body_char_start": unit["body_char_start"],
                        "body_char_end": unit["body_char_end"],
                    },
                )
                unit_nodes[unit_global] = unit_node
                graph.add_edge(event_node, unit_node, EdgeType.INCLUSION, {"relation": "event_evidence"})
                if previous_unit:
                    graph.add_edge(previous_unit, unit_node, EdgeType.MAINLINE, {"relation": "evidence_next"})
                previous_unit = unit_node
                records.append({
                    "node_id": unit_node, "retrieval_id": unit["unit_id"],
                    "doc_id": unit["doc_id"], "level": NodeLevel.L4.value,
                    "node_kind": "atomic_evidence",
                    "text": unit["fact_text"] if dataset == "quality" else unit["text"],
                    "order": int(unit["unit_index"]), "source": unit["source"],
                    "published_at": unit["published_at"], "schema_id": schema["schema_id"],
                })
                vectors.append(unit_array[unit_global])

    if dataset == "quality":
        schemas_by_doc: dict[str, list[int]] = defaultdict(list)
        for index, schema in enumerate(schemas):
            schemas_by_doc[str(schema["doc_ids"][0])].append(index)
        for indices in schemas_by_doc.values():
            indices.sort(key=lambda index: int(schemas[index].get("scene_index", index)))
            for left, right in zip(indices, indices[1:]):
                graph.add_edge(schema_nodes[left], schema_nodes[right], EdgeType.MAINLINE, {
                    "relation": "narrative_scene_next",
                })

    by_doc: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_doc[str(event["doc_id"])].append(index)
    for indices in by_doc.values():
        indices.sort(key=lambda index: int(events[index]["event_index"]))
        for left, right in zip(indices, indices[1:]):
            graph.add_edge(event_nodes[left], event_nodes[right], EdgeType.MAINLINE, {
                "relation": "document_event_next",
            })

    for relation in relations:
        left, right = int(relation["left"]), int(relation["right"])
        graph.add_edge(event_nodes[left], event_nodes[right], EdgeType.DEPENDENCY, {
            key: value for key, value in relation.items() if key not in {"left", "right"}
        })

    for schema in schemas:
        indices = sorted(
            schema["event_indices"],
            key=lambda index: (_date_key(events[index]["published_at"]), events[index]["doc_id"]),
        )
        if len({events[index]["doc_id"] for index in indices}) < 2:
            continue
        for left, right in zip(indices, indices[1:]):
            if events[left]["doc_id"] == events[right]["doc_id"]:
                continue
            graph.add_edge(event_nodes[left], event_nodes[right], EdgeType.MAINLINE, {
                "relation": "event_temporal_next",
                "left_date": _date_key(events[left]["published_at"]),
                "right_date": _date_key(events[right]["published_at"]),
                "time_basis": "published_at",
            })

    edge_types: dict[str, int] = defaultdict(int)
    edge_relations: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        edge_types[edge.edge_type.value] += 1
        edge_relations[str(edge.metadata.get("relation", edge.edge_type.value.casefold()))] += 1
    return graph, records, normalized(np.vstack(vectors)), {
        "dataset": dataset,
        "relation_scope": relation_scope,
        "retrieval_nodes": len(records),
        "level_counts": {
            **({"L1": len(documents or [])} if dataset == "quality" else {}),
            "L2": len(schemas), "L3": len(events), "L4": len(units),
        },
        "edge_type_counts": dict(edge_types),
        "edge_relation_counts": dict(edge_relations),
        "query_independent": True,
        "source_temporal_edges": edge_relations.get("source_temporal_next", 0),
        "virtual_relation_edges": 0,
    }
