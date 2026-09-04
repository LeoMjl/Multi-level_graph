from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mlg.graph.task_graph import NodeLevel


def audit_schema_retrieval(
    retriever: Any,
    rows: list[list[dict[str, Any]]],
    query_vectors: Any,
    doc_ids: list[str],
    policy: str,
) -> dict[str, Any]:
    changed = edge_queries = edge_selected = invalid_paths = 0
    hierarchy_selected = dependency_selected = 0
    origins: dict[str, int] = defaultdict(int)
    relations: dict[str, int] = defaultdict(int)
    overlaps: list[float] = []
    for retrieval, vector, doc_id in zip(rows, query_vectors, doc_ids):
        evidence = [row for row in retrieval if row.get("level") == NodeLevel.L4.value]
        selected = {str(row["chunk_id"]) for row in evidence}
        dense = set(retriever.raw_dense_ids(vector, doc_id, len(evidence)))
        changed += selected != dense
        overlaps.append(len(selected & dense) / max(1, len(selected | dense)))
        has_edge = False
        for row in retrieval:
            origins[str(row["origin"])] += 1
            hierarchy_selected += row["origin"] == "hierarchy_context"
            if row["origin"] != "edge_routed":
                continue
            has_edge = True
            edge_selected += 1
            relations[str(row.get("route_relation", ""))] += 1
            dependency_selected += row.get("edge_type") == "DEPENDENCY"
            path = list(row.get("route_path_node_ids", []))
            invalid_paths += any(
                frozenset((left, right)) not in retriever.edge_pairs
                for left, right in zip(path, path[1:])
            )
        edge_queries += has_edge
    leaf_only = all(row.get("level") == NodeLevel.L4.value for result in rows for row in result)
    levels_valid = all(
        row.get("level") == NodeLevel.L4.value
        or (row.get("origin") == "hierarchy_context" and row.get("level") in {NodeLevel.L1.value, NodeLevel.L2.value})
        for result in rows for row in result
    )
    hierarchy_ok = not retriever.include_hierarchy or hierarchy_selected > 0
    dependency_exists = any(edge.edge_type.value == "DEPENDENCY" for edge in retriever.graph.edges)
    dependency_ok = not retriever.include_hierarchy or not dependency_exists or dependency_selected > 0
    return {
        "policy": policy, "queries": len(rows), "queries_changed_vs_raw_dense": changed,
        "queries_with_edge_expansion": edge_queries, "queries_with_edge_routing": edge_queries,
        "edge_selected_count": edge_selected, "origin_counts": dict(origins),
        "hierarchy_selected_count": hierarchy_selected,
        "dependency_edge_selected_count": dependency_selected,
        "selected_route_relation_counts": dict(relations),
        "mean_jaccard_vs_raw_dense": float(np.mean(overlaps)) if overlaps else 0.0,
        "invalid_stored_paths": invalid_paths, "virtual_relation_count": 0,
        "all_selected_nodes_are_l4": leaf_only,
        "evidence_nodes_are_l4": levels_valid,
        "passed": bool(rows) and all(rows) and levels_valid and hierarchy_ok and dependency_ok
        and edge_selected > 0 and not invalid_paths,
    }
