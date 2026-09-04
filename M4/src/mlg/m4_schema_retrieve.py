from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mlg.graph.task_graph import EdgeType, NodeLevel, TaskGraph
from mlg.m4_retrieval import normalized
from mlg.m4_schema_query import metadata_bonuses, query_operator, relation_allowed, relation_bonus
from mlg.m4_schema_retrieve_audit import audit_schema_retrieval


POLICY = "taskgraph_query_independent_schema_v12"
_EDGE_WEIGHT = {
    EdgeType.INCLUSION: 0.55,
    EdgeType.MAINLINE: 0.65,
    EdgeType.DEPENDENCY: 0.95,
}


class SchemaGraphRetriever:
    def __init__(
        self,
        graph: TaskGraph,
        records: list[dict[str, Any]],
        vectors: Any,
        *,
        per_doc_limit: int | None = 4,
        direct_limit: int | None = None,
        include_hierarchy: bool = False,
        policy: str = POLICY,
    ) -> None:
        self.graph = graph
        self.records = records
        self.vectors = normalized(vectors)
        self.per_doc_limit = per_doc_limit
        self.direct_limit = direct_limit
        self.include_hierarchy = include_hierarchy
        self.policy = policy
        self.position_by_node = {str(row["node_id"]): index for index, row in enumerate(records)}
        self.by_level: dict[str, list[int]] = defaultdict(list)
        self.by_doc_level: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(records):
            self.by_level[str(row["level"])].append(index)
            self.by_doc_level[(str(row["doc_id"]), str(row["level"]))].append(index)
        self.children: dict[str, list[str]] = defaultdict(list)
        self.parents: dict[str, list[str]] = defaultdict(list)
        self.neighbors: dict[str, list[tuple[str, EdgeType, str, float, str, str]]] = defaultdict(list)
        self.edge_pairs: set[frozenset[str]] = set()
        for edge in graph.edges:
            relation = str(edge.metadata.get("relation", edge.edge_type.value.casefold()))
            strength = float(edge.metadata.get("strength", edge.metadata.get("cosine", 1.0)))
            label = ", ".join(str(value) for value in (
                edge.metadata.get("entities", []) or edge.metadata.get("anchors", [])
            ))
            self.neighbors[edge.source_id].append(
                (edge.target_id, edge.edge_type, relation, strength, "forward", label)
            )
            self.neighbors[edge.target_id].append(
                (edge.source_id, edge.edge_type, relation, strength, "reverse", label)
            )
            self.edge_pairs.add(frozenset((edge.source_id, edge.target_id)))
            if edge.edge_type == EdgeType.INCLUSION:
                self.children[edge.source_id].append(edge.target_id)
                self.parents[edge.target_id].append(edge.source_id)
        self._leaf_cache: dict[str, list[int]] = {}

    def _ancestor(self, node_id: str, level: str) -> int | None:
        frontier = list(self.parents.get(node_id, []))
        seen: set[str] = set()
        while frontier:
            candidate = frontier.pop(0)
            if candidate in seen:
                continue
            seen.add(candidate)
            position = self.position_by_node.get(candidate)
            if position is not None and self.records[position]["level"] == level:
                return position
            frontier.extend(self.parents.get(candidate, []))
        return None

    def _leaves(self, node_id: str) -> list[int]:
        if node_id in self._leaf_cache:
            return self._leaf_cache[node_id]
        position = self.position_by_node.get(node_id)
        if position is not None and self.records[position]["level"] == NodeLevel.L4.value:
            result = [position]
        else:
            result = []
            for child in self.children.get(node_id, []):
                result.extend(self._leaves(child))
        self._leaf_cache[node_id] = list(dict.fromkeys(result))
        return self._leaf_cache[node_id]

    def retrieve(
        self, query_vector: Any, *, doc_id: str, k: int = 10, query_text: str = "",
    ) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        qvec = normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        semantic_scores = self.vectors @ qvec
        operator = query_operator(query_text)
        bonuses = metadata_bonuses(self.records, query_text, operator)
        scores = semantic_scores + bonuses
        default_direct_count = max(8, 3 * k // 4)
        direct_count = min(k, self.direct_limit or default_direct_count)
        direct = self._ranked(scores, NodeLevel.L4.value, doc_id)[:direct_count]
        seeds = list(dict.fromkeys(
            self._ranked(scores, NodeLevel.L4.value, doc_id)[:8]
            + self._ranked(scores, NodeLevel.L3.value, doc_id)[:6]
            + self._ranked(scores, NodeLevel.L2.value, doc_id)[:4]
            + self._ranked(scores, NodeLevel.L1.value, doc_id)[:2]
        ))
        direct_set = set(direct)
        cutoff = float(scores[direct[-1]]) if direct else -1.0
        expansions: dict[int, dict[str, Any]] = {}
        for seed_position in seeds:
            seed = self.records[seed_position]
            seed_score = max(0.0, float(scores[seed_position]))
            frontier = [(str(seed["node_id"]), [str(seed["node_id"])], [], 1.0, "")]
            for depth in range(2):
                following = []
                for node_id, path, relations, path_weight, path_label in frontier:
                    for target, edge_type, relation, strength, direction, label in self.neighbors.get(node_id, []):
                        if target in path or target not in self.position_by_node:
                            continue
                        if not relation_allowed(operator, relation):
                            continue
                        weight = path_weight * _EDGE_WEIGHT[edge_type] * max(0.45, strength)
                        next_path = path + [target]
                        next_relations = relations + [relation]
                        leaves = sorted(self._leaves(target), key=lambda item: float(scores[item]), reverse=True)
                        if not self.include_hierarchy:
                            leaves = leaves[:2]
                        for position in leaves:
                            if position in direct_set or (doc_id and self.records[position]["doc_id"] != doc_id):
                                continue
                            semantic = float(scores[position])
                            if not self.include_hierarchy and semantic < cutoff - 0.18:
                                continue
                            bonus = relation_bonus(operator, next_relations)
                            graph_scale = 0.08 if self.include_hierarchy else 0.02
                            final = semantic + 0.01 * seed_score + graph_scale * weight + bonus
                            current = expansions.get(position)
                            if current is not None and final <= current["final_score"]:
                                continue
                            row = self.records[position]
                            expansions[position] = {
                                "seed_node_id": seed["node_id"],
                                "edge_type": edge_type.value,
                                "routed_via_node_id": target,
                                "route_relation": relation,
                                "route_label": label or path_label,
                                "route_direction": direction,
                                "route_path_node_ids": next_path,
                                "route_path_relation": " -> ".join(next_relations),
                                "query_operator": operator,
                                "semantic_score": semantic,
                                "graph_score": weight,
                                "final_score": final,
                                "route_source": row.get("source", ""),
                                "route_date": row.get("published_at", ""),
                            }
                        if depth == 0:
                            following.append((target, next_path, next_relations, weight, label or path_label))
                frontier = following

        rows = [self._result(position, float(scores[position]), "direct", operator, {
            "semantic_score": float(semantic_scores[position]),
            "metadata_score": float(bonuses[position]),
            "final_score": float(scores[position]),
        }) for position in direct]
        rows.extend(
            self._result(position, data["final_score"], "edge_routed", operator, data)
            for position, data in expansions.items()
        )
        rows.sort(key=lambda row: float(row["final_score"]), reverse=True)
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        per_doc: dict[str, int] = defaultdict(int)
        for row in rows:
            candidate_doc = str(row["doc_id"])
            text_key = " ".join(str(row["text"]).split())
            if (
                row["chunk_id"] in seen
                or text_key in seen
                or (self.per_doc_limit is not None and per_doc[candidate_doc] >= self.per_doc_limit)
            ):
                continue
            selected.append(row)
            seen.update((str(row["chunk_id"]), text_key))
            per_doc[candidate_doc] += 1
            if len(selected) == k:
                break
        if not self.include_hierarchy or not selected:
            return selected

        schema_positions: list[int] = []
        for row in selected:
            schema_position = self._ancestor(str(row["node_id"]), NodeLevel.L2.value)
            if schema_position is not None and schema_position not in schema_positions:
                schema_positions.append(schema_position)
        if not schema_positions:
            schema_positions = self._ranked(scores, NodeLevel.L2.value, doc_id)[:2]
        schema_positions.sort(key=lambda position: float(scores[position]), reverse=True)
        hierarchy: list[dict[str, Any]] = []
        document_positions = self._ranked(scores, NodeLevel.L1.value, doc_id)[:1]
        hierarchy.extend(
            self._result(position, float(scores[position]), "hierarchy_context", operator)
            for position in document_positions
        )
        hierarchy.extend(
            self._result(position, float(scores[position]), "hierarchy_context", operator)
            for position in schema_positions
        )
        schema_rank = {
            str(self.records[position]["retrieval_id"]): rank
            for rank, position in enumerate(schema_positions)
        }
        selected.sort(key=lambda row: (
            schema_rank.get(str(row.get("schema_id", "")), len(schema_rank)),
            int(row.get("order", 0)),
        ))
        return hierarchy + selected

    def audit(self, rows: list[list[dict[str, Any]]], query_vectors: Any, doc_ids: list[str]) -> dict[str, Any]:
        return audit_schema_retrieval(self, rows, query_vectors, doc_ids, self.policy)

    def raw_dense_ids(self, query_vector: Any, doc_id: str, k: int) -> list[str]:
        qvec = normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ qvec
        return [str(self.records[index]["retrieval_id"]) for index in self._ranked(scores, "L4", doc_id)[:k]]

    def _ranked(self, scores: np.ndarray, level: str, doc_id: str) -> list[int]:
        positions = self.by_doc_level[(doc_id, level)] if doc_id else self.by_level[level]
        return sorted(positions, key=lambda index: float(scores[index]), reverse=True)

    def _result(self, position: int, score: float, origin: str, operator: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        row = self.records[position]
        result = {
            "chunk_id": row["retrieval_id"], "doc_id": row["doc_id"], "node_id": row["node_id"],
            "level": row["level"], "score": score, "policy": self.policy, "origin": origin,
            "text": row["text"], "seed_node_id": row["node_id"], "edge_type": "",
            "route_relation": "", "route_direction": "", "route_label": "",
            "route_path_relation": "", "route_path_node_ids": [], "query_operator": operator,
            "semantic_score": score, "graph_score": 0.0, "final_score": score,
            "node_kind": row.get("node_kind", ""), "order": row.get("order", 0),
            "schema_id": row.get("schema_id", ""), "source": row.get("source", ""),
            "published_at": row.get("published_at", ""),
        }
        result.update(data or {})
        return result
