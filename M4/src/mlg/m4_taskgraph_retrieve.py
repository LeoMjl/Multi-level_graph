from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any

import numpy as np

from mlg.graph.task_graph import EdgeType, NodeLevel, TaskGraph
from mlg.m4_retrieval import normalized


POLICY = "taskgraph_temporal_relation_bundle_v9"
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_LEXICAL_STOP = {
    "a", "an", "and", "are", "article", "as", "at", "before", "between", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "in", "is", "of", "on", "or", "published",
    "report", "reports", "that", "the", "this", "to", "was", "were", "what", "while", "with",
}
EDGE_WEIGHTS = {
    EdgeType.INCLUSION: 0.85,
    EdgeType.MAINLINE: 0.70,
    EdgeType.DEPENDENCY: 0.75,
}


class TaskGraphRetriever:
    def __init__(self, graph: TaskGraph, records: list[dict[str, Any]], vectors: Any) -> None:
        self.graph = graph
        self.records = records
        self.vectors = normalized(vectors)
        self.position_by_node = {row["node_id"]: index for index, row in enumerate(records)}
        self.positions_by_doc: dict[str, list[int]] = defaultdict(list)
        self.positions_by_level: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.global_positions_by_level: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(records):
            doc_id, level = str(row["doc_id"]), str(row["level"])
            self.positions_by_doc[doc_id].append(index)
            self.positions_by_level[(doc_id, level)].append(index)
            self.global_positions_by_level[level].append(index)
        self.doc_node_position = {
            str(row["doc_id"]): index
            for index, row in enumerate(records) if row["level"] == NodeLevel.L2.value
        }
        self.document_metadata = {
            doc: self._document_metadata(records[positions[0]]["text"])
            for (doc, level), positions in self.positions_by_level.items()
            if level == NodeLevel.L4.value and positions
        }
        self.document_dates = {
            doc: self._parse_date(metadata.get("published_at", ""))
            for doc, metadata in self.document_metadata.items()
        }
        self.docs_by_source_alias: dict[str, list[str]] = defaultdict(list)
        for candidate_doc, metadata in self.document_metadata.items():
            source = str(metadata.get("source", "")).strip()
            for alias in self._source_aliases(source):
                self.docs_by_source_alias[alias].append(candidate_doc)
        self.document_terms = {
            doc: self._terms(" ".join(str(records[position]["text"]) for position in positions))
            for (doc, level), positions in self.positions_by_level.items()
            if level == NodeLevel.L4.value
        }
        document_frequency = Counter(term for terms in self.document_terms.values() for term in terms)
        total_documents = len(self.document_terms)
        self.term_idf = {
            term: math.log((total_documents + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.route_leaves: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(records):
            if row["level"] == NodeLevel.L4.value:
                self.route_leaves[str(row["node_id"])].append(index)
            elif row["level"] == NodeLevel.L2.value:
                self.route_leaves[str(row["node_id"])].extend(
                    self.positions_by_level[(str(row["doc_id"]), NodeLevel.L4.value)]
                )
        self.neighbors: dict[str, list[tuple[str, EdgeType, float, str, str, str]]] = defaultdict(list)
        for edge in graph.edges:
            weight = EDGE_WEIGHTS[edge.edge_type]
            if edge.edge_type == EdgeType.DEPENDENCY:
                strength = float(edge.metadata.get("strength", edge.metadata.get("cosine", 0.4)))
                weight *= max(0.4, strength)
            relation = str(edge.metadata.get("relation", edge.edge_type.value.lower()))
            label = ", ".join(str(item) for item in edge.metadata.get("entities", []))
            self.neighbors[edge.source_id].append(
                (edge.target_id, edge.edge_type, weight, relation, "forward", label)
            )
            self.neighbors[edge.target_id].append(
                (edge.source_id, edge.edge_type, weight, relation, "reverse", label)
            )
            source = self.position_by_node.get(edge.source_id)
            target = self.position_by_node.get(edge.target_id)
            if source is not None and target is not None:
                source_row, target_row = records[source], records[target]
                if (
                    edge.edge_type == EdgeType.INCLUSION
                    and source_row["level"] == NodeLevel.L3.value
                    and target_row["level"] == NodeLevel.L4.value
                ):
                    self.route_leaves[str(source_row["node_id"])].append(target)

    def retrieve(
        self, query_vector: Any, *, doc_id: str, k: int = 10, query_text: str = "",
    ) -> list[dict[str, Any]]:
        qvec = normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ qvec
        if k <= 0:
            return []
        operator = self._query_operator(query_text)
        query_terms = self._terms(query_text)
        direct_limit = min(k, max(4, 3 * k // 4))
        direct = self._top(scores, doc_id, NodeLevel.L4.value, direct_limit)
        seeds = list(dict.fromkeys(
            self._top(scores, doc_id, NodeLevel.L4.value, 4)
            + self._top(scores, doc_id, NodeLevel.L3.value, 2)
            + self._top(scores, doc_id, NodeLevel.L2.value, 2)
        ))
        direct_set = set(direct)
        direct_docs = {str(self.records[position]["doc_id"]) for position in direct}
        direct_cutoff = float(scores[direct[-1]]) if direct else -1.0
        expansion: dict[int, dict[str, Any]] = {}
        for seed_position in seeds:
            seed = self.records[seed_position]
            for neighbor_id, edge_type, edge_weight, relation, direction, label in self.neighbors.get(seed["node_id"], []):
                if relation in {"document_crossdoc", "topic_crossdoc"}:
                    continue
                if not doc_id and query_text:
                    continue
                if not doc_id and relation != "chunk_crossdoc":
                    continue
                if not doc_id and relation == "chunk_crossdoc" and operator != "inference":
                    continue
                target_position = self.position_by_node.get(neighbor_id)
                if target_position is None:
                    continue
                leaves = self.route_leaves.get(neighbor_id, [])
                ranked_leaves = sorted(leaves, key=lambda item: float(scores[item]), reverse=True)[:2]
                for position in ranked_leaves:
                    leaf = self.records[position]
                    if position in direct_set or (doc_id and leaf["doc_id"] != doc_id):
                        continue
                    semantic = float(scores[position])
                    if semantic < direct_cutoff - 0.04:
                        continue
                    target_score = max(0.0, float(scores[target_position]))
                    seed_score = max(0.0, float(scores[seed_position]))
                    graph_score = seed_score * edge_weight
                    final = (
                        semantic + 0.025 * seed_score + 0.015 * target_score
                        + 0.01 * edge_weight + self._operator_bonus(operator, edge_type, relation)
                    )
                    current = expansion.get(position)
                    if current is not None and final <= current["final_score"]:
                        continue
                    expansion[position] = {
                        "seed_node_id": seed["node_id"],
                        "edge_type": edge_type.value,
                        "routed_via_node_id": neighbor_id,
                        "route_relation": relation,
                        "route_label": label,
                        "route_direction": direction,
                        "query_operator": operator,
                        "semantic_score": semantic,
                        "graph_score": graph_score,
                        "final_score": final,
                    }
        source_route_docs: set[str] = set()
        if query_text and not doc_id:
            mentioned_sources = self._mentioned_sources(query_text)
            for source_alias in mentioned_sources:
                routed_docs = self._source_route_documents(
                    source_alias, scores, operator, query_text,
                    allow_pair=len(mentioned_sources) == 1,
                )
                is_temporal_pair = operator == "temporal" and len(routed_docs) == 2
                pair_dates = [
                    self.document_dates.get(candidate_doc) for candidate_doc in routed_docs
                ]
                bundle_id = (
                    f"temporal:{source_alias}:"
                    + ":".join(value.isoformat() if value else "unknown" for value in pair_dates)
                    if is_temporal_pair else ""
                )
                for route_index, candidate_doc in enumerate(routed_docs):
                    source_route_docs.add(candidate_doc)
                    leaves = self.positions_by_level[(candidate_doc, NodeLevel.L4.value)]
                    position = max(leaves, key=lambda item: float(scores[item]))
                    semantic = float(scores[position])
                    doc_position = self.doc_node_position[candidate_doc]
                    doc_score = max(0.0, float(scores[doc_position]))
                    final = semantic + (0.12 if is_temporal_pair else 0.10) + 0.02 * doc_score
                    current = expansion.get(position)
                    if current is not None and final <= current["final_score"]:
                        continue
                    metadata = self.document_metadata[candidate_doc]
                    expansion[position] = {
                        "seed_node_id": self.records[doc_position]["node_id"],
                        "edge_type": (
                            EdgeType.MAINLINE.value if is_temporal_pair else EdgeType.INCLUSION.value
                        ),
                        "routed_via_node_id": self.records[doc_position]["node_id"],
                        "route_relation": (
                            "query_temporal_source_pair" if is_temporal_pair
                            else "query_source_inclusion"
                        ),
                        "route_label": metadata.get("source", source_alias),
                        "route_direction": "forward",
                        "route_source": metadata.get("source", ""),
                        "route_date": metadata.get("published_at", ""),
                        "route_bundle_id": bundle_id,
                        "route_bundle_role": (
                            ("earlier" if route_index == 0 else "later") if is_temporal_pair else ""
                        ),
                        "route_pair_dates": " -> ".join(
                            value.isoformat() if value else "unknown" for value in pair_dates
                        ) if is_temporal_pair else "",
                        "route_path_relation": "source_temporal_next" if is_temporal_pair else "inclusion",
                        "query_operator": operator,
                        "semantic_score": semantic,
                        "graph_score": 1.0,
                        "route_source_score": 1.0,
                        "route_document_score": doc_score,
                        "final_score": final,
                    }

            denominator = sum(self.term_idf.get(term, 1.0) for term in query_terms) or 1.0
            document_routes = []
            for candidate_doc, terms in self.document_terms.items():
                if source_route_docs:
                    continue
                if candidate_doc in direct_docs or candidate_doc in source_route_docs:
                    continue
                overlap = query_terms & terms
                lexical = sum(self.term_idf.get(term, 1.0) for term in overlap) / denominator
                if not lexical:
                    continue
                doc_position = self.doc_node_position[candidate_doc]
                doc_score = max(0.0, float(scores[doc_position]))
                document_routes.append((lexical, candidate_doc, overlap, doc_score))
            for lexical, candidate_doc, overlap, doc_score in sorted(document_routes, reverse=True)[:10]:
                doc_position = self.doc_node_position[candidate_doc]
                leaves = self.positions_by_level[(candidate_doc, NodeLevel.L4.value)]
                for position in sorted(leaves, key=lambda item: float(scores[item]), reverse=True)[:1]:
                    if position in direct_set:
                        continue
                    semantic = float(scores[position])
                    if semantic < direct_cutoff - 0.08:
                        continue
                    final = semantic + 0.12 * lexical + 0.015 * doc_score
                    current = expansion.get(position)
                    if current is not None and final <= current["final_score"]:
                        continue
                    labels = sorted(overlap, key=lambda term: self.term_idf.get(term, 1.0), reverse=True)[:5]
                    expansion[position] = {
                        "seed_node_id": self.records[doc_position]["node_id"],
                        "edge_type": EdgeType.INCLUSION.value,
                        "routed_via_node_id": self.records[doc_position]["node_id"],
                        "route_relation": "query_document_inclusion",
                        "route_label": ", ".join(labels),
                        "route_direction": "forward",
                        "query_operator": operator,
                        "semantic_score": semantic,
                        "graph_score": lexical,
                        "route_lexical_score": lexical,
                        "route_document_score": doc_score,
                        "final_score": final,
                    }
        direct_rows = [
            self._result(position, float(scores[position]), "direct", {"query_operator": operator})
            for position in direct
        ]
        edge_rows = [
            self._result(position, expansion[position]["final_score"], "edge_routed", expansion[position])
            for position in sorted(expansion, key=lambda item: expansion[item]["final_score"], reverse=True)
        ]
        ranked = sorted(direct_rows + edge_rows, key=lambda row: float(row["final_score"]), reverse=True)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        seen_text: set[str] = set()
        per_doc: dict[str, int] = defaultdict(int)

        def add(row: dict[str, Any]) -> bool:
            text_key = " ".join(str(row["text"]).split())
            candidate_doc = str(row["doc_id"])
            if row["chunk_id"] in selected_ids or text_key in seen_text:
                return False
            if not doc_id and per_doc[candidate_doc] >= 3:
                return False
            selected.append(row)
            selected_ids.add(str(row["chunk_id"]))
            seen_text.add(text_key)
            per_doc[candidate_doc] += 1
            return True

        for row in ranked:
            add(row)
            if len(selected) == k:
                break
        return selected

    def audit(self, rows: list[list[dict[str, Any]]], query_vectors: Any, doc_ids: list[str]) -> dict[str, Any]:
        origins: dict[str, int] = defaultdict(int)
        edges: dict[str, int] = defaultdict(int)
        relations: dict[str, int] = defaultdict(int)
        levels: dict[str, int] = defaultdict(int)
        bundles: dict[tuple[int, str], set[str]] = defaultdict(set)
        changed = edge_queries = edge_selected = 0
        overlaps = []
        for query_index, (retrieval, query_vector, doc_id) in enumerate(zip(rows, query_vectors, doc_ids)):
            selected_ids = {row["chunk_id"] for row in retrieval}
            dense_ids = set(self.raw_dense_ids(query_vector, doc_id, len(retrieval)))
            overlaps.append(len(selected_ids & dense_ids) / max(1, len(selected_ids | dense_ids)))
            changed += int(selected_ids != dense_ids)
            has_edge = False
            for row in retrieval:
                origins[str(row["origin"])] += 1
                levels[str(row["level"])] += 1
                if row["origin"] == "edge_routed":
                    has_edge = True
                    edge_selected += 1
                    edges[str(row["edge_type"])] += 1
                    relations[str(row.get("route_relation", ""))] += 1
                    if row.get("route_bundle_id"):
                        bundles[(query_index, str(row["route_bundle_id"]))].add(
                            str(row.get("route_bundle_role", ""))
                        )
            edge_queries += int(has_edge)
        nonempty = bool(rows) and all(rows)
        leaf_only = all(row.get("level") == NodeLevel.L4.value for retrieval in rows for row in retrieval)
        incomplete_bundles = sum(roles != {"earlier", "later"} for roles in bundles.values())
        return {
            "policy": POLICY,
            "queries": len(rows),
            "queries_changed_vs_raw_dense": changed,
            "queries_with_edge_expansion": edge_queries,
            "queries_with_edge_routing": edge_queries,
            "edge_selected_count": edge_selected,
            "mean_jaccard_vs_raw_dense": float(np.mean(overlaps)) if overlaps else 0.0,
            "origin_counts": dict(origins),
            "selected_edge_type_counts": dict(edges),
            "selected_route_relation_counts": dict(relations),
            "selected_level_counts": dict(levels),
            "temporal_bundle_count": len(bundles),
            "incomplete_temporal_bundle_count": incomplete_bundles,
            "all_selected_nodes_are_l4": leaf_only,
            "passed": nonempty and leaf_only and edge_selected > 0 and changed > 0 and incomplete_bundles == 0,
        }

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", str(text).casefold())
            if len(token) > 1 and token not in _LEXICAL_STOP
        }

    @staticmethod
    def _normalized_phrase(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))

    @staticmethod
    def _document_metadata(text: str) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for line in str(text).splitlines()[:8]:
            match = re.match(r"^(title|source|published_at):\s*(.+?)\s*$", line, re.IGNORECASE)
            if match:
                metadata[match.group(1).casefold()] = match.group(2)
        return metadata

    @staticmethod
    def _parse_date(value: str) -> date | None:
        match = re.search(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)", str(value))
        if match is None:
            return None
        try:
            return date(*(int(part) for part in match.groups()))
        except ValueError:
            return None

    @classmethod
    def _query_dates(cls, query_text: str) -> list[date]:
        found: list[tuple[int, date]] = []
        occupied: list[tuple[int, int]] = []
        for match in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", str(query_text)):
            try:
                found.append((match.start(), date(*(int(part) for part in match.groups()))))
                occupied.append(match.span())
            except ValueError:
                continue
        named = list(re.finditer(
            rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b",
            str(query_text), re.IGNORECASE,
        ))
        known_years = [int(match.group(3)) for match in named if match.group(3)]
        inherited_year = known_years[0] if len(set(known_years)) == 1 else 0
        for match in named:
            year = int(match.group(3)) if match.group(3) else inherited_year
            if not year or any(left <= match.start() < right for left, right in occupied):
                continue
            try:
                found.append((match.start(), date(year, _MONTHS[match.group(1).casefold()], int(match.group(2)))))
            except ValueError:
                continue
        unique: list[date] = []
        for _, value in sorted(found):
            if value not in unique:
                unique.append(value)
        return unique

    @staticmethod
    def _temporal_pair_requested(query_text: str) -> bool:
        padded = f" {str(query_text).casefold()} "
        markers = (
            " between ", " change", " consistent", " earlier", " later", " before ",
            " after ", "both reports", "two reports", "from the report", "compared with",
        )
        return any(marker in padded for marker in markers)

    def _best_leaf_score(self, candidate_doc: str, scores: np.ndarray) -> float:
        return max(
            float(scores[position])
            for position in self.positions_by_level[(candidate_doc, NodeLevel.L4.value)]
        )

    def _source_route_documents(
        self, source_alias: str, scores: np.ndarray, operator: str,
        query_text: str, *, allow_pair: bool,
    ) -> list[str]:
        candidates = list(dict.fromkeys(self.docs_by_source_alias[source_alias]))
        semantic = {candidate: self._best_leaf_score(candidate, scores) for candidate in candidates}
        query_dates = self._query_dates(query_text)
        if operator == "temporal" and allow_pair and len(candidates) > 1:
            selected: list[str] = []
            endpoints = query_dates[:1] + query_dates[-1:] if len(query_dates) > 1 else []
            for endpoint in endpoints:
                dated = [candidate for candidate in candidates if self.document_dates.get(candidate)]
                if not dated:
                    break
                choice = min(
                    (candidate for candidate in dated if candidate not in selected),
                    key=lambda candidate: (
                        abs((self.document_dates[candidate] - endpoint).days), -semantic[candidate],
                    ),
                    default="",
                )
                if choice:
                    selected.append(choice)
            if len(selected) == 2:
                return sorted(selected, key=lambda candidate: self.document_dates[candidate] or date.min)
            if self._temporal_pair_requested(query_text):
                ranked = sorted(candidates, key=lambda candidate: semantic[candidate], reverse=True)
                first = ranked[0]
                second = next(
                    (candidate for candidate in ranked[1:]
                     if self.document_dates.get(candidate) != self.document_dates.get(first)), "",
                )
                if second:
                    return sorted([first, second], key=lambda candidate: self.document_dates[candidate] or date.min)
        if operator == "temporal" and query_dates:
            def route_score(candidate: str) -> float:
                published = self.document_dates.get(candidate)
                proximity = 0.0 if published is None else 1.0 / (1.0 + min(
                    abs((published - endpoint).days) for endpoint in query_dates
                ) / 30.0)
                return semantic[candidate] + 0.08 * proximity
            return [max(candidates, key=route_score)]
        return [max(candidates, key=lambda candidate: semantic[candidate])]

    @classmethod
    def _source_aliases(cls, source: str) -> set[str]:
        names = {str(source)}
        for separator in ("|", " - "):
            names.add(str(source).split(separator, 1)[0])
        aliases = {cls._normalized_phrase(name) for name in names}
        aliases.update(alias[4:] for alias in list(aliases) if alias.startswith("the "))
        return {alias for alias in aliases if len(alias) >= 3}

    def _mentioned_sources(self, query_text: str) -> list[str]:
        padded_query = f" {self._normalized_phrase(query_text)} "
        mentioned = [
            alias for alias in self.docs_by_source_alias
            if f" {alias} " in padded_query
        ]
        selected: list[str] = []
        for alias in sorted(mentioned, key=lambda item: (-len(item.split()), item)):
            if any(f" {alias} " in f" {longer} " for longer in selected):
                continue
            selected.append(alias)
        return selected

    @staticmethod
    def _query_operator(query_text: str) -> str:
        text = str(query_text).casefold()
        temporal = (" before ", " after ", " between ", "published", "earlier", "later", "change", "consistent")
        comparison = (" while ", "compared", "both", "align", "differ", "same", "respectively")
        padded = f" {text} "
        if any(term in padded for term in temporal):
            return "temporal"
        if any(term in padded for term in comparison):
            return "comparison"
        return "inference"

    @staticmethod
    def _operator_bonus(operator: str, edge_type: EdgeType, relation: str) -> float:
        if operator == "temporal" and edge_type == EdgeType.MAINLINE:
            return 0.015
        if operator == "comparison" and "crossdoc" in relation:
            return 0.015
        if operator == "inference" and edge_type == EdgeType.DEPENDENCY:
            return 0.01
        return 0.0

    def raw_dense_ids(self, query_vector: Any, doc_id: str, k: int) -> list[str]:
        qvec = normalized(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ qvec
        positions = self._ranked(scores, doc_id, NodeLevel.L4.value)
        return [str(self.records[position]["retrieval_id"]) for position in positions[:k]]

    def _top(self, scores: np.ndarray, doc_id: str, level: str, count: int) -> list[int]:
        return self._ranked(scores, doc_id, level)[:count]

    def _ranked(self, scores: np.ndarray, doc_id: str, level: str = "") -> list[int]:
        if doc_id:
            positions = self.positions_by_level[(doc_id, level)] if level else self.positions_by_doc[doc_id]
        else:
            positions = self.global_positions_by_level[level] if level else list(range(len(self.records)))
        ranked = sorted(positions, key=lambda item: float(scores[item]), reverse=True)
        if doc_id:
            return ranked
        output = []
        per_doc: dict[str, int] = defaultdict(int)
        for position in ranked:
            candidate_doc = str(self.records[position]["doc_id"])
            if per_doc[candidate_doc] >= 3:
                continue
            output.append(position)
            per_doc[candidate_doc] += 1
        return output

    def _result(
        self, position: int, score: float, origin: str, provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = self.records[position]
        result = {
            "chunk_id": row["retrieval_id"], "doc_id": row["doc_id"], "node_id": row["node_id"],
            "level": row["level"], "score": score, "policy": POLICY, "origin": origin,
            "text": row["text"],
        }
        result.update(provenance or {
            "seed_node_id": row["node_id"], "edge_type": "", "semantic_score": score,
            "graph_score": 0.0, "final_score": score,
        })
        result.setdefault("seed_node_id", row["node_id"])
        result.setdefault("edge_type", "")
        result.setdefault("routed_via_node_id", "")
        result.setdefault("route_relation", "")
        result.setdefault("route_label", "")
        result.setdefault("route_direction", "")
        result.setdefault("route_bundle_id", "")
        result.setdefault("route_bundle_role", "")
        result.setdefault("route_pair_dates", "")
        result.setdefault("route_path_relation", "")
        result.setdefault("query_operator", "inference")
        result.setdefault("semantic_score", score)
        result.setdefault("graph_score", 0.0)
        result.setdefault("final_score", score)
        return result
