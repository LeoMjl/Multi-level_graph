from __future__ import annotations

from typing import Any

from mlg.graph import EdgeType, Node, NodeStatus, TaskGraph
from mlg.m5.taskgraph_config import PaperDependencyConfig


class PaperDependencyBuilder:
    """Instantiate source->target DEPENDENCY edges using Section 3.2."""

    method = "paper_vc_cdep_vr_model_judge_v4"

    def __init__(self, config: PaperDependencyConfig | None = None) -> None:
        self.config = config or PaperDependencyConfig()

    def candidates(
        self,
        graph: TaskGraph,
        target_id: str,
        target_text: str,
        target_vector: list[float],
        source_vectors: dict[str, list[float]],
    ) -> dict[str, Any]:
        target = graph.nodes[target_id]
        node_ids = list(graph.nodes)
        target_index = node_ids.index(target_id)
        historical = [
            graph.nodes[node_id] for node_id in node_ids[:target_index]
            if node_id in source_vectors
            and _eligible_source(graph.nodes[node_id], target)
        ]
        structural = {
            node.node_id for node in historical
            if 0 <= target.turn_index - node.turn_index <= self.config.structural_window
            and _shares_non_root_scope(node.path, target.path)
        }
        ref_active = {
            node.node_id for node in historical
            if int(node.metadata.get("ref_count", 0)) >= self.config.reference_threshold
            or node.status == NodeStatus.ACTIVE
            or node.metadata.get("state_status") == "active"
        }
        embedding_scores = {
            node.node_id: _cosine(target_vector, source_vectors[node.node_id])
            for node in historical
        }
        semantic_scores = dict(embedding_scores)
        insertion = {node_id: index for index, node_id in enumerate(graph.nodes)}
        stable = {
            node.node_id for node in historical if _same_state_key(node, target)
        }
        semantic_ranked = sorted(
            (
                (score, node_id) for node_id, score in semantic_scores.items()
                if score >= self.config.semantic_threshold
            ),
            key=lambda item: (-item[0], insertion[item[1]], item[1]),
        )[: self.config.semantic_top_k]
        semantic = {node_id for _, node_id in semantic_ranked} | stable
        final = (structural & ref_active) | semantic
        task_scores = {
            node_id: _task_relevance(graph.nodes[node_id], target, target_text)
            for node_id in final
        }
        temporal_scores = {
            node_id: _temporal_relevance(graph.nodes[node_id], target)
            for node_id in final
        }
        fused_scores = {
            node_id: _fused_score(
                semantic_scores.get(node_id, 0.0), task_scores[node_id],
                temporal_scores[node_id], node_id in stable,
            )
            for node_id in final
        }
        all_ranked = sorted(final, key=lambda node_id: (
            -fused_scores[node_id],
            -semantic_scores.get(node_id, 0.0),
            -task_scores[node_id],
            -temporal_scores[node_id],
            insertion[node_id],
            node_id,
        ))
        mandatory = [node_id for node_id in all_ranked if node_id in stable]
        ordinary = [node_id for node_id in all_ranked if node_id not in stable]
        ranked = mandatory + ordinary[:max(0, self.config.candidate_limit - len(mandatory))]
        details = []
        for source_id in ranked:
            channels = []
            if source_id in structural and source_id in ref_active:
                channels.append("Vc_intersection_Vr")
            if source_id in semantic:
                channels.append("Cdep")
            details.append({
                "source_id": source_id,
                "candidate_channels": channels,
                "semantic_score": round(semantic_scores[source_id], 6),
                "task_relevance": round(task_scores[source_id], 6),
                "temporal_relevance": round(temporal_scores[source_id], 6),
                "fused_score": round(fused_scores[source_id], 6),
                "stable_key_match": _same_state_key(graph.nodes[source_id], target),
            })

        audit = {
            "method": self.method,
            "target_id": target_id,
            "Vc": sorted(structural),
            "Cdep": [
                *sorted(stable, key=lambda node_id: insertion[node_id]),
                *(node_id for _, node_id in semantic_ranked if node_id not in stable),
            ],
            "Vr": sorted(ref_active),
            "Cfinal": ranked,
            "candidate_details": details,
            "created": [],
            "config": {
                "w": self.config.structural_window,
                "theta": self.config.semantic_threshold,
                "delta": self.config.reference_threshold,
                "semantic_top_k": self.config.semantic_top_k,
                "candidate_limit": self.config.candidate_limit,
            },
        }
        target.metadata["dependency_audit"] = _compact_audit(audit)
        return audit

    def connect_selected(
        self,
        graph: TaskGraph,
        target_id: str,
        audit: dict[str, Any],
        decisions: list[dict[str, Any]],
        *,
        selection_phase: str = "pre_write",
    ) -> dict[str, Any]:
        if selection_phase != "pre_write":
            raise RuntimeError(
                "Model-selected dependency edges are restricted to pre-write context"
            )
        if graph.nodes[target_id].level.value != "L3":
            raise RuntimeError(
                "Model-selected dependency edges must target the current planned L3"
            )
        target = graph.nodes[target_id]
        if (
            target.status != NodeStatus.ACTIVE
            or target.metadata.get("lifecycle_phase") != "planned_before_write"
        ):
            raise RuntimeError(
                "Model-selected dependency edges require an active planned-before-write L3"
            )
        if str(audit.get("target_id", "")) != target_id:
            raise RuntimeError("Dependency audit is not bound to the requested target")
        allowed = {str(item) for item in audit["Cfinal"]}
        details = {item["source_id"]: item for item in audit["candidate_details"]}
        created: list[dict[str, Any]] = []
        ordered = sorted(decisions, key=lambda item: (
            int(item.get("priority", 10**9)), str(item.get("source_id", "")),
        ))
        for decision in ordered:
            source_id = str(decision.get("source_id", ""))
            if source_id not in allowed or graph.nodes[source_id].level.value == "L1":
                continue
            detail = details[source_id]
            metadata = {
                "method": self.method,
                "dependency_type": str(decision.get("dependency_type", "contextual_relevance")),
                "confidence": _safe_confidence(decision.get("confidence", 1.0)),
                "priority": max(1, int(decision.get("priority", len(created) + 1))),
                "reason": str(decision.get("reason", "")).strip(),
                "candidate_channels": detail["candidate_channels"],
                "semantic_score": detail["semantic_score"],
                "task_relevance": detail["task_relevance"],
                "temporal_relevance": detail["temporal_relevance"],
                "fused_score": detail["fused_score"],
                "stable_key_match": detail["stable_key_match"],
                "selection_phase": selection_phase,
                "used_in_prompt": False,
            }
            if _add_unique_dependency(graph, source_id, target_id, metadata):
                created.append({"source_id": source_id, **metadata})
        audit["created"] = created
        audit["judge"] = "model_relevance"
        audit["selection_phase"] = selection_phase
        graph.nodes[target_id].metadata["dependency_audit"] = _compact_audit(audit)
        return audit


def _add_unique_dependency(
    graph: TaskGraph, source_id: str, target_id: str, metadata: dict[str, Any],
) -> bool:
    if any(
        edge.source_id == source_id
        and edge.target_id == target_id
        and edge.edge_type == EdgeType.DEPENDENCY
        for edge in graph.edges
    ):
        return False
    graph.add_edge(source_id, target_id, EdgeType.DEPENDENCY, metadata)
    return True


def _shares_non_root_scope(left: str, right: str) -> bool:
    left_parts = set(part for part in left.split("/")[1:] if part)
    right_parts = set(part for part in right.split("/")[1:] if part)
    return bool(left_parts & right_parts)


def _eligible_source(source: Node, target: Node) -> bool:
    """Return only already released, strictly historical L2-L4 nodes."""
    if source.level.value == "L1":
        return False
    if source.status in {NodeStatus.PENDING, NodeStatus.DROPPED}:
        return False
    if source.metadata.get("state_status") == "superseded":
        return False
    if source.turn_index > target.turn_index:
        return False

    source_volume = int(source.metadata.get("volume_id", 0) or 0)
    target_volume = int(target.metadata.get("volume_id", 0) or 0)
    if source.level.value == "L2" and target_volume and source_volume > target_volume:
        return False

    source_chapter = int(source.metadata.get("chapter_id", 0) or 0)
    target_chapter = int(target.metadata.get("chapter_id", 0) or 0)
    if source_chapter and target_chapter and source_chapter >= target_chapter:
        return False
    return True


def _temporal_relevance(source: Node, target: Node) -> float:
    gap = max(0, target.turn_index - source.turn_index)
    return 1.0 / (1.0 + gap)


def _fused_score(
    semantic: float, task: float, temporal: float, stable_key_match: bool,
) -> float:
    semantic = max(0.0, min(1.0, semantic))
    base = 0.50 * semantic + 0.30 * task + 0.20 * temporal
    return min(1.0, base + (0.50 if stable_key_match else 0.0))


def _compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Keep graph checkpoints small; full frozen rows live in judgment records."""
    return {
        "method": audit.get("method"),
        "target_phase": audit.get("target_phase", ""),
        "candidate_counts": {
            key: len(audit.get(key, [])) for key in ("Vc", "Cdep", "Vr", "Cfinal")
        },
        "created_source_ids": [
            str(row.get("source_id", "")) for row in audit.get("created", [])
        ],
        "selection_phase": audit.get("selection_phase", ""),
        "config": dict(audit.get("config", {})),
    }


def _relation(source: Node, target: Node, target_text: str, semantic_score: float) -> str | None:
    if _same_state_key(source, target):
        return "variable_reference"
    if _shared_entities(source, target):
        return "entity_reference"
    if _lexical_score(_node_text(source), target_text) > 0:
        return "constraint_reference"
    if semantic_score >= 0.5:
        return "semantic_support"
    return None


def _confidence(source: Node, target: Node, target_text: str, semantic_score: float) -> float:
    if _same_state_key(source, target):
        return 1.0
    entity_bonus = 0.95 if _shared_entities(source, target) else 0.0
    return max(semantic_score, entity_bonus, _lexical_score(_node_text(source), target_text))


def _task_relevance(source: Node, target: Node, target_text: str) -> float:
    if _same_state_key(source, target):
        return 1.0
    shared = _shared_entities(source, target)
    entity_total = len(
        set(source.metadata.get("entities", []))
        | set(target.metadata.get("entities", []))
    )
    entity_score = len(shared) / entity_total if entity_total else 0.0
    return max(entity_score, _lexical_score(_node_text(source), target_text))


def _shared_entities(left: Node, right: Node) -> set[str]:
    return set(left.metadata.get("entities", [])) & set(right.metadata.get("entities", []))


def _same_state_key(left: Node, right: Node) -> bool:
    left_key = str(left.metadata.get("state_key", ""))
    right_key = str(right.metadata.get("state_key", ""))
    return bool(left_key and left_key == right_key)


def node_text(node: Node) -> str:
    """Canonical node representation used by Encode(v), never a RAG query."""
    return _node_text(node)


def _node_text(node: Node) -> str:
    entities = "、".join(str(item) for item in node.metadata.get("entities", []))
    if node.level.value == "L3" and node.status == NodeStatus.DONE:
        result = str(node.metadata.get("result_summary", "")).strip()
        if result:
            return "\n".join(item for item in (node.level.value, result, entities) if item)
    return "\n".join(
        item for item in (node.level.value, node.content, node.value, entities) if item
    )


def _lexical_score(left: str, right: str) -> float:
    left_terms, right_terms = _terms(left), _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / min(len(left_terms), len(right_terms))


def _terms(text: str) -> set[str]:
    compact = "".join(ch for ch in text.lower() if not ch.isspace())
    return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch: target={len(left)}, source={len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _safe_confidence(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 6)
    except (TypeError, ValueError):
        return 1.0
