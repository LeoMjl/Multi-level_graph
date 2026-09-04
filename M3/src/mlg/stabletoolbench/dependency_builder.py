from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from mlg.graph import EdgeType, Node, NodeLevel, NodeStatus, TaskGraph


@dataclass(frozen=True)
class DependencyConfig:
    """Deterministic candidate policy for progressive TaskGraph expansion."""

    window_turns: int = 8
    semantic_threshold: float = 0.18
    reference_threshold: int = 2
    max_candidates: int = 16
    min_shared_path_depth: int = 1
    candidate_levels: tuple[NodeLevel, ...] = (
        NodeLevel.L2,
        NodeLevel.L3,
        NodeLevel.L4,
    )
    excluded_subtypes: tuple[str, ...] = (
        "Thought",
        "ToolCall",
        "ToolArgument",
        "ToolCapability",
    )

    def __post_init__(self) -> None:
        if self.window_turns < 0 or self.reference_threshold < 0:
            raise ValueError("window and reference thresholds must be non-negative")
        if self.max_candidates < 0 or self.min_shared_path_depth < 0:
            raise ValueError("candidate limits must be non-negative")
        if not -1.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [-1, 1]")


@dataclass(frozen=True)
class CandidateDiagnostic:
    node_id: str
    hard: bool
    structural: bool
    semantic: bool
    referenced_or_active: bool
    semantic_score: float
    ref_count: int
    turn_distance: int
    selected: bool

    @property
    def sources(self) -> tuple[str, ...]:
        sources = []
        if self.hard:
            sources.append("hard")
        if self.structural:
            sources.append("Vc")
        if self.semantic:
            sources.append("Cdep")
        if self.referenced_or_active:
            sources.append("Vr")
        return tuple(sources)


@dataclass(frozen=True)
class DependencyCandidates:
    target_id: str
    hard_dependencies: tuple[str, ...]
    structural_candidates: tuple[str, ...]
    semantic_candidates: tuple[str, ...]
    reference_candidates: tuple[str, ...]
    fusion_pool: tuple[str, ...]
    fused_candidates: tuple[str, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]
    semantic_backend: str

    def diagnostic(self, node_id: str) -> CandidateDiagnostic:
        return next(item for item in self.diagnostics if item.node_id == node_id)


@dataclass(frozen=True)
class DependencyState:
    source_id: str
    status: str
    blocking: bool
    hard: bool
    outcome: str


@dataclass(frozen=True)
class DependencyReadiness:
    target_id: str
    ready: bool
    satisfied: tuple[str, ...]
    waiting: tuple[str, ...]
    blocked: tuple[str, ...]
    informational: tuple[str, ...]
    details: tuple[DependencyState, ...]


def build_dependency_candidates(
    graph: TaskGraph,
    target_id: str,
    *,
    hard_dependency_ids: tuple[str, ...] | list[str] = (),
    semantic_scores: Mapping[str, float] | None = None,
    config: DependencyConfig | None = None,
) -> DependencyCandidates:
    """Build ``hard union ((Vc intersect Vr) union Cdep)`` candidates.

    ``semantic_scores`` accepts scores from any separately configured encoder.
    When absent, a deterministic sparse token cosine is used; it requires no
    model or download and is recorded in the returned diagnostics.
    """

    policy = config or DependencyConfig()
    target = _require_node(graph, target_id)
    hard = _validated_hard(graph, target_id, hard_dependency_ids)
    history = _eligible_history(graph, target_id, policy)
    provided = semantic_scores is not None
    scores = {
        node.node_id: _safe_score(semantic_scores.get(node.node_id, 0.0))
        if provided else _sparse_cosine(node, target)
        for node in history
    }
    structural = {
        node.node_id for node in history
        if target.turn_index - node.turn_index <= policy.window_turns
        and _paths_overlap(node.path, target.path, policy.min_shared_path_depth)
    }
    semantic = {
        node.node_id for node in history
        if scores[node.node_id] >= policy.semantic_threshold
    }
    ref_counts = {node.node_id: _reference_count(graph, node.node_id) for node in history}
    referenced = {
        node.node_id for node in history
        if ref_counts[node.node_id] >= policy.reference_threshold
        or node.status == NodeStatus.ACTIVE
    }
    fused = (structural & referenced) | semantic
    insertion = {node_id: index for index, node_id in enumerate(graph.nodes)}
    ranked = sorted(
        fused - set(hard),
        key=lambda node_id: (
            -scores.get(node_id, 0.0),
            -ref_counts.get(node_id, 0),
            -graph.nodes[node_id].turn_index,
            insertion[node_id],
            node_id,
        ),
    )[: policy.max_candidates]
    selected = tuple(hard) + tuple(ranked)
    selected_set = set(selected)
    diagnostic_ids = list(dict.fromkeys([*hard, *(node.node_id for node in history)]))
    diagnostics = tuple(
        CandidateDiagnostic(
            node_id=node_id,
            hard=node_id in hard,
            structural=node_id in structural,
            semantic=node_id in semantic,
            referenced_or_active=node_id in referenced,
            semantic_score=scores.get(node_id, 0.0),
            ref_count=ref_counts.get(node_id, _reference_count(graph, node_id)),
            turn_distance=max(0, target.turn_index - graph.nodes[node_id].turn_index),
            selected=node_id in selected_set,
        )
        for node_id in diagnostic_ids
    )
    return DependencyCandidates(
        target_id=target_id,
        hard_dependencies=hard,
        structural_candidates=_graph_order(graph, structural),
        semantic_candidates=_graph_order(graph, semantic),
        reference_candidates=_graph_order(graph, referenced),
        fusion_pool=tuple(hard) + _graph_order(graph, fused - set(hard)),
        fused_candidates=selected,
        diagnostics=diagnostics,
        semantic_backend="provided" if provided else "deterministic_sparse_cosine_v1",
    )


def candidate_edge_metadata(
    candidate: CandidateDiagnostic,
    *,
    relation_type: str = "support",
    confidence: float | None = None,
    blocking: bool | None = None,
) -> dict[str, object]:
    """Create auditable metadata after a caller confirms a candidate relation."""

    metadata: dict[str, object] = {
        "reason": "progressive_dependency",
        "relation_type": relation_type,
        "sources": list(candidate.sources),
        "hard": candidate.hard,
        "blocking": candidate.hard if blocking is None else bool(blocking),
        "semantic_score": candidate.semantic_score,
        "ref_count": candidate.ref_count,
    }
    if confidence is not None:
        metadata["confidence"] = _safe_score(confidence)
    return metadata


def evaluate_dependency_readiness(
    graph: TaskGraph,
    target_id: str,
    dependency_ids: tuple[str, ...] | list[str] | None = None,
) -> DependencyReadiness:
    """Evaluate blocking prerequisites without treating support edges as gates."""

    _require_node(graph, target_id)
    specifications: dict[str, tuple[bool, bool]] = {}
    if dependency_ids is not None:
        for source_id in dict.fromkeys(dependency_ids):
            specifications[source_id] = (True, True)
    else:
        for edge in graph.edges:
            if edge.target_id != target_id or edge.edge_type != EdgeType.DEPENDENCY:
                continue
            source = graph.nodes.get(edge.source_id)
            hard = bool(edge.metadata.get("hard", False))
            default_blocking = hard or bool(
                source and source.level in {NodeLevel.L2, NodeLevel.L3}
            )
            blocking = hard or bool(edge.metadata.get("blocking", default_blocking))
            prior = specifications.get(edge.source_id, (False, False))
            specifications[edge.source_id] = (prior[0] or hard, prior[1] or blocking)

    details = []
    satisfied: list[str] = []
    waiting: list[str] = []
    blocked: list[str] = []
    informational: list[str] = []
    for source_id, (hard, blocking) in specifications.items():
        source = graph.nodes.get(source_id)
        if source is None:
            outcome = "blocked"
            blocked.append(source_id)
            status = "Missing"
        elif _is_available(source):
            outcome = "satisfied"
            satisfied.append(source_id)
            status = source.status.value
        elif not blocking:
            outcome = "informational"
            informational.append(source_id)
            status = source.status.value
        elif source.status == NodeStatus.DROPPED:
            outcome = "blocked"
            blocked.append(source_id)
            status = source.status.value
        else:
            outcome = "waiting"
            waiting.append(source_id)
            status = source.status.value
        details.append(DependencyState(source_id, status, blocking, hard, outcome))
    return DependencyReadiness(
        target_id=target_id,
        ready=not waiting and not blocked,
        satisfied=tuple(satisfied),
        waiting=tuple(waiting),
        blocked=tuple(blocked),
        informational=tuple(informational),
        details=tuple(details),
    )


def _require_node(graph: TaskGraph, node_id: str) -> Node:
    try:
        return graph.nodes[node_id]
    except KeyError as exc:
        raise KeyError(f"Unknown TaskGraph node: {node_id}") from exc


def _validated_hard(
    graph: TaskGraph,
    target_id: str,
    dependency_ids: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    hard = tuple(dict.fromkeys(str(item) for item in dependency_ids))
    for source_id in hard:
        _require_node(graph, source_id)
        if source_id == target_id:
            raise ValueError("A node cannot be its own hard dependency")
    return hard


def _eligible_history(
    graph: TaskGraph,
    target_id: str,
    policy: DependencyConfig,
) -> list[Node]:
    node_ids = list(graph.nodes)
    target_index = node_ids.index(target_id)
    target = graph.nodes[target_id]
    return [
        graph.nodes[node_id]
        for node_id in node_ids[:target_index]
        if graph.nodes[node_id].level in policy.candidate_levels
        and graph.nodes[node_id].sub_type not in policy.excluded_subtypes
        and graph.nodes[node_id].status != NodeStatus.DROPPED
        and graph.nodes[node_id].turn_index <= target.turn_index
    ]


def _paths_overlap(left: str, right: str, minimum_depth: int) -> bool:
    if minimum_depth == 0:
        return True
    left_parts = tuple(item for item in re.split(r"[./\\]+", left) if item)
    right_parts = tuple(item for item in re.split(r"[./\\]+", right) if item)
    shared = 0
    for left_item, right_item in zip(left_parts, right_parts):
        if left_item != right_item:
            break
        shared += 1
    return shared >= minimum_depth


def _reference_count(graph: TaskGraph, node_id: str) -> int:
    explicit = graph.nodes[node_id].metadata.get("ref_count", 0)
    try:
        explicit_count = max(0, int(explicit))
    except (TypeError, ValueError):
        explicit_count = 0
    edge_count = sum(
        edge.source_id == node_id and edge.edge_type == EdgeType.DEPENDENCY
        for edge in graph.edges
    )
    return max(explicit_count, edge_count)


def _node_terms(node: Node) -> Counter[str]:
    text = f"{node.content} {node.value}".lower()
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.extend(token)
            terms.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            terms.append(token)
    return Counter(terms)


def _sparse_cosine(left: Node, right: Node) -> float:
    left_terms, right_terms = _node_terms(left), _node_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    common = left_terms.keys() & right_terms.keys()
    numerator = sum(left_terms[key] * right_terms[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left_terms.values()))
    right_norm = math.sqrt(sum(value * value for value in right_terms.values()))
    return numerator / (left_norm * right_norm)


def _safe_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _graph_order(graph: TaskGraph, node_ids: set[str]) -> tuple[str, ...]:
    return tuple(node_id for node_id in graph.nodes if node_id in node_ids)


def _is_available(node: Node) -> bool:
    return node.status == NodeStatus.DONE or (
        node.level == NodeLevel.L4
        and node.status == NodeStatus.ACTIVE
        and bool(node.value)
    )
