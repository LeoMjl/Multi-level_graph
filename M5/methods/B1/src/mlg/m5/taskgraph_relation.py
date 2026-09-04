from __future__ import annotations

import json
import math
from typing import Any

from mlg.graph import TaskGraph
from mlg.m5.backend import GenerationResult, TextBackend
from mlg.m5.memory import parse_json_object
from mlg.m5.taskgraph_dependency import node_text


_DEPENDENCY_TYPES = {
    "state_continuity",
    "entity_continuity",
    "constraint",
    "causal_support",
    "long_range_support",
    "plot_support",
    "contextual_relevance",
}


def judge_dependencies(
    graph: TaskGraph,
    targets: list[tuple[str, dict[str, Any]]],
    backend: TextBackend,
) -> tuple[dict[str, list[dict[str, Any]]], GenerationResult]:
    """Ask the model to retain every substantively relevant L2-L4 candidate."""
    payload = dependency_request_payload(graph, targets)
    result = backend.generate(
        dependency_judge_system_prompt(), json.dumps(payload, ensure_ascii=False),
        purpose="dependency_judge",
    )
    raw = parse_json_object(result.text)
    return parse_dependency_decisions(graph, targets, raw), result


def dependency_request_payload(
    graph: TaskGraph, targets: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Build a request with one shared node catalog and lightweight references."""
    target_rows = [
        _target_payload(graph, target_id, audit) for target_id, audit in targets
    ]
    node_ids = {
        node_id
        for target_id, audit in targets
        for node_id in (target_id, *audit["Cfinal"])
    }
    return {
        "node_catalog": {
            node_id: _node_payload(graph, node_id) for node_id in sorted(node_ids)
        },
        "targets": target_rows,
    }


def dependency_request_batches(
    graph: TaskGraph,
    targets: list[tuple[str, dict[str, Any]]],
    *,
    max_targets: int = 4,
    max_candidate_refs: int = 768,
) -> list[tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]]:
    """Partition targets without dropping candidates; each batch is independently valid."""
    if max_targets < 1 or max_candidate_refs < 1:
        raise ValueError("Dependency batch limits must be positive")
    groups: list[list[tuple[str, dict[str, Any]]]] = []
    current: list[tuple[str, dict[str, Any]]] = []
    current_refs = 0
    for target in targets:
        refs = len(target[1]["Cfinal"])
        if current and (
            len(current) >= max_targets or current_refs + refs > max_candidate_refs
        ):
            groups.append(current)
            current, current_refs = [], 0
        current.append(target)
        current_refs += refs
    if current:
        groups.append(current)
    return [(group, dependency_request_payload(graph, group)) for group in groups]


def parse_dependency_decisions(
    graph: TaskGraph,
    targets: list[tuple[str, dict[str, Any]]],
    raw: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Validate model decisions against the frozen pre-judgment candidate set."""
    returned = raw.get("targets", [])
    if not isinstance(returned, list):
        raise RuntimeError("Dependency judge must return a targets array")
    allowed_targets = {target_id: set(audit["Cfinal"]) for target_id, audit in targets}
    required_stable = {
        target_id: {
            str(detail["source_id"])
            for detail in audit["candidate_details"]
            if detail.get("stable_key_match")
        }
        for target_id, audit in targets
    }
    stable_by_target = {target_id: set(rows) for target_id, rows in required_stable.items()}
    decisions: dict[str, list[dict[str, Any]]] = {target_id: [] for target_id, _ in targets}
    seen_targets: set[str] = set()
    for target_row in returned:
        if not isinstance(target_row, dict):
            raise RuntimeError("Every dependency target row must be an object")
        target_id = str(target_row.get("target_id", ""))
        if target_id not in allowed_targets:
            raise RuntimeError(f"Dependency judge returned unknown target: {target_id}")
        if target_id in seen_targets:
            raise RuntimeError(f"Dependency judge duplicated target: {target_id}")
        seen_targets.add(target_id)
        rows = target_row.get("dependencies", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"Dependencies for {target_id} must be an array")
        seen: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise RuntimeError(f"Every dependency for {target_id} must be an object")
            source_id = str(row.get("source_id", ""))
            if source_id not in allowed_targets[target_id]:
                raise RuntimeError(
                    f"Dependency judge selected non-candidate {source_id} for {target_id}"
                )
            if source_id in seen:
                raise RuntimeError(
                    f"Dependency judge duplicated source {source_id} for {target_id}"
                )
            if graph.nodes[source_id].level.value == "L1":
                raise RuntimeError("L1 cannot be selected as a dependency")
            dependency_type = str(row.get("dependency_type", ""))
            if dependency_type not in _DEPENDENCY_TYPES:
                raise RuntimeError(
                    f"Invalid dependency_type for {source_id}: {dependency_type}"
                )
            if (
                source_id in stable_by_target[target_id]
                and dependency_type != "state_continuity"
            ):
                raise RuntimeError(
                    f"Stable-state predecessor {source_id} must use state_continuity"
                )
            confidence = _validated_confidence(row.get("confidence"), source_id)
            priority = _validated_priority(row.get("priority", rank), source_id)
            reason = str(row.get("reason", "")).strip()
            if not reason:
                raise RuntimeError(f"Dependency reason is required for {source_id}")
            seen.add(source_id)
            decisions[target_id].append({
                "source_id": source_id,
                "dependency_type": dependency_type,
                "confidence": confidence,
                "priority": priority,
                "reason": reason,
            })
        missing_stable = required_stable[target_id] - seen
        if missing_stable:
            raise RuntimeError(
                f"Dependency judge omitted stable-state predecessor(s) for {target_id}: "
                f"{sorted(missing_stable)}"
            )
    missing_targets = set(allowed_targets) - seen_targets
    if missing_targets:
        raise RuntimeError(
            f"Dependency judge omitted target row(s): {sorted(missing_targets)}"
        )
    return decisions


def _target_payload(graph: TaskGraph, target_id: str, audit: dict[str, Any]) -> dict[str, Any]:
    details = {item["source_id"]: item for item in audit["candidate_details"]}
    return {
        "target_id": target_id,
        "target_node_id": target_id,
        "candidates": [
            {
                "node_id": source_id,
                **{
                    key: value for key, value in details[source_id].items()
                    if key != "source_id"
                },
            }
            for source_id in audit["Cfinal"]
        ],
    }


def _node_payload(graph: TaskGraph, node_id: str) -> dict[str, Any]:
    node = graph.nodes[node_id]
    return {
        "node_id": node_id,
        "level": node.level.value,
        "turn_index": node.turn_index,
        "status": node.status.value,
        "text": node_text(node),
    }


def _validated_confidence(value: Any, source_id: str) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid confidence for {source_id}: {value!r}") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError(f"Confidence for {source_id} must be in [0, 1]")
    return round(confidence, 6)


def _validated_priority(value: Any, source_id: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Invalid priority for {source_id}: {value!r}")
    try:
        priority = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid priority for {source_id}: {value!r}") from exc
    if priority < 1 or str(value).strip() not in {str(priority), f"{priority}.0"}:
        raise RuntimeError(f"Priority for {source_id} must be a positive integer")
    return priority


def dependency_judge_system_prompt() -> str:
    return (
        "你是TaskGraph依赖相关性判断器。输入包含一个或多个新目标节点及其历史L2-L4候选。"
        "node_catalog按node_id集中保存节点全文；targets中的target_node_id和candidates仅引用该目录，"
        "判断前必须通过node_id读取对应全文，输出时把所选候选的node_id写入source_id。"
        "对每个目标选择所有具有实质相关性的候选，不设数量配额。相关是指候选能帮助理解、续写或保持"
        "当前目标的人物、物品、地点、时间、世界规则、知识边界、承诺、因果、早期线索或情节连续性；"
        "不要求它是删除后任务无法执行的硬前置条件。仅因同属一种题材、时间相邻或共享普通套话不算相关。"
        "不得选择输入列表之外的节点，不得选择L1。stable_key_match为true表示同一状态槽的上一版本，"
        "必须选择为state_continuity。按对当前目标的帮助程度给priority，1最高。"
        "dependency_type使用state_continuity、entity_continuity、constraint、causal_support、"
        "long_range_support、plot_support或contextual_relevance之一。只输出严格JSON："
        '{"targets":[{"target_id":"...","dependencies":[{"source_id":"...",'
        '"dependency_type":"plot_support","confidence":0.0,"priority":1,"reason":"简短依据"}]}]}。'
        "没有相关候选时dependencies为空数组。不要输出分析过程。"
    )
