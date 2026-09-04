from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class NodeStatus(str, Enum):
    PENDING = "Pending"
    ACTIVE = "Active"
    DONE = "Done"
    DROPPED = "Dropped"


class EdgeType(str, Enum):
    MAINLINE = "MAINLINE"
    INCLUSION = "INCLUSION"
    DEPENDENCY = "DEPENDENCY"


@dataclass
class Node:
    node_id: str
    level: NodeLevel
    content: str
    turn_index: int
    path: str
    status: NodeStatus = NodeStatus.PENDING
    value: str = ""
    sub_type: str = ""
    is_global: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["level"] = self.level.value
        data["status"] = self.status.value
        return data


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "metadata": self.metadata,
        }


class TaskGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._counters = {level: 0 for level in NodeLevel}

    def _next_id(self, level: NodeLevel, hint: str = "") -> str:
        self._counters[level] += 1
        prefix = hint or {"L1": "Task", "L2": "Stage", "L3": "Step", "L4": "Item"}[level.value]
        return f"{level.value}_{prefix}{self._counters[level]}"

    def add_node(
        self,
        level: NodeLevel,
        content: str,
        turn_index: int,
        path: str,
        *,
        hint: str = "",
        status: NodeStatus = NodeStatus.PENDING,
        value: str = "",
        sub_type: str = "",
        is_global: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        node_id = self._next_id(level, hint)
        self.nodes[node_id] = Node(
            node_id=node_id,
            level=level,
            content=content,
            turn_index=turn_index,
            path=path,
            status=status,
            value=value,
            sub_type=sub_type,
            is_global=is_global,
            metadata=metadata or {},
        )
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError(f"Cannot add edge {source_id}->{target_id}: missing node")
        self.edges.append(Edge(source_id, target_id, edge_type, metadata or {}))

    def children(self, node_id: str, level: NodeLevel | None = None) -> list[Node]:
        ids = [
            edge.target_id
            for edge in self.edges
            if edge.source_id == node_id and edge.edge_type == EdgeType.INCLUSION
        ]
        result = [self.nodes[item] for item in ids]
        return [node for node in result if level is None or node.level == level]

    def dependencies(self, node_id: str) -> list[Node]:
        ids = [
            edge.source_id
            for edge in self.edges
            if edge.target_id == node_id and edge.edge_type == EdgeType.DEPENDENCY
        ]
        return [self.nodes[item] for item in ids]

    def mainline_order(self, parent_id: str, level: NodeLevel) -> list[Node]:
        siblings = self.children(parent_id, level=level)
        sibling_ids = {node.node_id for node in siblings}
        successors = {
            edge.source_id: edge.target_id
            for edge in self.edges
            if edge.edge_type == EdgeType.MAINLINE
            and edge.source_id in sibling_ids
            and edge.target_id in sibling_ids
        }
        targets = set(successors.values())
        starts = [node.node_id for node in siblings if node.node_id not in targets]
        if not starts:
            return siblings
        ordered: list[Node] = []
        current = starts[0]
        seen: set[str] = set()
        while current in sibling_ids and current not in seen:
            seen.add(current)
            ordered.append(self.nodes[current])
            current = successors.get(current, "")
        ordered.extend(node for node in siblings if node.node_id not in seen)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "taskgraph-v1",
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGraph":
        graph = cls()
        for raw in data.get("nodes", []):
            node = Node(
                node_id=raw["node_id"],
                level=NodeLevel(raw["level"]),
                content=raw.get("content", ""),
                turn_index=int(raw.get("turn_index", 0)),
                path=raw.get("path", ""),
                status=NodeStatus(raw.get("status", NodeStatus.PENDING.value)),
                value=raw.get("value", ""),
                sub_type=raw.get("sub_type", ""),
                is_global=bool(raw.get("is_global", False)),
                metadata=dict(raw.get("metadata", {})),
            )
            graph.nodes[node.node_id] = node
            graph._counters[node.level] = max(graph._counters[node.level], _trailing_int(node.node_id))
        for raw in data.get("edges", []):
            graph.edges.append(
                Edge(
                    raw["source_id"],
                    raw["target_id"],
                    EdgeType(raw["edge_type"]),
                    dict(raw.get("metadata", {})),
                )
            )
        return graph


def _trailing_int(value: str) -> int:
    digits = ""
    for ch in reversed(value):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits or 0)
