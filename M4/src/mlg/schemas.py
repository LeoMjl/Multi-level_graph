from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SAFE_METHOD_METADATA_KEYS = {
    "constraints",
    "context_length",
    "context_length_bucket",
    "graph_tracking",
    "needle_depth_ratio",
    "needle_position",
    "query_id",
    "request",
    "session_count",
    "session_date",
    "session_id",
    "solvable",
    "speaker",
    "source_file",
    "source_row_keys",
    "source_subset",
    "split",
    "subset",
    "synthetic_intervention",
    "task_type",
    "target_speaker",
    "timeframe",
    "tools",
}


@dataclass
class Message:
    role: str
    content: str
    turn_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    episode_id: str
    dataset: str
    split: str
    history: list[Message]
    query: str
    history_ref: dict[str, Any] = field(default_factory=dict)
    answers: list[str] = field(default_factory=list)
    gold_evidence: list[str] = field(default_factory=list)
    gold_stage: str = ""
    gold_dependencies: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    sidecar: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_sidecar: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_sidecar:
            data.pop("sidecar", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        history = [
            msg if isinstance(msg, Message) else Message(**msg)
            for msg in data.get("history", [])
        ]
        return cls(
            episode_id=str(data.get("episode_id", "")),
            dataset=str(data.get("dataset", "")),
            split=str(data.get("split", "")),
            history=history,
            query=str(data.get("query", "")),
            history_ref=dict(data.get("history_ref", {})),
            answers=[str(x) for x in data.get("answers", [])],
            gold_evidence=[str(x) for x in data.get("gold_evidence", [])],
            gold_stage=str(data.get("gold_stage", "")),
            gold_dependencies=list(data.get("gold_dependencies", [])),
            metadata=dict(data.get("metadata", {})),
            sidecar=dict(data.get("sidecar", {})),
        )

    def method_input(self, *, include_visible_graph: bool = False, graph_ablation: str = "") -> dict[str, Any]:
        """Payload visible to methods. Gold labels are intentionally excluded."""
        history = [_method_message(msg) for msg in self.history]
        history_ref = {
            key: self.history_ref[key]
            for key in ("schema", "signature", "message_count")
            if key in self.history_ref
        }
        payload = {
            "episode_id": self.episode_id,
            "dataset": self.dataset,
            "split": self.split,
            "history": history,
            "history_ref": history_ref,
            "query": self.query,
            "raw_episode": {
                "history": history,
                "history_ref": history_ref,
                "query": self.query,
                "dataset": self.dataset,
                "split": self.split,
            },
            "metadata": {
                key: value
                for key, value in self.metadata.items()
                if key in SAFE_METHOD_METADATA_KEYS
            },
        }
        if include_visible_graph and self.sidecar:
            payload["sidecar"] = visible_sidecar(self.sidecar, graph_ablation=graph_ablation)
        return payload


@dataclass
class Prediction:
    episode_id: str
    method: str
    model_profile: str = "default"
    answer: str = ""
    evidence: list[str] = field(default_factory=list)
    stage: str = ""
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    # API-billed usage only. ``None`` means at least one required provider call did
    # not return usage; formal reports must not replace it with a tokenizer estimate.
    token_usage: int | None = None
    api_usage: dict[str, Any] = field(default_factory=dict)
    memory_build_time_ms: float = 0.0
    memory_query_time_ms: float = 0.0
    reader_generation_time_ms: float = 0.0
    end_to_end_time_ms: float = 0.0
    status: str = "ok"
    failure_stage: str = ""
    failure_reason: str = ""
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricRecord:
    experiment: str
    method: str
    episode_id: str
    metrics: dict[str, float]
    model_profile: str = "default"
    # run_mode distinguishes real-LLM runs ("llm") from deterministic smoke runs ("fake").
    # Keeping them separate prevents smoke-test numbers from polluting publishable results.
    run_mode: str = "llm"
    # fallback records whether the method fell back to keyword retrieval because the LLM
    # returned an empty answer or errored. Aggregations can surface fallback rate per method.
    fallback: str = ""
    failure: str = ""
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def visible_sidecar(sidecar: dict[str, Any], *, graph_ablation: str = "") -> dict[str, Any]:
    """Return the structurally isolated, non-oracle sidecar method view.

    Version-1 sidecars mixed observable and evaluator data at the top level.  They
    fail closed here: the method receives no precomputed graph and rebuilds one from
    its raw payload instead of risking use of contaminated nodes or edges.
    """
    observable = sidecar.get("observable_graph", {}) if sidecar.get("schema") == "mlg-sidecar-v2" else {}
    nodes = [dict(node) for node in observable.get("nodes", []) if isinstance(node, dict)]
    mainline_edges = [dict(edge) for edge in observable.get("mainline_edges", []) if _is_visible_edge(edge)]
    inclusion_edges = [dict(edge) for edge in observable.get("inclusion_edges", []) if _is_visible_edge(edge)]
    dependency_edges = [dict(edge) for edge in observable.get("dependency_edges", []) if _is_visible_edge(edge)]
    active_pool = [
        dict(item)
        for item in observable.get("active_node_pool", [])
        if isinstance(item, dict)
    ]

    if graph_ablation == "wo_hier":
        inclusion_edges = []
        for node in nodes:
            if node.get("level") in {"L1", "L2"}:
                node["status"] = "Pending"
    elif graph_ablation == "wo_mainline":
        mainline_edges = []
    elif graph_ablation == "wo_dep":
        dependency_edges = []
    elif graph_ablation == "wo_active_pool":
        active_pool = []
        for node in nodes:
            if node.get("status") == "Active":
                node["status"] = "Pending"

    return {
        "schema": sidecar.get("schema", ""),
        "item_id": sidecar.get("item_id", ""),
        "raw_episode": _visible_raw_episode(sidecar.get("raw_episode", {})),
        "visible_graph": {
            "nodes": nodes,
            "edges": mainline_edges + inclusion_edges + dependency_edges,
            "mainline_edges": mainline_edges,
            "inclusion_edges": inclusion_edges,
            "dependency_edges": dependency_edges,
            "active_node_pool": active_pool,
        },
        "metadata": {
            key: value
            for key, value in dict(sidecar.get("metadata", {})).items()
            if key in {"source_dataset", "source_experiment"}
        },
    }


def _is_visible_edge(edge: dict[str, Any]) -> bool:
    if not edge.get("source_id") or not edge.get("target_id"):
        return False
    serialized = json.dumps(edge, ensure_ascii=False).lower()
    return not any(marker in serialized for marker in ("evaluator_gold", "gold_outputs", "gold_visible"))


def _method_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "turn_index": message.turn_index,
        "metadata": {
            key: value
            for key, value in message.metadata.items()
            if key in SAFE_METHOD_METADATA_KEYS
        },
    }


def _visible_raw_episode(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    history = []
    for message in raw.get("history", []):
        if not isinstance(message, dict):
            continue
        history.append({
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
            "turn_index": int(message.get("turn_index", 0) or 0),
            "metadata": {
                key: value
                for key, value in dict(message.get("metadata", {})).items()
                if key in SAFE_METHOD_METADATA_KEYS
            },
        })
    return {
        "episode_id": str(raw.get("episode_id", "")),
        "dataset": str(raw.get("dataset", "")),
        "split": str(raw.get("split", "")),
        "history": history,
        "query": str(raw.get("query", "")),
        "metadata": {
            key: value
            for key, value in dict(raw.get("metadata", {})).items()
            if key in SAFE_METHOD_METADATA_KEYS
        },
    }
