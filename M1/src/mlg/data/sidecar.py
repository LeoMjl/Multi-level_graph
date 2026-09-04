from __future__ import annotations

from typing import Any

from mlg.graph import EdgeType, NodeLevel, NodeStatus, TaskGraph
from mlg.schemas import SAFE_METHOD_METADATA_KEYS, Episode


SIDECAR_SCHEMA = "mlg-sidecar-v2"


def make_sidecar(
    episode: Episode,
    *,
    experiment: str,
    stage_records: list[dict[str, Any]] | None = None,
    dependency_edges: list[dict[str, Any]] | None = None,
    judge_required_fields: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build observable method state and evaluator annotations in separate namespaces.

    The observable graph is derived exclusively from the raw episode.  Gold stages,
    dependencies, answers, and judge labels live under ``evaluator_gold`` and are
    never used to construct it.
    """
    graph = TaskGraph()
    task_id = graph.add_node(
        NodeLevel.L1,
        content=f"{episode.dataset}: {episode.query[:140]}",
        turn_index=0,
        path="T1",
        status=NodeStatus.ACTIVE,
    )
    # A precomputed sidecar must not infer its active stage from evaluator labels.
    # The graph method can infer a more specific stage from raw history at runtime.
    stage_name = "task_execution"
    stage_id = graph.add_node(
        NodeLevel.L2,
        content=stage_name,
        turn_index=0,
        path="T1.S1",
        status=NodeStatus.ACTIVE,
        metadata={"stage": stage_name},
    )
    graph.add_edge(task_id, stage_id, EdgeType.INCLUSION)

    prev_step = ""
    for idx, msg in enumerate(episode.history, start=1):
        step_id = graph.add_node(
            NodeLevel.L3,
            content=f"{msg.role} turn {idx}",
            turn_index=msg.turn_index or idx,
            path=f"T1.S1.Step{idx}",
            status=NodeStatus.DONE,
            value=msg.content,
            metadata={
                key: value
                for key, value in msg.metadata.items()
                if key in SAFE_METHOD_METADATA_KEYS
            },
        )
        graph.add_edge(stage_id, step_id, EdgeType.INCLUSION)
        if prev_step:
            graph.add_edge(prev_step, step_id, EdgeType.MAINLINE)
        prev_step = step_id
        for fact_idx, fact in enumerate(_split_facts(msg.content), start=1):
            item_id = graph.add_node(
                NodeLevel.L4,
                content=fact[:120],
                turn_index=msg.turn_index or idx,
                path=f"T1.S1.Step{idx}.Item{fact_idx}",
                status=NodeStatus.DONE,
                value=fact,
                sub_type=_fact_type(fact),
                is_global=_is_global_fact(fact),
            )
            graph.add_edge(step_id, item_id, EdgeType.INCLUSION)
            graph.add_edge(item_id, step_id, EdgeType.DEPENDENCY, {"reason": "extracted_from_raw_episode"})

    query_id = graph.add_node(
        NodeLevel.L3,
        content=f"Answer query: {episode.query[:160]}",
        turn_index=max((msg.turn_index for msg in episode.history), default=0) + 1,
        path="T1.S1.Query",
        status=NodeStatus.ACTIVE,
        metadata={"query": episode.query},
    )
    graph.add_edge(stage_id, query_id, EdgeType.INCLUSION)
    if prev_step:
        graph.add_edge(prev_step, query_id, EdgeType.MAINLINE)
    if (
        experiment == "m3"
        and episode.metadata.get("task_type") == "stabletoolbench_tool_use"
    ):
        graph = _stabletoolbench_initial_graph(episode)
    graph_dict = graph.to_dict()
    mainline_edges = [edge for edge in graph_dict["edges"] if edge["edge_type"] == EdgeType.MAINLINE.value]
    inclusion_edges = [edge for edge in graph_dict["edges"] if edge["edge_type"] == EdgeType.INCLUSION.value]
    graph_dependency_edges = [edge for edge in graph_dict["edges"] if edge["edge_type"] == EdgeType.DEPENDENCY.value]
    evaluator_dependency_edges = dependency_edges if dependency_edges is not None else [
        {
            "source": str(dep.get("source", "")),
            "target": str(dep.get("target", "")),
            "edge_type": EdgeType.DEPENDENCY.value,
            "metadata": {"reason": "evaluator_gold_dependency"},
        }
        for dep in episode.gold_dependencies
    ]

    return {
        "schema": SIDECAR_SCHEMA,
        "item_id": episode.episode_id,
        "experiment": experiment,
        "raw_episode": {
            "episode_id": episode.episode_id,
            "dataset": episode.dataset,
            "split": episode.split,
            "history": [msg.__dict__ for msg in episode.history],
            "query": episode.query,
            "metadata": {
                key: value
                for key, value in episode.metadata.items()
                if _safe_raw_metadata_key(key)
            },
        },
        "observable_graph": {
            "nodes": graph_dict["nodes"],
            "mainline_edges": mainline_edges,
            "inclusion_edges": inclusion_edges,
            "dependency_edges": graph_dependency_edges,
            "active_node_pool": [
                {"node_id": node["node_id"], "level": node["level"], "content": node["content"]}
                for node in graph_dict["nodes"]
                if node.get("status") == NodeStatus.ACTIVE.value
            ],
        },
        "evaluator_gold": {
            "stage_records": stage_records if stage_records is not None else _stage_records(episode),
            "dependency_edges": evaluator_dependency_edges,
            "gold_outputs": {
                "answers": list(episode.answers),
                "evidence": list(episode.gold_evidence),
                "stage": episode.gold_stage,
                "dependencies": list(episode.gold_dependencies),
                "metadata": dict(episode.metadata),
            },
            "judge_required_fields": (
                judge_required_fields
                if judge_required_fields is not None
                else _judge_fields(episode, experiment)
            ),
        },
        "metadata": {
            "source_dataset": episode.dataset,
            "source_experiment": experiment,
            **(metadata or {}),
        },
    }


def attach_sidecars(episodes: list[Episode], experiment: str) -> list[Episode]:
    for episode in episodes:
        if not episode.sidecar or episode.sidecar.get("schema") != SIDECAR_SCHEMA:
            episode.sidecar = make_sidecar(episode, experiment=experiment)
    return episodes


def _split_facts(text: str) -> list[str]:
    chunks = []
    for raw in text.replace("\r", "\n").split("\n"):
        for part in raw.split(". "):
            clean = part.strip()
            if clean:
                chunks.append(clean[:500])
    return chunks[:12]


def _stabletoolbench_initial_graph(episode: Episode) -> TaskGraph:
    """Build the observable pre-execution graph from official API definitions."""
    graph = TaskGraph()
    task_id = graph.add_node(
        NodeLevel.L1,
        content=f"StableToolBench query: {episode.query[:160]}",
        turn_index=0,
        path="T1",
        status=NodeStatus.ACTIVE,
    )
    stage_id = graph.add_node(
        NodeLevel.L2,
        content="interactive_tool_use",
        turn_index=0,
        path="T1.S1",
        status=NodeStatus.ACTIVE,
        metadata={"stage": "tool_execution"},
    )
    graph.add_edge(task_id, stage_id, EdgeType.INCLUSION)
    query_id = graph.add_node(
        NodeLevel.L3,
        content="User query",
        turn_index=0,
        path="T1.S1.Query",
        status=NodeStatus.ACTIVE,
        value=episode.query,
    )
    graph.add_edge(stage_id, query_id, EdgeType.INCLUSION)
    tools = episode.metadata.get("tools", [])
    for api_index, api in enumerate(tools, start=1):
        if not isinstance(api, dict):
            continue
        tool_name = str(api.get("tool_name", ""))
        api_name = str(api.get("api_name", ""))
        api_id = graph.add_node(
            NodeLevel.L3,
            content=f"API {tool_name}.{api_name}",
            turn_index=0,
            path=f"T1.S1.API{api_index}",
            status=NodeStatus.PENDING,
            value=str(api.get("api_description", "")),
            sub_type="AvailableAPI",
            metadata={
                "category_name": str(api.get("category_name", "")),
                "http_method": str(api.get("method", "")),
            },
        )
        graph.add_edge(stage_id, api_id, EdgeType.INCLUSION)
        graph.add_edge(
            api_id,
            query_id,
            EdgeType.DEPENDENCY,
            {"reason": "official_available_api"},
        )
        parameters = [
            (parameter, required)
            for required, key in (
                (True, "required_parameters"),
                (False, "optional_parameters"),
            )
            for parameter in api.get(key, [])
            if isinstance(parameter, dict)
        ]
        for param_index, (parameter, required) in enumerate(parameters, start=1):
            param_id = graph.add_node(
                NodeLevel.L4,
                content=str(parameter.get("name", "")),
                turn_index=0,
                path=f"T1.S1.API{api_index}.Param{param_index}",
                status=NodeStatus.PENDING,
                value=str(parameter.get("description", "")),
                sub_type="RequiredParameter" if required else "OptionalParameter",
                metadata={"parameter_type": str(parameter.get("type", ""))},
            )
            graph.add_edge(api_id, param_id, EdgeType.INCLUSION)
            graph.add_edge(
                param_id,
                api_id,
                EdgeType.DEPENDENCY,
                {"reason": "official_api_parameter"},
            )
    return graph


def _fact_type(text: str) -> str:
    lower = text.lower()
    if any(marker in lower for marker in ("must", "should", "constraint", "required", "do not", "avoid")):
        return "Constraint"
    if any(marker in lower for marker in ("tool", "api", "action", "click", "type", "select")):
        return "Action"
    return "Fact"


def _is_global_fact(text: str) -> bool:
    lower = text.lower()
    return "global constraint" in lower or lower.startswith("constraint:") or "must" in lower


def _stage_records(episode: Episode) -> list[dict[str, Any]]:
    if episode.gold_stage:
        return [{"turn_index": len(episode.history), "stage": episode.gold_stage, "status": "active"}]
    return []


def _judge_fields(episode: Episode, experiment: str) -> list[str]:
    fields: list[str] = []
    if episode.gold_stage:
        fields.append("stage")
    if episode.gold_dependencies:
        fields.append("dependency")
    if episode.metadata.get("constraints"):
        fields.append("constraint")
    if experiment == "m2":
        fields.append("event_summary")
    if experiment in {"m3"}:
        fields.append("next_action")
    return sorted(set(fields))


def _safe_raw_metadata_key(key: str) -> bool:
    lower = key.lower()
    if any(marker in lower for marker in ("gold", "expected", "answer", "label", "stage_sequence")):
        return False
    return lower not in {"intervention_template", "seed"}
