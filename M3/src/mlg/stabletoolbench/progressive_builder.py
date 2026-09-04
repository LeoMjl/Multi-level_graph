from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mlg.graph import EdgeType, NodeLevel, NodeStatus, TaskGraph


@dataclass
class ProgressiveLayout:
    task_id: str
    stage_ids: list[str]
    business_stage_ids: list[str]
    final_stage_id: str
    stage_by_ref: dict[str, str]
    step_ids: list[str] = field(default_factory=list)
    step_by_ref: dict[str, str] = field(default_factory=dict)
    step_tools: dict[str, list[str]] = field(default_factory=dict)
    tools_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_state_ids: list[str] = field(default_factory=list)


def build_coarse_graph(
    query: str,
    tools: list[dict[str, Any]],
    coarse_plan: dict[str, Any],
) -> tuple[TaskGraph, ProgressiveLayout]:
    graph = TaskGraph()
    task_id = graph.add_node(
        NodeLevel.L1,
        "StableToolBench task",
        0,
        "T1",
        status=NodeStatus.ACTIVE,
        value=query,
        metadata={"source": "observable_user_query"},
    )
    stage_ids: list[str] = []
    stage_by_ref: dict[str, str] = {}
    prior_stage = ""
    for index, stage in enumerate(coarse_plan.get("stages", []), start=1):
        stage_ref = str(stage.get("id", f"S{index}"))
        stage_id = graph.add_node(
            NodeLevel.L2,
            str(stage.get("name", f"Stage {index}")),
            0,
            f"T1.{stage_ref}",
            status=NodeStatus.PENDING,
            value=str(stage.get("goal", "")),
            sub_type="CoarseStage",
            metadata={
                "plan_ref": stage_ref,
                "depends_on": list(stage.get("depends_on", [])),
                "expanded": False,
            },
        )
        graph.add_edge(task_id, stage_id, EdgeType.INCLUSION)
        if prior_stage:
            graph.add_edge(prior_stage, stage_id, EdgeType.MAINLINE)
        stage_ids.append(stage_id)
        stage_by_ref[stage_ref] = stage_id
        prior_stage = stage_id
    final_stage_id = graph.add_node(
        NodeLevel.L2,
        "Finalize answer",
        0,
        "T1.FINALIZE",
        status=NodeStatus.PENDING,
        value="Synthesize verified observations into the complete final answer.",
        sub_type="FinalizationStage",
        metadata={
            "plan_ref": "FINALIZE",
            "expanded": False,
            "is_finalization": True,
        },
    )
    graph.add_edge(task_id, final_stage_id, EdgeType.INCLUSION)
    if prior_stage:
        graph.add_edge(prior_stage, final_stage_id, EdgeType.MAINLINE)
    all_stages = [*stage_ids, final_stage_id]
    stage_by_ref["FINALIZE"] = final_stage_id
    layout = ProgressiveLayout(
        task_id=task_id,
        stage_ids=all_stages,
        business_stage_ids=list(stage_ids),
        final_stage_id=final_stage_id,
        stage_by_ref=stage_by_ref,
        tools_by_name=_index_tools(tools),
    )
    _attach_stage_dependencies(graph, coarse_plan, stage_by_ref)
    _attach_global_state(graph, layout, query, coarse_plan.get("global_state", []))
    return graph, layout


def expand_stage(
    graph: TaskGraph,
    layout: ProgressiveLayout,
    stage_id: str,
    steps: list[dict[str, Any]],
) -> list[str]:
    stage = graph.nodes[stage_id]
    if stage.metadata.get("expanded"):
        return [node.node_id for node in graph.children(stage_id, NodeLevel.L3)]
    created: list[str] = []
    prior_step = ""
    for index, step in enumerate(steps, start=1):
        step_ref = str(step.get("id", f"{stage.metadata['plan_ref']}.{index}"))
        if step_ref in layout.step_by_ref:
            step_ref = f"{stage.metadata['plan_ref']}.{index}"
        names = [
            str(name) for name in step.get("candidate_tools", [])
            if str(name) in layout.tools_by_name and str(name) != "Finish"
        ][:1]
        if not names:
            continue
        step_id = graph.add_node(
            NodeLevel.L3,
            str(step.get("name", f"Step {index}")),
            0,
            f"{stage.path}.{step_ref}",
            status=NodeStatus.PENDING,
            value=str(step.get("goal", "")),
            sub_type="ExpandedStep",
            metadata={
                "plan_ref": step_ref,
                "depends_on": list(step.get("depends_on", [])),
                "references": list(step.get("references", [])),
                "candidate_tools": names,
                "expanded_in_stage": stage.metadata.get("plan_ref"),
            },
        )
        graph.add_edge(stage_id, step_id, EdgeType.INCLUSION)
        if prior_step:
            graph.add_edge(prior_step, step_id, EdgeType.MAINLINE)
        prior_step = step_id
        created.append(step_id)
        layout.step_ids.append(step_id)
        layout.step_by_ref[step_ref] = step_id
        layout.step_tools[step_id] = names
        _attach_tool_capability(graph, step_id, names[0], layout.tools_by_name)
    stage.metadata["expanded"] = True
    return created


def expand_final_stage(graph: TaskGraph, layout: ProgressiveLayout) -> str:
    existing = graph.children(layout.final_stage_id, NodeLevel.L3)
    if existing:
        return existing[0].node_id
    step_id = graph.add_node(
        NodeLevel.L3,
        "Synthesize and return the final answer",
        0,
        "T1.FINALIZE.FINAL",
        status=NodeStatus.PENDING,
        value="Answer the complete user query from verified TaskGraph state.",
        sub_type="FinalAnswerStep",
        metadata={"plan_ref": "FINAL", "candidate_tools": ["Finish"]},
    )
    graph.add_edge(layout.final_stage_id, step_id, EdgeType.INCLUSION)
    for source_id in layout.step_ids:
        graph.add_edge(
            source_id,
            step_id,
            EdgeType.DEPENDENCY,
            {
                "reason": "available_for_final_synthesis",
                "hard": False,
                "blocking": False,
            },
        )
    layout.step_ids.append(step_id)
    layout.step_by_ref["FINAL"] = step_id
    layout.step_tools[step_id] = ["Finish"]
    graph.nodes[layout.final_stage_id].metadata["expanded"] = True
    return step_id


def _index_tools(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in tools:
        function = raw.get("function", raw)
        if isinstance(function, dict) and function.get("name"):
            result[str(function["name"])] = raw
    return result


def _attach_stage_dependencies(graph, plan, stage_by_ref) -> None:
    for stage in plan.get("stages", []):
        target = stage_by_ref.get(str(stage.get("id", "")))
        for source_ref in stage.get("depends_on", []):
            source = stage_by_ref.get(str(source_ref))
            if source and target and source != target:
                graph.add_edge(source, target, EdgeType.DEPENDENCY, {
                    "reason": "declared_stage_prerequisite", "hard": True,
                })


def _attach_global_state(graph, layout, query, state_items) -> None:
    normalized_query = " ".join(query.casefold().split())
    for index, item in enumerate(state_items, start=1):
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", "")).strip()
        if not key or not value:
            continue
        if " ".join(value.casefold().split()) not in normalized_query:
            continue
        node_id = graph.add_node(
            NodeLevel.L4, key, 0,
            f"T1.Global{index}", status=NodeStatus.ACTIVE, value=value,
            sub_type="GlobalConstraint", is_global=True,
            metadata={"source": "observable_user_query"},
        )
        graph.add_edge(layout.task_id, node_id, EdgeType.INCLUSION)
        layout.global_state_ids.append(node_id)


def _attach_tool_capability(graph, step_id, name, tools_by_name) -> None:
    function = tools_by_name[name].get("function", tools_by_name[name])
    parameters = function.get("parameters", {})
    raw_properties = parameters.get("properties", {})
    property_hints = {
        str(key): {
            hint: value.get(hint)
            for hint in ("description", "example_value", "enum")
            if value.get(hint) not in (None, "", [])
        }
        for key, value in raw_properties.items()
        if isinstance(value, dict)
    }
    summary = {
        "description": str(function.get("description", ""))[:320],
        "required": list(parameters.get("required", [])),
        "properties": property_hints,
    }
    node_id = graph.add_node(
        NodeLevel.L4, f"Available API: {name}", 0,
        f"{graph.nodes[step_id].path}.Tool", status=NodeStatus.ACTIVE,
        value=json.dumps(summary, ensure_ascii=False), sub_type="ToolCapability",
        metadata={"tool_name": name},
    )
    graph.add_edge(step_id, node_id, EdgeType.INCLUSION)
