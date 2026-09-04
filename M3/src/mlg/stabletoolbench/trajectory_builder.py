from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mlg.graph import EdgeType, NodeLevel, NodeStatus, TaskGraph


@dataclass
class TrajectoryLayout:
    task_id: str
    stage_ids: list[str]
    step_ids: list[str]
    step_by_ref: dict[str, str]
    step_tools: dict[str, list[str]]
    tools_by_name: dict[str, dict[str, Any]]


def build_planned_graph(
    query: str,
    tools: list[dict[str, Any]],
    plan: list[dict[str, Any]],
) -> tuple[TaskGraph, TrajectoryLayout]:
    graph = TaskGraph()
    task_id = graph.add_node(
        NodeLevel.L1,
        content="StableToolBench task",
        turn_index=0,
        path="T1",
        status=NodeStatus.ACTIVE,
        value=query,
        metadata={"source": "observable_user_query"},
    )
    tools_by_name = _index_tools(tools)
    stage_ids: list[str] = []
    step_ids: list[str] = []
    step_by_ref: dict[str, str] = {}
    step_tools: dict[str, list[str]] = {}
    prior_stage = ""
    for stage_index, stage in enumerate(plan, start=1):
        stage_ref = str(stage.get("id", f"S{stage_index}"))
        stage_id = graph.add_node(
            NodeLevel.L2,
            content=str(stage.get("name", f"Stage {stage_index}")),
            turn_index=0,
            path=f"T1.{stage_ref}",
            status=NodeStatus.PENDING,
            value=str(stage.get("goal", "")),
            sub_type="PlannedStage",
            metadata={"plan_ref": stage_ref},
        )
        graph.add_edge(task_id, stage_id, EdgeType.INCLUSION)
        if prior_stage:
            graph.add_edge(prior_stage, stage_id, EdgeType.MAINLINE)
        prior_stage = stage_id
        stage_ids.append(stage_id)
        prior_step = ""
        for local_index, step in enumerate(stage.get("steps", []), start=1):
            step_ref = str(step.get("id", f"{stage_ref}.{local_index}"))
            step_id = graph.add_node(
                NodeLevel.L3,
                content=str(step.get("name", f"Step {local_index}")),
                turn_index=0,
                path=f"T1.{stage_ref}.{step_ref}",
                status=NodeStatus.PENDING,
                value=str(step.get("goal", "")),
                sub_type="PlannedStep",
                metadata={
                    "plan_ref": step_ref,
                    "depends_on": list(step.get("depends_on", [])),
                    "candidate_tools": list(step.get("candidate_tools", [])),
                },
            )
            graph.add_edge(stage_id, step_id, EdgeType.INCLUSION)
            if prior_step:
                graph.add_edge(prior_step, step_id, EdgeType.MAINLINE)
            prior_step = step_id
            step_ids.append(step_id)
            step_by_ref[step_ref] = step_id
            selected = [
                name for name in step.get("candidate_tools", [])
                if name in tools_by_name and name != "Finish"
            ]
            step_tools[step_id] = selected
            _attach_tool_capabilities(graph, step_id, selected, tools_by_name)
    _attach_declared_dependencies(graph, plan, step_by_ref)
    final_stage = _ensure_final_stage(graph, task_id, stage_ids, prior_stage)
    final_step = graph.add_node(
        NodeLevel.L3,
        content="Synthesize and return the final answer",
        turn_index=0,
        path=f"{graph.nodes[final_stage].path}.Final",
        status=NodeStatus.PENDING,
        value="Use verified tool observations to answer the complete user query.",
        sub_type="FinalAnswerStep",
        metadata={"plan_ref": "FINAL", "candidate_tools": ["Finish"]},
    )
    graph.add_edge(final_stage, final_step, EdgeType.INCLUSION)
    siblings = graph.mainline_order(final_stage, NodeLevel.L3)
    prior_final = next((node for node in reversed(siblings) if node.node_id != final_step), None)
    if prior_final:
        graph.add_edge(prior_final.node_id, final_step, EdgeType.MAINLINE)
    for planned_step in step_ids:
        graph.add_edge(
            planned_step,
            final_step,
            EdgeType.DEPENDENCY,
            {"reason": "result_required_for_final_answer"},
        )
    step_ids.append(final_step)
    step_by_ref["FINAL"] = final_step
    step_tools[final_step] = ["Finish"]
    return graph, TrajectoryLayout(
        task_id, stage_ids, step_ids, step_by_ref, step_tools, tools_by_name
    )


def _index_tools(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in tools:
        function = raw.get("function", raw)
        if isinstance(function, dict) and function.get("name"):
            result[str(function["name"])] = raw
    return result


def _attach_tool_capabilities(graph, step_id, names, tools_by_name) -> None:
    for index, name in enumerate(names, start=1):
        function = tools_by_name[name].get("function", tools_by_name[name])
        parameters = function.get("parameters", {})
        summary = {
            "description": str(function.get("description", ""))[:320],
            "required": list(parameters.get("required", [])),
            "properties": list(parameters.get("properties", {})),
        }
        node_id = graph.add_node(
            NodeLevel.L4, f"Available API: {name}", 0,
            f"{graph.nodes[step_id].path}.Tool{index}",
            status=NodeStatus.ACTIVE, value=json.dumps(summary, ensure_ascii=False),
            sub_type="ToolCapability", metadata={"tool_name": name},
        )
        graph.add_edge(step_id, node_id, EdgeType.INCLUSION)


def _attach_declared_dependencies(graph, plan, step_by_ref) -> None:
    for stage in plan:
        for step in stage.get("steps", []):
            target = step_by_ref.get(str(step.get("id", "")))
            if not target:
                continue
            for source_ref in step.get("depends_on", []):
                source = step_by_ref.get(str(source_ref))
                if source and source != target:
                    graph.add_edge(source, target, EdgeType.DEPENDENCY, {"reason": "planned_prerequisite"})


def _ensure_final_stage(graph, task_id, stage_ids, prior_stage) -> str:
    if stage_ids:
        return stage_ids[-1]
    stage_id = graph.add_node(
        NodeLevel.L2, "Answer task", 0, "T1.S1",
        status=NodeStatus.PENDING, sub_type="PlannedStage",
    )
    graph.add_edge(task_id, stage_id, EdgeType.INCLUSION)
    stage_ids.append(stage_id)
    return stage_id
