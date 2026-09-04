from __future__ import annotations

from mlg.graph import EdgeType, NodeLevel, NodeStatus, TaskGraph


def render_flat_history_context(
    graph: TaskGraph,
    task_id: str,
    step_id: str,
    *,
    max_chars: int = 6000,
) -> str:
    """Render a relation-free chronological history with the same action budget."""
    current = graph.nodes[step_id]
    routed_tools = list(current.metadata.get("candidate_tools", []))
    failed_attempts = int(current.metadata.get("failed_attempts", 0))
    header = (
        "Chronological flat execution history\n"
        f"TASK={graph.nodes[task_id].value}\n"
        f"CURRENT_STEP={current.metadata.get('plan_ref', step_id)}: "
        f"{current.content}\n"
        f"CURRENT_GOAL={current.value}\n"
        f"ROUTED_TOOLS={routed_tools or ['none: planning failed']}\n"
        f"FAILED_ATTEMPTS={failed_attempts}\n"
        "Execute only CURRENT_STEP. Do not infer graph edges or redo completed "
        "steps. Use the recent events below in their original chronological "
        "order. Finish is allowed only on the final-answer step.\n"
        "Recent events (oldest to newest):\n"
    )
    excluded = {task_id, step_id}
    allowed_types = {
        "CoarseStage", "FinalizationStage", "ExpandedStep", "FinalAnswerStep",
        "GlobalConstraint", "StructuredState", "Thought", "ToolCall",
        "ToolArgument", "ToolObservation", "ToolError",
    }
    indexed = [
        (index, node)
        for index, node in enumerate(graph.nodes.values())
        if node.node_id not in excluded
        and node.sub_type in allowed_types
        and node.status != NodeStatus.PENDING
    ]
    indexed.sort(key=lambda item: (item[1].turn_index, item[0]))
    lines = [_flat_event_line(node) for _, node in indexed]
    budget = max(0, max_chars - len(header))
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        if used + len(line) > budget:
            continue
        kept.append(line)
        used += len(line)
    return (header + "".join(reversed(kept)))[:max_chars]


def _flat_event_line(node) -> str:
    ref = node.metadata.get("plan_ref", node.node_id)
    value = str(node.value or node.content).replace("\n", " ")
    limit = 1400 if node.sub_type in {"ToolObservation", "ToolError"} else 600
    return (
        f"- turn={node.turn_index} ref={ref} type={node.sub_type or 'Node'} "
        f"status={node.status.value}: {value[:limit]}\n"
    )


def render_scheduled_context(
    graph: TaskGraph,
    task_id: str,
    step_id: str,
    *,
    max_chars: int = 6000,
    keep_node_ids: set[str] | None = None,
) -> str:
    """Pack an anchored execution subgraph without tail-truncating its task path."""
    stage_id = _parent(graph, step_id, NodeLevel.L2)
    current = graph.nodes[step_id]
    routed_tools = list(current.metadata.get("candidate_tools", []))
    failed_attempts = int(current.metadata.get("failed_attempts", 0))
    prior_calls = [
        (
            str(node.metadata.get("tool_name", "")),
            str(node.metadata.get("arguments", ""))[:320],
        )
        for node in graph.children(step_id, NodeLevel.L4)
        if node.sub_type == "ToolCall"
    ][-3:]
    header = (
        "TaskGraph scheduled execution context\n"
        f"CURRENT_STEP={current.metadata.get('plan_ref', step_id)}: {current.content}\n"
        f"CURRENT_GOAL={current.value}\n"
        f"ROUTED_TOOLS={routed_tools or ['none: planning failed']}\n"
        f"FAILED_ATTEMPTS={failed_attempts}\n"
        f"PRIOR_CALLS={prior_calls}\n"
        "Execute only CURRENT_STEP. Use dependency observations as prerequisites; "
        "do not redo Done steps. If FAILED_ATTEMPTS is positive, returning a prior "
        "tool-and-arguments pair is forbidden. For a not-found search, try a shorter "
        "distinctive query or a schema-example-shaped variant derived from the user "
        "entity; never substitute an unrelated example entity. For other failures, "
        "make a schema-valid correction using the observed error. Finish is allowed "
        "only on the final-answer step.\n"
    )
    priority_ids = scheduled_context_node_ids(graph, task_id, stage_id, step_id)
    if keep_node_ids is not None:
        required = {
            task_id,
            stage_id,
            step_id,
            *(
                node.node_id for node in graph.children(task_id, NodeLevel.L4)
                if node.is_global and node.status != NodeStatus.DROPPED
            ),
        }
        priority_ids = [
            node_id for node_id in priority_ids
            if node_id in keep_node_ids or node_id in required
        ]
    included = set(priority_ids)
    blocks = [header, "Nodes:\n"]
    used = sum(len(item) for item in blocks)
    for node_id in priority_ids:
        line = _node_line(graph, node_id)
        if used + len(line) > max_chars:
            remaining = max_chars - used
            if remaining > 80:
                blocks.append(line[:remaining - 4] + "...\n")
            break
        blocks.append(line)
        used += len(line)
    edge_lines = []
    for edge in graph.edges:
        if edge.source_id not in included or edge.target_id not in included:
            continue
        reason = str(edge.metadata.get("reason", ""))
        edge_lines.append(
            f"- {edge.source_id} -{edge.edge_type.value}"
            f"{f'({reason})' if reason else ''}-> {edge.target_id}\n"
        )
    if edge_lines and used < max_chars:
        label = "Relations:\n"
        blocks.append(label)
        used += len(label)
        for line in edge_lines:
            if used + len(line) > max_chars:
                break
            blocks.append(line)
            used += len(line)
    return "".join(blocks)[:max_chars]


def scheduled_context_node_ids(
    graph: TaskGraph,
    task_id: str,
    stage_id: str,
    step_id: str,
) -> list[str]:
    stage_id = stage_id or _parent(graph, step_id, NodeLevel.L2)
    ordered = [task_id]
    if stage_id:
        ordered.append(stage_id)
    ordered.append(step_id)
    task_state = [
        node for node in graph.children(task_id, NodeLevel.L4)
        if node.is_global and node.status != NodeStatus.DROPPED
    ]
    ordered.extend(node.node_id for node in _trace_first(task_state))
    if stage_id:
        stage_state = [
            node for node in graph.children(stage_id, NodeLevel.L4)
            if node.sub_type == "StructuredState" and node.status != NodeStatus.DROPPED
        ]
        ordered.extend(node.node_id for node in _trace_first(stage_state))
    current_children = graph.children(step_id, NodeLevel.L4)
    ordered.extend(node.node_id for node in _trace_first(current_children))

    dependencies = graph.dependencies(step_id)
    dependencies.sort(key=lambda node: (node.turn_index, node.path), reverse=True)
    for dependency in dependencies:
        ordered.append(dependency.node_id)
        children = graph.children(dependency.node_id, NodeLevel.L4)
        ordered.extend(node.node_id for node in _trace_first(children))

    predecessor, successor = _mainline_neighbors(graph, step_id)
    for neighbor in (predecessor, successor):
        if neighbor:
            ordered.append(neighbor)
            children = graph.children(neighbor, NodeLevel.L4)
            ordered.extend(node.node_id for node in _trace_first(children))
    return _deduplicate(ordered)


def _parent(graph: TaskGraph, child_id: str, level: NodeLevel) -> str:
    for edge in graph.edges:
        if edge.edge_type != EdgeType.INCLUSION or edge.target_id != child_id:
            continue
        parent = graph.nodes[edge.source_id]
        if parent.level == level:
            return parent.node_id
    return ""


def _mainline_neighbors(graph: TaskGraph, node_id: str) -> tuple[str, str]:
    predecessor = ""
    successor = ""
    for edge in graph.edges:
        if edge.edge_type != EdgeType.MAINLINE:
            continue
        if edge.target_id == node_id:
            predecessor = edge.source_id
        elif edge.source_id == node_id:
            successor = edge.target_id
    return predecessor, successor


def _trace_first(nodes):
    return sorted(
        nodes,
        key=lambda node: (
            node.sub_type not in {"ToolObservation", "ToolError", "ToolCall"},
            -node.turn_index,
            node.path,
        ),
    )


def _node_line(graph: TaskGraph, node_id: str) -> str:
    node = graph.nodes[node_id]
    value = str(node.value or node.content).replace("\n", " ")
    limit = 1400 if node.sub_type in {"ToolObservation", "ToolError"} else 700
    return (
        f"- [{node.node_id} {node.level.value}/{node.sub_type or 'Node'} "
        f"{node.status.value}] {node.content}: {value[:limit]}\n"
    )


def _deduplicate(items: list[str]) -> list[str]:
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]
