from __future__ import annotations

from mlg.graph import NodeStatus, TaskGraph


def apply_observation_state(
    graph: TaskGraph,
    call_id: str,
    step_id: str,
    status: int,
    validation_outcome: str,
    validation_reason: str,
) -> None:
    call = graph.nodes[call_id]
    call.status = NodeStatus.DONE if status in {0, 3} else NodeStatus.DROPPED
    step = graph.nodes[step_id]
    if status == 3 or (status == 0 and validation_outcome == "done"):
        step.status = NodeStatus.DONE
    elif status == 4:
        step.status = NodeStatus.DROPPED
    else:
        step.status = NodeStatus.ACTIVE
        step.metadata["failed_attempts"] = int(
            step.metadata.get("failed_attempts", 0)
        ) + 1
    if status == 0:
        step.metadata["last_validation"] = {
            "outcome": validation_outcome,
            "reason": validation_reason,
        }
