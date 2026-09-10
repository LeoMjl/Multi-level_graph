from __future__ import annotations

import json
from typing import Any

from mlg.graph import EdgeType, NodeLevel, NodeStatus
from mlg.stabletoolbench.ablation import TaskGraphAblation
from mlg.stabletoolbench.dependency_builder import (
    build_dependency_candidates,
    candidate_edge_metadata,
    evaluate_dependency_readiness,
)
from mlg.stabletoolbench.planning import PLAN_PROTOCOL
from mlg.stabletoolbench.progressive_builder import (
    build_coarse_graph,
    expand_final_stage,
    expand_stage as add_expanded_stage,
)
from mlg.stabletoolbench.progressive_planning import fallback_coarse_plan
from mlg.stabletoolbench.trajectory_context import (
    render_flat_history_context,
    render_scheduled_context,
    scheduled_context_node_ids,
)
from mlg.stabletoolbench.trajectory_state import apply_observation_state
from mlg.stabletoolbench.dependency_filter import dependency_context, validate_selection
class ProgressiveToolGraph:
    """Progressively expanded, dependency-aware TaskGraph for one tool task."""
    def __init__(
        self,
        query: str,
        tools: list[dict[str, Any]] | None = None,
        plan: dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        plan_source: str = "model",
        ablation: TaskGraphAblation | None = None,
        dependency_filter=None,
    ) -> None:
        self.ablation = ablation or TaskGraphAblation()
        self.dependency_filter = dependency_filter
        self.tools = list(tools or [])
        self.query = query
        self.plan = self._normalize_plan(plan)
        self.graph, self.layout = build_coarse_graph(query, self.tools, self.plan)
        self.task_id = self.layout.task_id
        self.stage_ids = self.layout.stage_ids
        self.turn = 0
        self.current_stage_id = ""
        self.current_step_id = ""
        self.latest_observation = ""
        self.call_steps: dict[str, str] = {}
        self.expansion_trace: list[dict[str, Any]] = []
        self.dependency_trace: list[dict[str, Any]] = []
        self.state_update_trace: list[dict[str, Any]] = []
        self.l4_writeback_trace: list[dict[str, Any]] = []
        self.protocol = PLAN_PROTOCOL
        self.graph.nodes[self.task_id].metadata.update({
            "protocol": PLAN_PROTOCOL,
            "plan_source": plan_source,
            "ablation": self.ablation.to_dict(),
        })

    @staticmethod
    def _normalize_plan(plan) -> dict[str, Any]:
        if isinstance(plan, dict):
            stages = plan.get("stages", [])
            state = plan.get("global_state", [])
            if not isinstance(stages, list) or not stages:
                return fallback_coarse_plan()
            return {
                "stages": list(stages),
                "global_state": list(state) if isinstance(state, list) else [],
            }
        if isinstance(plan, list):
            return {
                "stages": [{
                    key: value for key, value in stage.items()
                    if key in {"id", "name", "goal", "depends_on"}
                } for stage in plan],
                "global_state": [],
            }
        return fallback_coarse_plan()

    def stage_to_expand(self) -> str:
        """Activate the next dependency-ready L2 and return it if L3 is absent."""
        self._refresh_stage_states()
        for stage_id in self.layout.business_stage_ids:
            stage = self.graph.nodes[stage_id]
            if stage.status == NodeStatus.ACTIVE:
                self.current_stage_id = stage_id
                return stage_id if not stage.metadata.get("expanded") else ""
        for stage_id in self.layout.business_stage_ids:
            stage = self.graph.nodes[stage_id]
            if stage.status != NodeStatus.PENDING:
                continue
            readiness = self._dependency_readiness(stage_id)
            if readiness == "blocked":
                self._drop_blocked_node(stage_id)
                continue
            if readiness == "ready":
                stage.status = NodeStatus.ACTIVE
                self.current_stage_id = stage_id
                return stage_id if not stage.metadata.get("expanded") else ""
        business_terminal = all(
            self.graph.nodes[item].metadata.get("expansion_valid") is not False
            and self.graph.nodes[item].status in {
                NodeStatus.DONE, NodeStatus.DROPPED,
            }
            for item in self.layout.business_stage_ids
        )
        final = self.graph.nodes[self.layout.final_stage_id]
        if business_terminal and final.status in {NodeStatus.PENDING, NodeStatus.ACTIVE}:
            final.status = NodeStatus.ACTIVE
            self.current_stage_id = final.node_id
            if not final.metadata.get("expanded"):
                self.expand_stage(final.node_id)
            return ""
        return ""

    def expand_stage(
        self,
        stage_id_or_ref: str,
        steps: list[dict[str, Any]] | None = None,
        *,
        source: str = "model",
        raw_response: str = "",
        tokens: int = 0,
    ) -> list[str]:
        stage_id = self.layout.stage_by_ref.get(stage_id_or_ref, stage_id_or_ref)
        self.turn += 1
        if stage_id == self.layout.final_stage_id:
            created = [expand_final_stage(self.graph, self.layout)]
            source = "system"
        else:
            created = add_expanded_stage(
                self.graph, self.layout, stage_id, list(steps or []),
            )
            for node_id in created:
                self.graph.nodes[node_id].turn_index = self.turn
                for child in self.graph.children(node_id, NodeLevel.L4):
                    child.turn_index = self.turn
            self._attach_expansion_dependencies(created)
            stage = self.graph.nodes[stage_id]
            stage.metadata["expansion_valid"] = bool(created)
            if not created:
                step_ref = f"{stage.metadata.get('plan_ref', stage_id)}.FAILED"
                placeholder_id = self.graph.add_node(
                    NodeLevel.L3,
                    "Stage expansion produced no executable step",
                    self.turn,
                    f"{stage.path}.{step_ref}",
                    status=NodeStatus.DROPPED,
                    value="No model-proposed tool matched an available routed tool.",
                    sub_type="ExpandedStep",
                    metadata={
                        "plan_ref": step_ref,
                        "candidate_tools": [],
                        "executable": False,
                        "audit_only": True,
                        "drop_reason": "stage expansion produced no executable step",
                    },
                )
                self.graph.add_edge(stage_id, placeholder_id, EdgeType.INCLUSION)
                self.layout.step_ids.append(placeholder_id)
                self.layout.step_by_ref[step_ref] = placeholder_id
                self.layout.step_tools[placeholder_id] = []
                created = [placeholder_id]
                stage.status = NodeStatus.DROPPED
                stage.metadata["drop_reason"] = (
                    "stage expansion produced no executable step"
                )
        for node_id in created:
            self.graph.nodes[node_id].turn_index = self.turn
            for child in self.graph.children(node_id, NodeLevel.L4):
                child.turn_index = self.turn
        trace = {
            "stage_id": stage_id,
            "stage_ref": self.graph.nodes[stage_id].metadata.get("plan_ref"),
            "source": source,
            "step_ids": created,
            "tokens": tokens,
            "raw_response": str(raw_response)[:6000],
        }
        self.expansion_trace.append(trace)
        return created

    def ensure_audit_final_step(self) -> str:
        """Materialize a pending final step when execution terminated early."""
        existing = self.graph.children(self.layout.final_stage_id, NodeLevel.L3)
        if existing:
            return existing[0].node_id
        final_stage = self.graph.nodes[self.layout.final_stage_id]
        step_id = expand_final_stage(self.graph, self.layout)
        final_step = self.graph.nodes[step_id]
        final_step.metadata.update({
            "materialized_for_audit": True,
            "execution_reached": False,
        })
        self.expansion_trace.append({
            "stage_id": final_stage.node_id,
            "stage_ref": final_stage.metadata.get("plan_ref"),
            "source": "system_audit",
            "step_ids": [step_id],
            "tokens": 0,
            "raw_response": "",
            "materialized_for_audit": True,
        })
        return step_id

    def schedule(self) -> str:
        """Activate one dependency-ready L3; unexpanded L2s remain a hard boundary."""
        self._refresh_stage_states()
        if not any(
            self.graph.nodes[item].status == NodeStatus.ACTIVE
            for item in self.stage_ids
        ):
            self.stage_to_expand()
        for stage_id in self.stage_ids:
            stage = self.graph.nodes[stage_id]
            if stage.status != NodeStatus.ACTIVE or not stage.metadata.get("expanded"):
                continue
            steps = self.graph.mainline_order(stage_id, NodeLevel.L3)
            active = [step for step in steps if step.status == NodeStatus.ACTIVE]
            if active:
                self.current_stage_id = stage_id
                self.current_step_id = active[0].node_id
                return self.current_step_id
            for step in steps:
                if step.status != NodeStatus.PENDING:
                    continue
                readiness = self._dependency_readiness(step.node_id)
                if readiness == "blocked":
                    self._drop_blocked_node(step.node_id)
                    continue
                if readiness == "ready":
                    step.status = NodeStatus.ACTIVE
                    self.current_stage_id = stage_id
                    self.current_step_id = step.node_id
                    return step.node_id
        self._refresh_stage_states()
        self.current_step_id = ""
        if all(
            self.graph.nodes[item].status in {NodeStatus.DONE, NodeStatus.DROPPED}
            for item in self.stage_ids
        ):
            self.graph.nodes[self.task_id].status = NodeStatus.DONE
        return ""

    def _refresh_stage_states(self) -> None:
        for stage_id in self.stage_ids:
            stage = self.graph.nodes[stage_id]
            if stage.status != NodeStatus.ACTIVE or not stage.metadata.get("expanded"):
                continue
            steps = self.graph.children(stage_id, NodeLevel.L3)
            if steps and all(
                item.status in {NodeStatus.DONE, NodeStatus.DROPPED} for item in steps
            ):
                dropped = [
                    item.metadata.get("plan_ref", item.node_id)
                    for item in steps if item.status == NodeStatus.DROPPED
                ]
                stage.status = (
                    NodeStatus.DROPPED if dropped else NodeStatus.DONE
                )
                if dropped:
                    stage.metadata["dropped_steps"] = dropped

    def _dependency_readiness(self, node_id: str) -> str:
        readiness = evaluate_dependency_readiness(self.graph, node_id)
        if readiness.blocked:
            return "blocked"
        if readiness.waiting:
            return "waiting"
        return "ready"

    def _drop_blocked_node(self, node_id: str) -> None:
        node = self.graph.nodes[node_id]
        node.status = NodeStatus.DROPPED
        node.metadata["drop_reason"] = "hard dependency was dropped"

    def drop_step(self, step_id: str, reason: str) -> None:
        step = self.graph.nodes[step_id]
        step.status = NodeStatus.DROPPED
        step.metadata["drop_reason"] = str(reason)[:600]

    def _attach_expansion_dependencies(self, step_ids: list[str], *, new_state_ids=()) -> None:
        for step_id in step_ids:
            step = self.graph.nodes[step_id]
            for source_ref in ([] if new_state_ids else step.metadata.get("references", [])):
                source = self.layout.step_by_ref.get(str(source_ref), str(source_ref))
                if source in self.graph.nodes and source != step_id:
                    source_node = self.graph.nodes[source]
                    source_node.metadata["ref_count"] = int(
                        source_node.metadata.get("ref_count", 0)
                    ) + 1
            hard_ids = []
            unresolved = []
            for source_ref in step.metadata.get("depends_on", []):
                source = self.layout.step_by_ref.get(str(source_ref))
                if source and source != step_id:
                    hard_ids.append(source)
                else:
                    unresolved.append(str(source_ref))
            if unresolved:
                step.status = NodeStatus.DROPPED
                step.metadata["unresolved_dependencies"] = unresolved
                step.metadata["drop_reason"] = "unresolved hard dependency"
                continue
            if self.ablation.hard_dependencies_only:
                for source_id in hard_ids:
                    if self._has_dependency_edge(source_id, step_id):
                        continue
                    self.graph.add_edge(
                        source_id,
                        step_id,
                        EdgeType.DEPENDENCY,
                        {
                            "reason": "declared_step_prerequisite",
                            "relation_type": "prerequisite",
                            "sources": ["hard"],
                            "hard": True,
                            "blocking": True,
                        },
                    )
                self.dependency_trace.append({
                    "target_id": step_id,
                    "policy": "explicit_hard_only",
                    "hard": list(hard_ids),
                    "Vc": [],
                    "Cdep": [],
                    "Vr": [],
                    "fusion_pool": list(hard_ids),
                    "selected": list(hard_ids),
                    "semantic_backend": "disabled_by_ablation",
                })
                continue
            candidates = build_dependency_candidates(
                self.graph, step_id, hard_dependency_ids=hard_ids,
                new_state_ids=tuple(dict.fromkeys([
                    *new_state_ids,
                    *(edge.source_id for edge in self.graph.edges
                      if new_state_ids and edge.target_id == step_id
                      and edge.metadata.get("selection") == "agent_dependency_filter"
                      and self.graph.nodes[edge.source_id].level == NodeLevel.L4
                      and self.graph.nodes[edge.source_id].status != NodeStatus.DROPPED),
                ])),
            )
            context = dependency_context(self.graph, step_id, candidates.fused_candidates)
            if not candidates.fused_candidates:
                retained, filter_trace = [], {"status": "empty", "calls": 0, "tokens": 0}
            elif self.dependency_filter is None:
                raise RuntimeError("Full TaskGraph requires the agent dependency-filter callback")
            else:
                retained, filter_trace = self.dependency_filter(context)
            if retained is not None:
                retained = validate_selection({"retain": retained}, context)
                if new_state_ids:
                    self.graph.edges[:] = [edge for edge in self.graph.edges
                                           if not (edge.target_id == step_id
                                                   and edge.metadata.get("selection")
                                                   == "agent_dependency_filter")]
            else:
                self.drop_step(step_id, "invalid agent dependency-filter response")
            for decision in retained or []:
                source_id = decision["node_id"]
                diagnostic = candidates.diagnostic(source_id)
                if self._has_dependency_edge(source_id, step_id):
                    continue
                metadata = candidate_edge_metadata(
                    diagnostic, relation_type=decision["relation_type"],
                    blocking=diagnostic.hard or decision["relation_type"] == "prerequisite",
                )
                metadata.update({
                    "rationale": decision["rationale"], "evidence": decision["evidence"],
                    "source": {"node_id": source_id,
                               "provenance": dict(self.graph.nodes[source_id].metadata)},
                    "selection": "agent_dependency_filter",
                })
                self.graph.add_edge(
                    source_id,
                    step_id,
                    EdgeType.DEPENDENCY,
                    metadata,
                )
            self.dependency_trace.append({
                "target_id": step_id,
                "policy": "three_channel_union_then_agent_filter",
                "trigger": "state_writeback" if new_state_ids else "stage_expansion",
                "hard": list(candidates.hard_dependencies),
                "Vc": list(candidates.structural_candidates),
                "Cdep": list(candidates.semantic_candidates),
                "Vr": list(candidates.reference_candidates),
                "fusion_pool": list(candidates.fusion_pool),
                "candidates": list(candidates.fused_candidates),
                "selected": [item["node_id"] for item in retained or []],
                "filter": filter_trace,
                "context": context,
                "semantic_backend": candidates.semantic_backend,
            })

    def _has_dependency_edge(self, source_id: str, target_id: str) -> bool:
        return any(
            edge.edge_type == EdgeType.DEPENDENCY
            and edge.source_id == source_id
            and edge.target_id == target_id
            for edge in self.graph.edges
        )

    def routed_tools(self, step_id: str | None = None) -> list[dict[str, Any]]:
        """Expose only the exact tool selected for the scheduled step."""
        selected_step = step_id or self.current_step_id or self.schedule()
        names = self.layout.step_tools.get(selected_step, [])
        if names == ["Finish"]:
            finish = self.layout.tools_by_name.get("Finish")
            return [finish] if finish else []
        if names:
            return [self.layout.tools_by_name[name] for name in names]
        return []

    def add_thought(self, content: str, step_id: str | None = None) -> str:
        return self._add_trace(
            step_id or self.current_step_id or self.schedule(),
            "Thought",
            "Decision rationale",
            content,
            NodeStatus.DONE,
        )

    def add_call(
        self,
        name: str,
        arguments: str,
        step_id: str | None = None,
    ) -> str:
        selected_step = step_id or self.current_step_id or self.schedule()
        call_id = self._add_trace(
            selected_step, "ToolCall", f"Call {name}", name, NodeStatus.ACTIVE,
            metadata={"tool_name": name, "arguments": arguments},
        )
        self.call_steps[call_id] = selected_step
        for child in self.graph.children(selected_step, NodeLevel.L4):
            if (
                child.sub_type == "ToolCapability"
                and child.metadata.get("tool_name") == name
            ):
                self.graph.add_edge(
                    child.node_id, call_id, EdgeType.DEPENDENCY,
                    {"reason": "scheduled_tool_capability"},
                )
        if self.latest_observation:
            self.graph.add_edge(
                self.latest_observation, call_id, EdgeType.DEPENDENCY,
                {"reason": "prior_observation_informs_call"},
            )
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            parsed = {"raw": arguments}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        for key, value in parsed.items():
            argument_id = self._add_trace(
                selected_step, "ToolArgument", str(key),
                json.dumps(value, ensure_ascii=False), NodeStatus.DONE,
            )
            self.graph.add_edge(
                argument_id, call_id, EdgeType.DEPENDENCY,
                {"reason": "argument_for_tool_call"},
            )
        return call_id

    def add_observation(
        self,
        call_id: str,
        observation: str,
        status: int,
        *,
        validation_outcome: str = "",
        validation_reason: str = "",
        state_updates: list[dict[str, str]] | None = None,
    ) -> str:
        step_id = self.call_steps[call_id]
        subtype = "ToolObservation" if status in {0, 3} else "ToolError"
        observation_id = self._add_trace(
            step_id, subtype, "Execution result", observation[:1800],
            NodeStatus.DONE, metadata={
                "status_code": status,
                "validation_outcome": validation_outcome,
                "validation_reason": validation_reason,
            },
        )
        self.graph.add_edge(
            call_id, observation_id, EdgeType.DEPENDENCY,
            {"reason": "tool_call_produced_observation", "status": status},
        )
        apply_observation_state(
            self.graph, call_id, step_id, status,
            validation_outcome, validation_reason,
        )
        self._apply_state_updates(
            step_id,
            observation_id,
            observation,
            list(state_updates or []),
        )
        self.latest_observation = observation_id
        return observation_id

    def _apply_state_updates(
        self,
        step_id: str,
        observation_id: str,
        observation: str,
        updates: list[dict[str, str]],
    ) -> None:
        if not self.ablation.materialize_structured_l4:
            for update in updates[:16]:
                key = str(update.get("key", "")).strip()[:120]
                value = str(update.get("value", "")).strip()[:800]
                evidence = str(update.get("evidence", "")).strip()[:800]
                scope = str(update.get("scope", "")).lower()
                grounded = bool(
                    key and value and evidence
                    and evidence in observation
                    and value.casefold() in observation.casefold()
                    and scope in {"task", "stage", "step"}
                )
                self.state_update_trace.append({
                    "step_id": step_id,
                    "key": key,
                    "scope": scope,
                    "grounded": grounded,
                    "accepted": False,
                    "materialized": False,
                    "reason": "structured L4 writeback disabled by ablation",
                })
            return
        new_state_ids = []
        for update in updates[:16]:
            key = str(update.get("key", "")).strip()[:120]
            value = str(update.get("value", "")).strip()[:800]
            evidence = str(update.get("evidence", "")).strip()[:800]
            scope = str(update.get("scope", "")).lower()
            accepted = bool(
                key and value and evidence
                and evidence in observation
                and value.casefold() in observation.casefold()
                and scope in {"task", "stage", "step"}
            )
            trace: dict[str, Any] = {
                "step_id": step_id,
                "key": key,
                "scope": scope,
                "accepted": accepted,
            }
            if not accepted:
                trace["reason"] = "update was not exactly grounded in observation"
                self.state_update_trace.append(trace)
                continue
            parent_id = {
                "task": self.task_id,
                "stage": self._stage_parent(step_id),
                "step": step_id,
            }[scope]
            previous = next((
                node for node in reversed(list(self.graph.nodes.values()))
                if node.level == NodeLevel.L4
                and node.metadata.get("state_key") == key
                and node.metadata.get("scope_parent") == parent_id
                and node.status == NodeStatus.ACTIVE
            ), None)
            if previous:
                previous.status = NodeStatus.DONE
            self.turn += 1
            state_id = self.graph.add_node(
                NodeLevel.L4,
                f"State: {key}",
                self.turn,
                f"{self.graph.nodes[parent_id].path}.State{self.turn}",
                status=NodeStatus.ACTIVE,
                value=value,
                sub_type="StructuredState",
                is_global=scope == "task",
                metadata={
                    "state_key": key,
                    "scope": scope,
                    "scope_parent": parent_id,
                    "evidence": evidence,
                    "source_observation": observation_id,
                },
            )
            self.graph.add_edge(parent_id, state_id, EdgeType.INCLUSION)
            new_state_ids.append(state_id)
            self.graph.add_edge(
                observation_id,
                state_id,
                EdgeType.DEPENDENCY,
                {"reason": "observation_grounded_state_update", "hard": False,
                 "blocking": False},
            )
            if previous:
                self.graph.add_edge(
                    previous.node_id,
                    state_id,
                    EdgeType.DEPENDENCY,
                    {"reason": "state_refresh", "hard": False, "blocking": False},
                )
            trace.update({"state_node_id": state_id, "value": value})
            self.state_update_trace.append(trace)
            self.l4_writeback_trace.append({
                "node_id": state_id,
                "step_id": step_id,
                "sub_type": "StructuredState",
                "source_observation": observation_id,
            })
        if new_state_ids and not self.ablation.hard_dependencies_only:
            stage_id = self._stage_parent(step_id)
            pending_ids = [
                node.node_id for node in self.graph.children(stage_id, NodeLevel.L3)
                if node.node_id != step_id and node.status == NodeStatus.PENDING
            ]
            self._attach_expansion_dependencies(pending_ids, new_state_ids=new_state_ids)

    def _stage_parent(self, step_id: str) -> str:
        for edge in self.graph.edges:
            if edge.edge_type != EdgeType.INCLUSION or edge.target_id != step_id:
                continue
            if self.graph.nodes[edge.source_id].level == NodeLevel.L2:
                return edge.source_id
        return self.current_stage_id

    def render_for_expansion(
        self,
        stage_id_or_ref: str,
        max_chars: int = 12000,
    ) -> str:
        """Render stable state available before the activated stage creates L3."""
        stage_id = self.layout.stage_by_ref.get(stage_id_or_ref, stage_id_or_ref)
        stage = self.graph.nodes[stage_id]
        blocks = [
            "TaskGraph progressive expansion context\n",
            f"TASK={self.query}\n",
            f"CURRENT_STAGE={stage.metadata.get('plan_ref')}: {stage.content}\n",
            f"CURRENT_STAGE_GOAL={stage.value}\n",
            "Observable prior state:\n",
        ]
        selected = []
        for node in self.graph.nodes.values():
            if node.node_id in {self.task_id, stage_id}:
                continue
            if node.sub_type in {
                "GlobalConstraint", "StructuredState", "ToolObservation", "ToolError",
            } or (
                node.level in {NodeLevel.L2, NodeLevel.L3}
                and node.status in {NodeStatus.DONE, NodeStatus.DROPPED}
            ):
                selected.append(node)
        selected.sort(key=lambda node: (
            node.sub_type not in {"StructuredState", "ToolObservation", "ToolError"},
            -node.turn_index,
            node.path,
        ))
        used = sum(len(item) for item in blocks)
        for node in selected:
            ref = node.metadata.get("plan_ref", node.node_id)
            value = str(node.value or node.content).replace("\n", " ")[:1600]
            line = (
                f"- [{node.node_id} ref={ref} {node.level.value}/"
                f"{node.sub_type or 'Node'} {node.status.value}] {value}\n"
            )
            if used + len(line) > max_chars:
                break
            blocks.append(line)
            used += len(line)
        return "".join(blocks)

    def context_node_ids(self, step_id: str | None = None) -> list[str]:
        selected_step = step_id or self.current_step_id or self.schedule()
        if not selected_step:
            return []
        return scheduled_context_node_ids(
            self.graph, self.task_id, "", selected_step
        )

    def render(
        self,
        max_chars: int = 6000,
        keep_node_ids: set[str] | None = None,
    ) -> str:
        step_id = self.current_step_id or self.schedule()
        if not step_id:
            return "TaskGraph has no Pending or Active execution step."
        if self.ablation.flat_action_context:
            return render_flat_history_context(
                self.graph,
                self.task_id,
                step_id,
                max_chars=max_chars,
            )
        return self.render_anchored(
            max_chars=max_chars,
            keep_node_ids=keep_node_ids,
        )

    def render_anchored(
        self,
        max_chars: int = 6000,
        keep_node_ids: set[str] | None = None,
    ) -> str:
        """Render the normal step-anchored graph for non-action cognition."""
        step_id = self.current_step_id or self.schedule()
        if not step_id:
            return "TaskGraph has no Pending or Active execution step."
        return render_scheduled_context(
            self.graph,
            self.task_id,
            step_id,
            max_chars=max_chars,
            keep_node_ids=keep_node_ids,
        )

    def _add_trace(
        self,
        step_id: str,
        sub_type: str,
        content: str,
        value: str,
        status: NodeStatus,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.turn += 1
        node_id = self.graph.add_node(
            NodeLevel.L4, content, self.turn,
            f"{self.graph.nodes[step_id].path}.Trace{self.turn}",
            status=status, value=value, sub_type=sub_type,
            metadata=metadata,
        )
        self.graph.add_edge(step_id, node_id, EdgeType.INCLUSION)
        if sub_type in {
            "ToolCall", "ToolArgument", "ToolObservation", "ToolError",
        }:
            self.l4_writeback_trace.append({
                "node_id": node_id,
                "step_id": step_id,
                "sub_type": sub_type,
            })
        return node_id
