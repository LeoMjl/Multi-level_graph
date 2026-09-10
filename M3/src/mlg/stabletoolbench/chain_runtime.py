from __future__ import annotations

from mlg.stabletoolbench.chain_cognition import (
    request_context_pruning,
    request_plan,
    request_result_validation,
    request_stage_expansion,
)
from mlg.stabletoolbench.planning import PLAN_PROTOCOL
from mlg.stabletoolbench.dependency_filter import (
    DEPENDENCY_FILTER_PROTOCOL, request_dependency_filter,
)


class StateDrivenCognitionMixin:
    def _plan_task(self):
        plan, source, total_tokens, self.planner_raw, self.planner_calls = request_plan(
            self.llm,
            self.io_func.input_description,
            self.io_func.functions,
            self.process_id,
            self.ablation.sampling_seed,
        )
        self.total_tokens += total_tokens
        self.query_count += 1
        self.planner_tokens = total_tokens
        self.stage_expander_tokens = 0
        self.stage_expander_calls = 0
        self.pruner_tokens = 0
        self.pruning_trace = []
        self.validator_tokens = 0
        self.validation_trace = []
        self.action_routing_trace = []
        self.dependency_filter_tokens = 0
        self.dependency_filter_calls = 0
        return plan, source

    def _filter_dependencies(self, context):
        retained, trace = request_dependency_filter(
            self.llm, context, self.process_id, self.ablation.sampling_seed,
        )
        self.total_tokens += trace["tokens"]
        self.query_count += trace["calls"]
        self.dependency_filter_tokens += trace["tokens"]
        self.dependency_filter_calls += trace["calls"]
        return retained, trace

    def _expand_stage(self, stage_id: str, *, phase: str = "online") -> list[str]:
        stage_node = self.trajectory.graph.nodes[stage_id]
        if stage_id == self.trajectory.layout.final_stage_id:
            created = self.trajectory.expand_stage(stage_id)
            self.trajectory.expansion_trace[-1]["phase"] = phase
            return created
        stage = {
            "id": stage_node.metadata.get("plan_ref", stage_id),
            "name": stage_node.content,
            "goal": stage_node.value,
            "depends_on": list(stage_node.metadata.get("depends_on", [])),
        }
        steps, source, tokens, raw, calls = request_stage_expansion(
            self.llm,
            self.io_func.input_description,
            stage,
            self.trajectory.render_for_expansion(stage_id),
            self.io_func.functions,
            self.process_id,
            self.ablation.sampling_seed,
        )
        self.total_tokens += tokens
        self.query_count += calls
        self.stage_expander_tokens += tokens
        self.stage_expander_calls += calls
        created = self.trajectory.expand_stage(
            stage_id,
            steps,
            source=source,
            raw_response=raw,
            tokens=tokens,
        )
        self.trajectory.expansion_trace[-1]["phase"] = phase
        return created

    def _bounded_tool_calls(self, message: dict, step_id: str):
        returned = list(message.get("tool_calls") or [])
        executed = returned[:1]
        message["tool_calls"] = executed
        self.action_routing_trace.append({
            "step_id": step_id,
            "routed_tool_names": [
                tool["function"]["name"]
                for tool in self.trajectory.routed_tools(step_id)
            ],
            "returned_tool_call_count": len(returned),
            "executed_tool_call_count": len(executed),
            "truncated_to_first": len(returned) > 1,
        })
        return executed

    def _state_driven_artifacts(self):
        self.trajectory.ensure_audit_final_step()
        return {
            "mlg_protocol": PLAN_PROTOCOL,
            "mlg_ablation": self.ablation.to_dict(),
            "mlg_coarse_plan": {
                "source": self.plan_source,
                "calls": self.planner_calls,
                "planner_tokens": self.planner_tokens,
                "raw_response": self.planner_raw,
                "stages": self.trajectory.plan.get("stages", []),
                "global_state": self.trajectory.plan.get("global_state", []),
            },
            "mlg_stage_expansion": {
                "calls": self.stage_expander_calls,
                "tokens": self.stage_expander_tokens,
                "trace": self.trajectory.expansion_trace,
            },
            "mlg_dependency_construction": {
                "protocol": DEPENDENCY_FILTER_PROTOCOL,
                "calls": self.dependency_filter_calls,
                "tokens": self.dependency_filter_tokens,
                "events": len(self.trajectory.dependency_trace),
                "trace": self.trajectory.dependency_trace,
            },
            "mlg_state_updates": {
                "calls": len(self.trajectory.state_update_trace),
                "trace": self.trajectory.state_update_trace,
                "writeback_trace": self.trajectory.l4_writeback_trace,
            },
            "mlg_context_pruning": {
                "calls": len(self.pruning_trace),
                "tokens": self.pruner_tokens,
                "trace": self.pruning_trace,
            },
            "mlg_action_routing": {
                "calls": len(self.action_routing_trace),
                "trace": self.action_routing_trace,
            },
            "mlg_result_validation": {
                "calls": sum(
                    int(item.get("calls", 1)) for item in self.validation_trace
                ),
                "tokens": self.validator_tokens,
                "trace": self.validation_trace,
            },
            "mlg_observable_graph": self.trajectory.graph.to_dict(),
        }

    def _prune_context(self, step_id: str):
        if self.ablation.flat_action_context:
            packed = self.trajectory.render(
                max_chars=self.ablation.action_context_chars,
            )
            self.pruning_trace.append({
                "step_id": step_id,
                "status": "disabled_by_flat_context_ablation",
                "candidate_count": 0,
                "kept_node_ids": [],
                "tokens": 0,
                "raw_response": "",
                "packed_context": packed,
                "packed_chars": len(packed),
                "context_mode": "chronological_flat_history",
            })
            return set()
        allowed_ids = self.trajectory.context_node_ids(step_id)
        candidate = self.trajectory.render(max_chars=24000)
        keep_ids, trace, total_tokens = request_context_pruning(
            self.llm,
            candidate,
            allowed_ids,
            self.process_id,
            self.ablation.sampling_seed,
        )
        self.total_tokens += total_tokens
        self.query_count += 1
        self.pruner_tokens += total_tokens
        trace["step_id"] = step_id
        if keep_ids is not None:
            trace["packed_context"] = self.trajectory.render(
                max_chars=6000,
                keep_node_ids=keep_ids,
            )
            trace["packed_chars"] = len(trace["packed_context"])
        self.pruning_trace.append(trace)
        return keep_ids

    def _validate_result(
        self,
        step_id: str,
        call_id: str,
        tool_name: str,
        arguments: str,
        observation: str,
    ):
        validation_context = self.trajectory.render_anchored(max_chars=6000)
        parsed, trace, total_tokens = request_result_validation(
            self.llm,
            validation_context,
            tool_name,
            arguments,
            observation,
            self.process_id,
            self.ablation.sampling_seed,
        )
        self.total_tokens += total_tokens
        self.query_count += int(trace.get("calls", 1))
        self.validator_tokens += total_tokens
        trace.update({
            "step_id": step_id,
            "call_id": call_id,
            "tool_name": tool_name,
            "context_mode": "step_anchored_graph",
            "context_chars": len(validation_context),
        })
        self.validation_trace.append(trace)
        return parsed

    def _exhausted_step_attempts(self, step_id: str) -> int:
        attempts = int(
            self.trajectory.graph.nodes[step_id].metadata.get("failed_attempts", 0)
        )
        return attempts if attempts >= 3 else 0
