from __future__ import annotations
from copy import deepcopy
from Algorithms.single_chain import single_chain
from Prompts.ReAct_prompts import FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, FORMAT_INSTRUCTIONS_USER_FUNCTION
from Tree.Tree import tree_node
from mlg.stabletoolbench.ablation import TaskGraphAblation
from mlg.stabletoolbench.chain_runtime import StateDrivenCognitionMixin
from mlg.stabletoolbench.chain_tree import failure_node
from mlg.stabletoolbench.planning import STATE_DRIVEN_TASK_DESCRIPTION
from mlg.stabletoolbench.trajectory_graph import ToolTrajectoryGraph
class StableToolBenchGraphChain(StateDrivenCognitionMixin, single_chain):
    """Official CoT execution loop augmented with observable TaskGraph state."""
    def __init__(
        self,
        llm,
        io_func,
        extra_prefix="",
        process_id=0,
        start_message_list=None,
        *,
        ablation: TaskGraphAblation | None = None,
    ):
        self.ablation = ablation or TaskGraphAblation()
        super().__init__(
            llm,
            io_func,
            extra_prefix=extra_prefix,
            process_id=process_id,
            start_message_list=start_message_list,
        )

    def to_json(self, answer: bool = False, process: bool = True):
        payload = super().to_json(answer=answer, process=process)
        if hasattr(self, "trajectory"):
            payload.update(self._state_driven_artifacts())
        return payload

    def _finish_chain(self, node):
        return node

    def do_chain(self, now_node, single_chain_max_step):
        if self.start_message_list is None:
            system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION.replace(
                "{task_description}",
                STATE_DRIVEN_TASK_DESCRIPTION,
            )
            user = FORMAT_INSTRUCTIONS_USER_FUNCTION.replace(
                "{input_description}",
                self.io_func.input_description,
            )
            self.tree.root.messages.extend(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        else:
            self.tree.root.messages = self.start_message_list
        self.action_base_messages = [
            deepcopy(message)
            for message in self.tree.root.messages
            if message.get("role") in {"system", "user"}
        ]
        plan, self.plan_source = self._plan_task()
        self.trajectory = ToolTrajectoryGraph(
            self.io_func.input_description,
            self.io_func.functions,
            plan,
            plan_source=self.plan_source,
            ablation=self.ablation,
        )
        if self.ablation.expand_all_l3_before_execution:
            for stage_id in self.trajectory.layout.business_stage_ids:
                self._expand_stage(stage_id, phase="static_pre_execution")
        now_node = self.tree.root
        while True:
            stage_id = self.trajectory.stage_to_expand()
            if stage_id:
                self._expand_stage(stage_id)
                continue
            scheduled_step_id = self.trajectory.schedule()
            if not scheduled_step_id:
                stage_id = self.trajectory.stage_to_expand()
                if stage_id:
                    self._expand_stage(stage_id)
                    continue
                return self._finish_chain(
                    failure_node(now_node, "scheduler found no executable step")
                )
            keep_node_ids = self._prune_context(scheduled_step_id)
            if keep_node_ids is None:
                return self._finish_chain(
                    failure_node(now_node, "context pruning output was invalid")
                )
            graph_message = {
                "role": "system",
                "content": (
                    "Follow the state-driven TaskGraph below. The scheduler has "
                    "selected exactly one L3 step and exposed only tools routed to "
                    "that step. The graph contains only the user query, observable "
                    "API schemas, the model-created plan, and prior observations.\n"
                    + self.trajectory.render(keep_node_ids=keep_node_ids)
                ),
            }
            self.llm.change_messages([*self.action_base_messages, graph_message])
            new_message, error_code, total_tokens = self.llm.parse(
                tools=self.trajectory.routed_tools(scheduled_step_id),
                process_id=self.process_id,
                temperature=self.ablation.action_temperature,
                seed=self.ablation.sampling_seed,
            )
            self.total_tokens += total_tokens
            self.query_count += 1
            if new_message.get("role") != "assistant":
                new_message = {
                    "role": "assistant",
                    "content": str(new_message),
                }
                error_code = error_code or -1
            content = new_message.get("content")
            if content:
                thought_node = tree_node()
                thought_node.node_type = "Thought"
                thought_node.description = content
                thought_node.io_state = deepcopy(now_node.io_state)
                thought_node.is_terminal = thought_node.io_state.check_success() != 0
                thought_node.messages = now_node.messages.copy()
                thought_node.father = now_node
                now_node.children.append(thought_node)
                now_node = thought_node
                self.trajectory.add_thought(content, scheduled_step_id)
                if error_code != 0:
                    now_node.observation_code = error_code
                    now_node.pruned = True
            tool_calls = self._bounded_tool_calls(new_message, scheduled_step_id)
            for index, tool_call in enumerate(tool_calls):
                function = tool_call["function"]
                function_name = function["name"]
                arguments = function["arguments"]
                action_node = tree_node()
                action_node.node_type = "Action"
                action_node.description = function_name
                action_node.io_state = deepcopy(now_node.io_state)
                action_node.is_terminal = action_node.io_state.check_success() != 0
                action_node.messages = now_node.messages.copy()
                action_node.father = now_node
                now_node.children.append(action_node)
                now_node = action_node
                call_id = self.trajectory.add_call(
                    function_name,
                    arguments,
                    scheduled_step_id,
                )
                input_node = tree_node()
                input_node.node_type = "Action Input"
                input_node.description = arguments
                child_state = deepcopy(now_node.io_state)
                observation, status = child_state.step(
                    action_name=function_name,
                    action_input=arguments,
                )
                input_node.observation = observation
                input_node.observation_code = status
                input_node.io_state = child_state
                input_node.is_terminal = child_state.check_success() != 0
                input_node.messages = now_node.messages.copy()
                input_node.father = now_node
                now_node.children.append(input_node)
                now_node = input_node
                validation = self._validate_result(
                    scheduled_step_id,
                    call_id,
                    function_name,
                    arguments,
                    observation,
                ) if status == 0 else None
                outcome, reason, state_updates = validation or (
                    "invalid", "", [],
                )
                self.trajectory.add_observation(
                    call_id,
                    observation,
                    status,
                    validation_outcome=outcome,
                    validation_reason=reason,
                    state_updates=state_updates,
                )
                if status == 0 and validation is None:
                    return self._finish_chain(
                        failure_node(
                            now_node,
                            "result validation output was invalid",
                        )
                    )
                if status == 3:
                    self.trajectory.schedule()
                if status == 4:
                    now_node.observation_code = status
                elif status == 1:
                    tool_call["function"]["name"] = (
                        "invalid_hallucination_function_name"
                    )
                if index == 0:
                    now_node.messages.append(new_message)
                now_node.messages.append(
                    {
                        "role": "tool",
                        "name": tool_call["function"]["name"],
                        "content": observation,
                        "tool_call_id": tool_call["id"],
                    }
                )
                attempts = self._exhausted_step_attempts(scheduled_step_id)
                if attempts:
                    self.trajectory.drop_step(
                        scheduled_step_id,
                        f"retry budget exhausted after {attempts} attempts",
                    )
            if not tool_calls:
                now_node.messages.append(new_message)
                return self._finish_chain(
                    failure_node(
                        now_node,
                        "action model returned no structured tool call",
                    )
                )
            if (
                now_node.get_depth() >= single_chain_max_step
                and not now_node.is_terminal
            ):
                now_node.pruned = True
            if now_node.pruned or now_node.is_terminal:
                return self._finish_chain(now_node)
