from __future__ import annotations

import json
from typing import Any

from mlg.stabletoolbench.context_pruning import (
    parse_keep_node_ids,
    pruning_messages,
)
from mlg.stabletoolbench.progressive_planning import (
    coarse_planning_messages,
    fallback_coarse_plan,
    parse_coarse_plan,
    parse_stage_expansion,
    stage_expansion_messages,
)
from mlg.stabletoolbench.result_validation import (
    VALIDATION_RESPONSE_FORMAT,
    parse_validation,
    validation_messages,
)


def request_plan(
    llm, query: str, tools: list[dict[str, Any]], process_id: int,
    sampling_seed: int = 20260831,
):
    messages = coarse_planning_messages(query, tools)
    llm.change_messages(messages)
    message, error_code, total_tokens = llm.parse(
        tools=[], process_id=process_id, max_tokens=900, temperature=0,
        seed=sampling_seed,
        response_format={"type": "json_object"},
    )
    content = message.get("content", "") if isinstance(message, dict) else ""
    plan = parse_coarse_plan(content)
    source = (
        "fallback"
        if error_code != 0 or plan == fallback_coarse_plan()
        else "model"
    )
    return plan, source, total_tokens, str(content)[:6000], 1


def request_stage_expansion(
    llm,
    query: str,
    stage: dict[str, Any],
    graph_context: str,
    tools: list[dict[str, Any]],
    process_id: int,
    sampling_seed: int = 20260831,
):
    messages = stage_expansion_messages(query, stage, graph_context, tools)
    llm.change_messages(messages)
    message, error_code, total_tokens = llm.parse(
        tools=[], process_id=process_id, max_tokens=1200, temperature=0,
        seed=sampling_seed,
        response_format={"type": "json_object"},
    )
    content = message.get("content", "") if isinstance(message, dict) else ""
    stage_ref = str(stage.get("id", "S1"))
    steps = (
        parse_stage_expansion(content, stage_ref, tools)
        if error_code == 0 else []
    )
    calls = 1
    raw_responses = [str(content)[:6000]]
    source = "model" if steps else "invalid"
    if not steps:
        allowed = []
        for raw_tool in tools:
            function = raw_tool.get("function", raw_tool)
            if not isinstance(function, dict):
                continue
            name = str(function.get("name", ""))
            if name and name != "Finish":
                allowed.append(name)
        llm.change_messages([
            *messages,
            {"role": "assistant", "content": str(content)},
            {
                "role": "user",
                "content": (
                    "The previous stage expansion was invalid. Repair it once. "
                    "Return exactly one JSON object with one to three steps. Never "
                    "enumerate an observed list into one step per item; choose one "
                    "representative or suitable item for a single API step. "
                    "Every step must contain exactly one candidate_tools value, "
                    "copied verbatim from ALLOWED_API_NAMES. Never use Finish and "
                    "never emit a reasoning-only, selection, presentation, summary, "
                    "or final-answer step. If reasoning is needed, fold it into an "
                    "executable API step.\n\n"
                    f"ALLOWED_API_NAMES={json.dumps(sorted(set(allowed)))}"
                ),
            },
        ])
        repaired, repair_error, repair_tokens = llm.parse(
            tools=[], process_id=process_id, max_tokens=1200, temperature=0,
            seed=sampling_seed,
            response_format={"type": "json_object"},
        )
        repaired_content = (
            repaired.get("content", "") if isinstance(repaired, dict) else ""
        )
        total_tokens += repair_tokens
        calls += 1
        raw_responses.append(str(repaired_content)[:6000])
        steps = (
            parse_stage_expansion(repaired_content, stage_ref, tools)
            if repair_error == 0 else []
        )
        source = "model_repaired" if steps else "invalid"
    return (
        steps,
        source,
        total_tokens,
        "\n--- stage expansion repair ---\n".join(raw_responses),
        calls,
    )


def request_context_pruning(
    llm,
    candidate_context: str,
    allowed_ids: list[str],
    process_id: int,
    sampling_seed: int = 20260831,
):
    llm.change_messages(pruning_messages(candidate_context))
    message, error_code, total_tokens = llm.parse(
        # Reasoning models count internal reasoning against max_tokens.  A
        # 500-token cap can therefore end before the small JSON payload is
        # emitted when the candidate graph is non-trivial.
        tools=[], process_id=process_id, max_tokens=1200, temperature=0,
        seed=sampling_seed,
        response_format={"type": "json_object"},
    )
    content = message.get("content", "") if isinstance(message, dict) else ""
    keep_ids = None if error_code != 0 else parse_keep_node_ids(content, allowed_ids)
    trace = {
        "status": "model" if keep_ids is not None else "invalid",
        "candidate_count": len(allowed_ids),
        "kept_node_ids": sorted(keep_ids) if keep_ids is not None else allowed_ids,
        "tokens": total_tokens,
        "finish_reason": (
            message.get("_mlg_finish_reason", "")
            if isinstance(message, dict) else ""
        ),
        "completion_tokens": (
            message.get("_mlg_completion_tokens")
            if isinstance(message, dict) else None
        ),
        "reasoning_tokens": (
            message.get("_mlg_reasoning_tokens")
            if isinstance(message, dict) else None
        ),
        "raw_response": str(content)[:4000],
    }
    return keep_ids, trace, total_tokens


def request_result_validation(
    llm,
    step_context: str,
    tool_name: str,
    arguments: str,
    observation: str,
    process_id: int,
    sampling_seed: int = 20260831,
):
    messages = validation_messages(
        step_context, tool_name, arguments, observation,
    )
    llm.change_messages(messages)
    message, error_code, total_tokens = llm.parse(
        tools=[], process_id=process_id, max_tokens=300, temperature=0,
        seed=sampling_seed,
        response_format=VALIDATION_RESPONSE_FORMAT,
    )
    content = message.get("content", "") if isinstance(message, dict) else ""
    parsed = None if error_code != 0 else parse_validation(content)
    calls = 1
    raw_responses = [str(content)[:4000]]
    if parsed is None:
        llm.change_messages([
            *messages,
            {"role": "assistant", "content": str(content)},
            {
                "role": "user",
                "content": (
                    "The previous validator response was invalid. Return exactly "
                    "one valid JSON object matching the schema. Every state update "
                    "must include scope. Return no more than three updates, keep every "
                    "value under 160 characters, and replace long lists with a short "
                    "count or omit that update. To avoid malformed nested JSON, set every "
                    "evidence value to exactly the plain string tool observation. "
                    "Do not copy objects, arrays, quotes, or backslashes into evidence."
                ),
            },
        ])
        repaired, repair_error, repair_tokens = llm.parse(
            tools=[], process_id=process_id, max_tokens=400, temperature=0,
            seed=sampling_seed,
            response_format=VALIDATION_RESPONSE_FORMAT,
        )
        repaired_content = (
            repaired.get("content", "") if isinstance(repaired, dict) else ""
        )
        total_tokens += repair_tokens
        calls += 1
        raw_responses.append(str(repaired_content)[:4000])
        parsed = (
            None if repair_error != 0 else parse_validation(repaired_content)
        )
    trace = {
        "status": (
            "model" if parsed is not None and calls == 1
            else "model_repaired" if parsed is not None
            else "invalid"
        ),
        "calls": calls,
        "outcome": parsed[0] if parsed else "invalid",
        "reason": parsed[1] if parsed else "",
        "state_update_count": len(parsed[2]) if parsed else 0,
        "tokens": total_tokens,
        "raw_response": "\n--- validator retry ---\n".join(raw_responses),
    }
    return parsed, trace, total_tokens
