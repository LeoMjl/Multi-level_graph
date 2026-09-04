from __future__ import annotations

import json
import re
from typing import Any


PLAN_PROTOCOL = "taskgraph_progressive_v4"
MAX_STAGES = 6
MAX_STEPS = 16
STATE_DRIVEN_TASK_DESCRIPTION = """Execute the user's task through the supplied
progressive TaskGraph. The graph starts with coarse stages; only the activated stage
is expanded into atomic tool steps. At each round, execute only CURRENT_STEP and call
only the function schema exposed for it. Use structured state derived from prior
observations and never guess unexposed functions. When the independent finalization
step is scheduled, call Finish with a complete answer; correct a failed active step
instead of silently skipping it."""


def planning_messages(
    query: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a query-only planning request from observable API schemas."""
    catalog = []
    for raw in tools:
        function = raw.get("function", raw)
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", ""))
        if not name or name == "Finish":
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        required = parameters.get("required", []) if isinstance(parameters, dict) else []
        catalog.append({
            "name": name,
            "description": str(function.get("description", ""))[:320],
            "parameters": list(properties) if isinstance(properties, dict) else [],
            "required": [str(item) for item in required if item],
        })
    schema = {
        "stages": [{
            "id": "S1",
            "name": "short stage name",
            "goal": "stage outcome",
            "steps": [{
                "id": "S1.1",
                "name": "one atomic action",
                "goal": "observable result needed from this action",
                "candidate_tools": ["one exact API name from the catalog"],
                "depends_on": [],
            }],
        }],
    }
    return [
        {
            "role": "system",
            "content": (
                "Decompose the user's tool-use task into a small executable TaskGraph. "
                "Return JSON only, matching the supplied schema. Each L3 step must be "
                "one atomic tool action with at most one exact candidate API name. Use "
                "depends_on for required prior step ids. The plan must be end-to-end "
                "complete: cover every distinct information or operation requested by "
                "the user. Never stop at an identifier, slug, or search result when an "
                "observable downstream API can use it to obtain the requested details; "
                "include that downstream call as a dependent step. Match the semantic "
                "domain and entity type of every API: never use a music, concert, product, "
                "or other domain-specific API merely because its name contains a generic "
                "word that also appears in the query. Before returning JSON, audit the "
                "whole plan: if it contains a search/lookup step and the user asked for "
                "details or an operation, it must also contain the relevant downstream "
                "detail/action step whenever such an API exists in the catalog. A plan "
                "that ends at an intermediate identifier is invalid. Do not include a final-answer "
                "step and do not invent tools. Use only the user query and API catalog; "
                "no gold answer, target API, or future observation is available."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{query}\n\nObservable API catalog:\n"
                f"{json.dumps(catalog, ensure_ascii=False)}\n\n"
                f"Required JSON schema example:\n{json.dumps(schema)}"
            ),
        },
    ]


def parse_plan(content: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize an LLM plan and fail closed to a single executable step."""
    available = _available_names(tools)
    try:
        raw = json.loads(_json_object(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback_plan()
    raw_stages = raw.get("stages", []) if isinstance(raw, dict) else []
    stages: list[dict[str, Any]] = []
    seen_steps: set[str] = set()
    step_count = 0
    for stage_index, raw_stage in enumerate(raw_stages[:MAX_STAGES], start=1):
        if not isinstance(raw_stage, dict):
            continue
        stage_id = _clean_id(raw_stage.get("id"), f"S{stage_index}")
        steps = []
        for local_index, raw_step in enumerate(raw_stage.get("steps", []), start=1):
            if not isinstance(raw_step, dict) or step_count >= MAX_STEPS:
                continue
            step_id = _clean_id(raw_step.get("id"), f"{stage_id}.{local_index}")
            if step_id in seen_steps:
                step_id = f"{stage_id}.{local_index}"
            seen_steps.add(step_id)
            requested = raw_step.get("candidate_tools", [])
            candidates = [
                str(name) for name in requested
                if str(name) in available and str(name) != "Finish"
            ][:1]
            if not candidates:
                continue
            dependencies = [
                _clean_id(item, "") for item in raw_step.get("depends_on", [])
                if _clean_id(item, "")
            ]
            steps.append({
                "id": step_id,
                "name": str(raw_step.get("name", "Tool action"))[:160],
                "goal": str(raw_step.get("goal", raw_step.get("name", "")))[:600],
                "candidate_tools": candidates,
                "depends_on": dependencies,
            })
            step_count += 1
        if steps:
            stages.append({
                "id": stage_id,
                "name": str(raw_stage.get("name", f"Stage {stage_index}"))[:160],
                "goal": str(raw_stage.get("goal", ""))[:600],
                "steps": steps,
            })
    return stages or fallback_plan()


def fallback_plan() -> list[dict[str, Any]]:
    return [{
        "id": "S1",
        "name": "Execute tool workflow",
        "goal": "Gather the information required by the user query.",
        "steps": [{
            "id": "S1.1",
            "name": "Gather required evidence",
            "goal": "Use the best observable API to gather evidence for the user query.",
            "candidate_tools": [],
            "depends_on": [],
        }],
    }]


def _available_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        str(raw.get("function", raw).get("name", ""))
        for raw in tools
        if isinstance(raw.get("function", raw), dict)
    }


def _json_object(content: str) -> str:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("planner did not return a JSON object")
    return text[start:end + 1]


def _clean_id(value: object, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", str(value or ""))
    return cleaned[:64] or default
