from __future__ import annotations

import json
import re
from typing import Any


MAX_STAGES = 6
MAX_STAGE_STEPS = 3
MAX_GLOBAL_STATE = 16


def coarse_planning_messages(
    query: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Ask only for the stable L1/L2/L4 skeleton; never create L3 here."""
    catalog = _catalog(tools)
    schema = {
        "stages": [{
            "id": "S1",
            "name": "short stage name",
            "goal": "observable stage outcome",
            "depends_on": [],
        }],
        "global_state": [{
            "key": "constraint_name",
            "value": "fact explicitly present in the user query",
        }],
    }
    return [
        {
            "role": "system",
            "content": (
                "Create the coarse skeleton of a progressive TaskGraph. Return "
                "JSON only. Output L2 stages and query-grounded global L4 state; "
                "do not output tool calls or L3 steps. Stages must cover the full "
                "task, while depends_on contains only indispensable prior stage "
                "ids. Global state may contain only constraints or entities stated "
                "verbatim in the user query. Do not use hidden answers, target APIs, "
                "or future observations. The API catalog is supplied only to judge "
                "whether a stage can eventually be executed. Every L2 business stage "
                "must require at least one non-Finish API call from the supplied "
                "catalog. Do not create reasoning-only stages such as identify missing "
                "input, select, compare, interpret, summarize, present, or answer. "
                "Fold such reasoning into the nearest executable API stage. Do not "
                "create a final-answer stage; the runtime adds one independently."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{query}\n\nObservable API catalog:\n"
                f"{json.dumps(catalog, ensure_ascii=False)}\n\n"
                f"Required schema example:\n{json.dumps(schema)}"
            ),
        },
    ]


def parse_coarse_plan(content: str) -> dict[str, list[dict[str, Any]]]:
    try:
        raw = json.loads(_json_object(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback_coarse_plan()
    if not isinstance(raw, dict):
        return fallback_coarse_plan()
    raw_stages = raw.get("stages", [])
    if not isinstance(raw_stages, list):
        return fallback_coarse_plan()
    stages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_stages[:MAX_STAGES], start=1):
        if not isinstance(item, dict):
            continue
        stage_id = _clean_id(item.get("id"), f"S{index}")
        if stage_id in seen:
            stage_id = f"S{index}"
        seen.add(stage_id)
        dependencies = [
            _clean_id(value, "")
            for value in item.get("depends_on", [])
            if _clean_id(value, "")
        ]
        stages.append({
            "id": stage_id,
            "name": str(item.get("name", f"Stage {index}"))[:160],
            "goal": str(item.get("goal", ""))[:600],
            "depends_on": dependencies,
        })
    if not stages:
        return fallback_coarse_plan()
    global_state = []
    raw_state = raw.get("global_state", [])
    if not isinstance(raw_state, list):
        raw_state = []
    for item in raw_state[:MAX_GLOBAL_STATE]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()[:120]
        value = str(item.get("value", "")).strip()[:600]
        if key and value:
            global_state.append({"key": key, "value": value})
    return {"stages": stages, "global_state": global_state}


def fallback_coarse_plan() -> dict[str, list[dict[str, Any]]]:
    return {
        "stages": [{
            "id": "S1",
            "name": "Execute tool workflow",
            "goal": "Gather the evidence required by the user query.",
            "depends_on": [],
        }],
        "global_state": [],
    }


def stage_expansion_messages(
    query: str,
    stage: dict[str, Any],
    graph_context: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {"steps": [{
        "id": f"{stage.get('id', 'S1')}.1",
        "name": "one atomic action",
        "goal": "one observable result",
        "candidate_tools": ["one exact API name"],
        "depends_on": [],
        "references": [],
    }]}
    return [
        {
            "role": "system",
            "content": (
                "Expand only the activated L2 stage into executable L3 steps. "
                "Return JSON only. Each step is one atomic action and has exactly "
                "one observable API name. depends_on lists indispensable step ids; "
                "references lists useful but non-blocking prior node or step ids. "
                "Return between one and three steps. Never create one step per item "
                "in an observed list. When a list is long, use one representative "
                "or suitable item in a single API step rather than enumerating it. "
                "Use completed observations in the supplied TaskGraph and do not "
                "repeat Done work. Never include Finish and never invent an API."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query:\n{query}\n\nActivated stage:\n"
                f"{json.dumps(stage, ensure_ascii=False)}\n\n"
                f"Current TaskGraph:\n{graph_context}\n\nObservable API catalog:\n"
                f"{json.dumps(_catalog(tools), ensure_ascii=False)}\n\n"
                f"Required schema example:\n{json.dumps(schema)}"
            ),
        },
    ]


def parse_stage_expansion(
    content: str,
    stage_ref: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    available = {item["name"] for item in _catalog(tools)} - {"Finish"}
    try:
        raw = json.loads(_json_object(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(raw, dict) or not isinstance(raw.get("steps", []), list):
        return []
    raw_steps = raw.get("steps", [])
    if not raw_steps or len(raw_steps) > MAX_STAGE_STEPS:
        return []
    steps = []
    seen: set[str] = set()
    for index, item in enumerate(raw_steps[:MAX_STAGE_STEPS], start=1):
        if not isinstance(item, dict):
            return []
        step_id = _clean_id(item.get("id"), f"{stage_ref}.{index}")
        if step_id in seen:
            step_id = f"{stage_ref}.{index}"
        requested = [str(value) for value in item.get("candidate_tools", [])]
        if len(requested) != 1 or requested[0] not in available:
            return []
        selected = requested
        seen.add(step_id)
        steps.append({
            "id": step_id,
            "name": str(item.get("name", f"Step {index}"))[:160],
            "goal": str(item.get("goal", ""))[:600],
            "candidate_tools": selected,
            "depends_on": _clean_ids(item.get("depends_on", [])),
            "references": _clean_ids(item.get("references", [])),
        })
    return steps


def _catalog(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for raw in tools:
        function = raw.get("function", raw)
        if not isinstance(function, dict) or not function.get("name"):
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        result.append({
            "name": str(function["name"]),
            "description": str(function.get("description", ""))[:320],
            "parameters": list(properties) if isinstance(properties, dict) else [],
            "required": list(parameters.get("required", [])) if isinstance(parameters, dict) else [],
        })
    return result


def _clean_ids(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean_id(value, "") for value in values if _clean_id(value, "")]


def _clean_id(value: object, default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", str(value or ""))
    return cleaned[:64] or default


def _json_object(content: str) -> str:
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", str(content or "").strip(),
        flags=re.IGNORECASE,
    )
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model did not return a JSON object")
    return text[start:end + 1]
