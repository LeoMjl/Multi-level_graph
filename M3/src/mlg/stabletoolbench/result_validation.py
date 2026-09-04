from __future__ import annotations

import json
import re


VALIDATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "step_validation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["done", "retry"]},
                "reason": {"type": "string"},
                "state_updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                            "scope": {
                                "type": "string",
                                "enum": ["task", "stage", "step"],
                            },
                            "evidence": {"type": "string"},
                        },
                        "required": ["key", "value", "scope", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["outcome", "reason", "state_updates"],
            "additionalProperties": False,
        },
    },
}


def validation_messages(
    step_context: str,
    tool_name: str,
    arguments: str,
    observation: str,
) -> list[dict[str, str]]:
    schema_example = {
        "outcome": "done",
        "reason": "The requested result is present.",
        "state_updates": [{
            "key": "stable_fact_name",
            "value": "exact reusable value",
            "scope": "task",
            "evidence": "exact scalar substring from the observation",
        }],
    }
    return [
        {
            "role": "system",
            "content": (
                "Validate whether one successful tool result completes CURRENT_STEP. "
                "Check intent consistency and whether the observation contains the "
                "result required by CURRENT_GOAL. Return exactly one JSON object "
                "with both outcome and reason. Use outcome=done only when the goal "
                "is satisfied; use outcome=retry for empty, not-found, error-like, "
                "mismatched, or incomplete results. "
                "Use retry when another or corrected call is needed. Also return "
                "state_updates for reusable facts extracted from the observation. "
                "Every state update must contain exactly key, value, scope, and "
                "evidence. Return at most three state updates. Each value must be "
                "at most 160 characters; summarize long lists with a count and a "
                "few representative identifiers instead of copying the list. If a "
                "fact cannot fit, omit that update. Scope must be one of task, stage, "
                "or step. Evidence must "
                "be a short plain string of at most 160 characters. Never paste a "
                "JSON object or array into evidence, and never use quote characters "
                "or backslashes inside it. A safe evidence value is tool observation. "
                "Use an empty list when "
                "nothing reusable was observed. This textual contract is mandatory "
                "even when the API supports JSON-object mode but not JSON Schema. "
                "Do not answer the user or use facts absent from the supplied result."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{step_context}\n\nTOOL={tool_name}\nARGUMENTS={arguments}\n"
                f"OBSERVATION={observation[:2400]}\n\n"
                "Required JSON shape:\n"
                f"{json.dumps(schema_example, ensure_ascii=False)}"
            ),
        },
    ]


def parse_validation(
    content: str,
) -> tuple[str, str, list[dict[str, str]]] | None:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    outcome = str(payload.get("outcome", "")).lower() if isinstance(payload, dict) else ""
    if outcome not in {"done", "retry"}:
        return None
    raw_updates = payload.get("state_updates")
    if not isinstance(raw_updates, list):
        return None
    updates = []
    for item in raw_updates[:16]:
        if not isinstance(item, dict):
            return None
        key = str(item.get("key", "")).strip()[:120]
        value = str(item.get("value", "")).strip()[:800]
        evidence = str(item.get("evidence", "")).strip()[:800]
        scope = str(item.get("scope", "")).lower()
        if not (key and value and evidence and scope in {"task", "stage", "step"}):
            return None
        updates.append({
            "key": key,
            "value": value,
            "scope": scope,
            "evidence": evidence,
        })
    return outcome, str(payload.get("reason", ""))[:1000], updates
