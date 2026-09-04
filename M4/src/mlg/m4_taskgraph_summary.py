from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "Summarize each supplied document-memory group independently for semantic retrieval and question "
    "answering. Preserve names, events, temporal order, comparisons, motivations, causal links, and "
    "distinctive details. Use at most 90 words per item. Return JSON only as "
    "{\"summaries\":[{\"item_id\":\"...\",\"summary\":\"...\"}]} and process every item."
)


def _parse(raw: dict[str, Any]) -> dict[str, str]:
    rows = raw.get("summaries", [])
    if isinstance(rows, dict):
        rows = [{"item_id": key, "summary": value} for key, value in rows.items()]
    return {
        str(row.get("item_id", "")): str(row.get("summary", "")).strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("summary", "")).strip()
    }


def _call(model: Any, items: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, Any]]:
    payload = {"items": [
        {"item_id": item["item_id"], "passages": item["text"]}
        for item in items
    ]}
    raw = model.chat_json(
        SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False),
        max_tokens=max(512, 180 * len(items)), phase="memory_build",
    )
    if raw.get("_error"):
        raise RuntimeError(str(raw["_error"]))
    values = _parse(raw)
    expected = {item["item_id"] for item in items}
    if len(expected) == 1 and not expected.intersection(values) and len(values) == 1:
        values = {next(iter(expected)): next(iter(values.values()))}
    call = dict(model._api_calls[-1]) if model._api_calls else {}
    return values, call


def summarize_taskgraph_batch(
    model: Any, items: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, Any]]:
    values, primary_call = _call(model, items)
    by_id = {item["item_id"]: item for item in items}
    missing = sorted(set(by_id) - set(values))
    recovery_calls = []
    for item_id in missing:
        recovered, call = _call(model, [by_id[item_id]])
        recovery_calls.append(call)
        values.update(recovered)
    still_missing = sorted(set(by_id) - set(values))
    if still_missing:
        raise RuntimeError(f"TaskGraph summarizer omitted {len(still_missing)} items: {still_missing[:3]}")
    if recovery_calls:
        primary_call["partial_batch_recovery_count"] = len(recovery_calls)
        primary_call["recovery_calls"] = recovery_calls
    return {item_id: values[item_id] for item_id in by_id}, primary_call
