from __future__ import annotations

import json
import re


def pruning_messages(candidate_context: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Prune a candidate TaskGraph subgraph for the scheduled L3 step. "
                "Keep every node needed to execute CURRENT_STEP: its task/stage path, "
                "relevant tool capability and parameters, prerequisite results, "
                "constraints, and failed attempts that prevent repetition. Remove "
                "unrelated branches. Return JSON only as "
                '{"keep_node_ids":["exact node id", "..."]}. Do not add facts.'
            ),
        },
        {"role": "user", "content": candidate_context},
    ]


def parse_keep_node_ids(
    content: str,
    allowed_ids: list[str],
) -> set[str] | None:
    """Return None on invalid output so execution safely keeps the full candidate."""
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    requested = payload.get("keep_node_ids") if isinstance(payload, dict) else None
    if not isinstance(requested, list):
        return None
    allowed = set(allowed_ids)
    return {str(node_id) for node_id in requested if str(node_id) in allowed}
