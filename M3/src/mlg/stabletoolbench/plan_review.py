from __future__ import annotations

from typing import Any


INTERMEDIATE_MARKERS = ("_search", "search_", "_lookup", "lookup_")


def needs_plan_review(
    plan: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> bool:
    """Detect plans that may stop at a lookup despite an observable downstream API."""
    return bool(downstream_candidates(plan, tools))


def downstream_candidates(
    plan: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[str]:
    selected = {
        str(name)
        for stage in plan
        for step in stage.get("steps", [])
        for name in step.get("candidate_tools", [])
    }
    available = {
        str(raw.get("function", raw).get("name", ""))
        for raw in tools
        if isinstance(raw.get("function", raw), dict)
    } - {"", "Finish"}
    candidates: set[str] = set()
    for name in selected:
        family = _intermediate_family(name)
        if family:
            candidates.update(
                candidate for candidate in available
                if candidate not in selected and candidate.startswith(f"{family}_")
            )
    return sorted(candidates)


def review_messages(
    base_messages: list[dict[str, str]],
    draft: str,
    candidate_names: list[str],
):
    return [
        *base_messages,
        {"role": "assistant", "content": draft},
        {
            "role": "user",
            "content": (
                "Audit this draft against the entire user request and API catalog, "
                "then return the corrected complete plan as JSON only. A plan ending "
                "at search/lookup is invalid when a same-family API can consume its "
                "identifier, slug, or result to obtain requested details or perform "
                "the operation. The observable downstream candidates omitted by the "
                f"draft are: {candidate_names}. Check their catalog descriptions and "
                "add at least one relevant downstream step with depends_on. This audit "
                "was triggered because the draft stops at an intermediate tool, so "
                "returning the same one-step lookup plan is invalid. Remove "
                "tools from incompatible semantic domains. If already complete, "
                "return the draft unchanged."
            ),
        },
    ]


def _intermediate_family(name: str) -> str:
    lowered = name.lower()
    for marker in INTERMEDIATE_MARKERS:
        index = lowered.find(marker)
        if index > 0:
            return name[:index].rstrip("_")
    return ""
