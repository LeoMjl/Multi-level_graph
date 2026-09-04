from __future__ import annotations

from typing import Any

from mlg.schemas import Episode


FORBIDDEN_METHOD_KEYS = {
    "evaluator_gold",
    "gold_dependencies",
    "gold_evidence",
    "gold_outputs",
    "gold_stage",
    "judge_required_fields",
    "stage_records",
}


def assert_no_oracle_payload(payload: dict[str, Any]) -> None:
    forbidden = _forbidden_paths(payload)
    if forbidden:
        raise AssertionError(f"Method payload contains oracle fields: {sorted(forbidden)}")
    for value in _provenance_values(payload):
        if any(marker in value for marker in ("evaluator_gold", "gold_visible", "oracle_sidecar")):
            raise AssertionError(f"Method payload contains oracle provenance {value}")


def assert_episode_method_input_safe(episode: Episode) -> None:
    assert_no_oracle_payload(episode.method_input())


def _forbidden_paths(value: Any, path: tuple[str, ...] = ()) -> set[str]:
    if isinstance(value, dict):
        found: set[str] = set()
        for key, item in value.items():
            normalized = str(key).lower()
            current = (*path, normalized)
            if normalized in FORBIDDEN_METHOD_KEYS or (normalized == "answers" and not path):
                found.add(".".join(current))
            found.update(_forbidden_paths(item, current))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for index, item in enumerate(value):
            found.update(_forbidden_paths(item, (*path, str(index))))
        return found
    return set()


def _provenance_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        values = []
        for key, item in value.items():
            if str(key).lower() in {"reason", "origin", "provenance", "namespace"}:
                values.append(str(item).lower())
            values.extend(_provenance_values(item))
        return values
    if isinstance(value, list):
        return [item for value_item in value for item in _provenance_values(value_item)]
    return []
