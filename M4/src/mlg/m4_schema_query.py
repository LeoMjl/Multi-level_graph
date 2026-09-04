from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))


def query_operator(text: str) -> str:
    padded = f" {str(text).casefold()} "
    if any(word in padded for word in (" before ", " after ", " between ", " earlier ", " later ", " change")):
        return "temporal"
    if any(word in padded for word in (" both ", " compared", " differ", " same ", " respectively")):
        return "comparison"
    return "inference"


def relation_allowed(operator: str, relation: str) -> bool:
    if relation == "event_temporal_next":
        return operator == "temporal"
    if relation == "same_attribute":
        return operator == "comparison"
    return True


def relation_bonus(operator: str, relations: list[str]) -> float:
    values = set(relations)
    bonus = 0.005 if values & {
        "same_entity_event", "same_attribute", "event_temporal_next", "narrative_dependency",
    } else 0.0
    if operator == "temporal" and "event_temporal_next" in values:
        bonus += 0.02
    if operator == "comparison" and "same_attribute" in values:
        bonus += 0.02
    if operator == "inference" and "same_entity_event" in values:
        bonus += 0.02
    if operator == "inference" and "narrative_dependency" in values:
        bonus += 0.02
    return bonus


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _query_dates(text: str) -> list[date]:
    found: list[date] = []
    years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)]
    inherited = years[0] if len(set(years)) == 1 else 0
    for year, month, day in re.findall(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)", text):
        try:
            found.append(date(int(year), int(month), int(day)))
        except ValueError:
            pass
    pattern = rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b"
    for month, day, year in re.findall(pattern, text.casefold()):
        try:
            found.append(date(int(year) if year else inherited, _MONTHS[month], int(day)))
        except ValueError:
            pass
    return list(dict.fromkeys(found))


def metadata_bonuses(records: list[dict[str, Any]], query_text: str, operator: str) -> np.ndarray:
    query = f" {_normalized(query_text)} "
    dates = _query_dates(query_text) if operator == "temporal" else []
    bonuses = np.zeros(len(records), dtype=np.float32)
    for index, row in enumerate(records):
        source = str(row.get("source", "")).strip()
        aliases = {source}
        for separator in ("|", " - "):
            aliases.add(source.split(separator, 1)[0])
        if any(alias and f" {_normalized(alias)} " in query for alias in aliases):
            bonuses[index] += 0.10
        if not dates or not row.get("published_at"):
            continue
        match = re.search(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)", str(row["published_at"]))
        if not match:
            continue
        try:
            published = date(*(int(value) for value in match.groups()))
        except ValueError:
            continue
        distance = min(abs((published - target).days) for target in dates)
        bonuses[index] += 0.08 / (1.0 + distance / 30.0)
    return bonuses
