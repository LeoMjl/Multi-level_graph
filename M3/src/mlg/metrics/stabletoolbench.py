from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


OFFICIAL_SUBSETS = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)


def official_sopr(label_counts: dict[str, Any]) -> dict[str, float | int]:
    """Reproduce StableToolEval's Solvable Pass Rate aggregation."""
    if not label_counts:
        raise ValueError("SoPR input contains no evaluated queries")
    rounds = sorted(
        {
            int(round_id)
            for item in label_counts.values()
            if isinstance(item, dict)
            for round_id in dict(item.get("is_solved", {}))
        }
    )
    if not rounds:
        raise ValueError("SoPR input contains no evaluator rounds")
    round_scores = []
    for round_id in rounds:
        total = 0.0
        for item in label_counts.values():
            labels = dict(item.get("is_solved", {})) if isinstance(item, dict) else {}
            label = str(labels.get(str(round_id), labels.get(round_id, "")))
            if label == "AnswerStatus.Solved":
                total += 1.0
            elif label == "AnswerStatus.Unsure":
                total += 0.5
        round_scores.append(100.0 * total / len(label_counts))
    return {
        "score": mean(round_scores),
        "std": pstdev(round_scores),
        "query_count": len(label_counts),
        "evaluate_times": len(round_scores),
    }


def official_sowr(
    preferences: dict[str, Any],
    *,
    reference_model: str,
    candidate_model: str,
) -> dict[str, float | int]:
    """Reproduce StableToolEval's candidate win/tie/lose aggregation."""
    if not preferences:
        raise ValueError("SoWR input contains no evaluated queries")
    wins = ties = losses = 0
    for item in preferences.values():
        if not isinstance(item, dict):
            losses += 1
            continue
        reference = int(item.get(reference_model, 0))
        candidate = int(item.get(candidate_model, 0))
        if candidate > reference:
            wins += 1
        elif candidate < reference:
            losses += 1
        else:
            ties += 1
    total = len(preferences)
    return {
        "score": 100.0 * wins / total,
        "tie_rate": 100.0 * ties / total,
        "lose_rate": 100.0 * losses / total,
        "query_count": total,
    }


def official_fac(path: Path) -> dict[str, float | int]:
    """Aggregate the official FAC evaluator CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"FAC input contains no evaluated queries: {path}")
    solved = 0
    for row in rows:
        label = str(row.get("evaluation", "")).lower().strip()
        if "unsolved" not in label and "solved" in label:
            solved += 1
    return {
        "score": 100.0 * solved / len(rows),
        "query_count": len(rows),
    }


def load_official_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Official evaluator artifact must be a JSON object: {path}")
    return payload


def macro_average(subsets: dict[str, dict[str, Any]]) -> float:
    if not subsets:
        raise ValueError("Cannot average empty StableToolBench subset results")
    return mean(float(item["score"]) for item in subsets.values())
