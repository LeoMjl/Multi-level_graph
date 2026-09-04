from __future__ import annotations

from collections import defaultdict
from typing import Any


def candidate_score(candidate: dict[str, Any]) -> float:
    """Combine query-local retrieval signals on a comparable scale."""
    dense_score = float(candidate.get("dense_score", 0.0) or 0.0)
    lexical_rank = candidate.get("lexical_rank")
    lexical_bonus = (
        0.25 / (1.0 + int(lexical_rank))
        if lexical_rank is not None
        else 0.0
    )
    anchor_bonus = 0.35 * len(candidate.get("anchor_matches", []) or [])
    scope_bonus = 0.40 * len(candidate.get("scope_matches", []) or [])
    term_bonus = 0.02 * min(10, int(candidate.get("term_hits", 0) or 0))
    return dense_score + lexical_bonus + anchor_bonus + scope_bonus + term_bonus


def rank_trajectory_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    scope_items: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Aggregate fragment evidence before selecting trajectories.

    Explicit scope items receive coverage first. Remaining slots are filled by
    trajectory-level relevance, rather than by the first fragment encountered.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        trajectory_id = str(candidate.get("trajectory_id", ""))
        if not trajectory_id:
            continue
        enriched = dict(candidate)
        enriched["candidate_score"] = candidate_score(candidate)
        grouped[trajectory_id].append(enriched)

    ranked: list[dict[str, Any]] = []
    for trajectory_id, rows in grouped.items():
        rows.sort(key=lambda row: (-float(row["candidate_score"]), int(row["index"])))
        top_scores = [float(row["candidate_score"]) for row in rows[:3]]
        covered_scope = sorted({
            str(item)
            for row in rows
            for item in (row.get("scope_matches", []) or [])
        })
        covered_anchors = sorted({
            str(item)
            for row in rows
            for item in (row.get("anchor_matches", []) or [])
        })
        aggregate = top_scores[0]
        aggregate += 0.20 * (sum(top_scores) / len(top_scores))
        aggregate += 0.20 * len(covered_scope)
        aggregate += 0.10 * len(covered_anchors)
        ranked.append({
            "trajectory_id": trajectory_id,
            "score": aggregate,
            "best_fragment_score": top_scores[0],
            "matched_scope_items": covered_scope,
            "matched_anchors": covered_anchors,
            "candidate_count": len(rows),
        })
    ranked.sort(key=lambda row: (-float(row["score"]), str(row["trajectory_id"])))

    selected: list[str] = []
    for scope_item in scope_items or []:
        if any(
            row["trajectory_id"] in selected
            and scope_item in row["matched_scope_items"]
            for row in ranked
        ):
            continue
        match = next(
            (
                row for row in ranked
                if scope_item in row["matched_scope_items"]
                and row["trajectory_id"] not in selected
            ),
            None,
        )
        if match:
            selected.append(str(match["trajectory_id"]))
        if len(selected) >= limit:
            return selected, ranked
    for row in ranked:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id not in selected:
            selected.append(trajectory_id)
        if len(selected) >= limit:
            break
    return selected, ranked


def rank_state_seed_ids(
    candidates: list[dict[str, Any]],
    selected_trajectory_ids: list[str],
    *,
    scope_items: list[str] | None = None,
    max_total: int = 12,
    per_trajectory: int = 3,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Select diverse high-scoring states inside the shortlisted trajectories."""
    selected_set = set(selected_trajectory_ids)
    best_by_node: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate.get("trajectory_id") not in selected_set:
            continue
        row = dict(candidate)
        row["score"] = candidate_score(candidate)
        node_id = str(row.get("node_id", ""))
        current = best_by_node.get(node_id)
        if node_id and (
            current is None
            or float(row["score"]) > float(current["score"])
        ):
            best_by_node[node_id] = row
    ranked = sorted(
        best_by_node.values(),
        key=lambda row: (
            selected_trajectory_ids.index(str(row["trajectory_id"])),
            -float(row["score"]),
            int(row["index"]),
        ),
    )

    chosen: list[str] = []
    counts: dict[str, int] = defaultdict(int)
    for scope_item in scope_items or []:
        if any(
            row["node_id"] in chosen
            and scope_item in (row.get("scope_matches", []) or [])
            for row in ranked
        ):
            continue
        match = next(
            (
                row for row in ranked
                if scope_item in (row.get("scope_matches", []) or [])
                and row["node_id"] not in chosen
            ),
            None,
        )
        if match:
            chosen.append(str(match["node_id"]))
            counts[str(match["trajectory_id"])] += 1
    for row in ranked:
        node_id = str(row["node_id"])
        trajectory_id = str(row["trajectory_id"])
        if node_id in chosen or counts[trajectory_id] >= per_trajectory:
            continue
        chosen.append(node_id)
        counts[trajectory_id] += 1
        if len(chosen) >= max_total:
            break
    return chosen[:max_total], ranked
