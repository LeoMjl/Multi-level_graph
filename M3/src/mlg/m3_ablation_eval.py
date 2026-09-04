from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path


SUBSETS = ("G2_instruction", "G2_category", "G3_instruction")


def load_sopr_scores(root: Path, method: str) -> dict[str, dict[str, float]]:
    scores = {}
    weights = {
        "AnswerStatus.Solved": 1.0,
        "AnswerStatus.Unsure": 0.5,
        "AnswerStatus.Unsolved": 0.0,
    }
    for subset in SUBSETS:
        path = root / "pass_rate" / method / f"{subset}_{method}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        subset_scores = {}
        for query_id, item in payload.items():
            rounds = dict(item.get("is_solved", {}))
            if not rounds:
                raise ValueError(f"missing SoPR rounds: {path} query={query_id}")
            subset_scores[str(query_id)] = sum(
                weights[str(label)] for label in rounds.values()
            ) / len(rounds)
        scores[subset] = subset_scores
    return scores


def load_fac_scores(
    root: Path,
    official_root: Path,
    method: str,
) -> dict[str, dict[str, int]]:
    scores = {}
    for subset in SUBSETS:
        ids_path = official_root / "solvable_queries" / "test_query_ids" / f"{subset}.json"
        query_ids = [str(item) for item in json.loads(ids_path.read_text(encoding="utf-8"))]
        fac_path = root / "fac" / method / f"{subset}.csv"
        with fac_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(query_ids):
            raise ValueError(f"FAC count mismatch: {fac_path}")
        scores[subset] = {
            query_id: int(
                "solved" in str(row.get("evaluation", "")).lower()
                and "unsolved" not in str(row.get("evaluation", "")).lower()
            )
            for query_id, row in zip(query_ids, rows)
        }
    return scores


def paired_macro_difference(
    control: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
) -> float:
    differences = []
    for subset in SUBSETS:
        if set(control[subset]) != set(treatment[subset]):
            raise ValueError(f"paired IDs differ for {subset}")
        ids = control[subset]
        differences.append(
            sum(control[subset][item] - treatment[subset][item] for item in ids)
            / len(ids)
        )
    return sum(differences) / len(differences)


def stratified_bootstrap(
    control: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    observed = paired_macro_difference(control, treatment)
    rng = random.Random(seed)
    distributions = []
    paired = {
        subset: [
            control[subset][item] - treatment[subset][item]
            for item in control[subset]
        ]
        for subset in SUBSETS
    }
    for _ in range(samples):
        subset_means = []
        for subset in SUBSETS:
            values = paired[subset]
            subset_means.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
        distributions.append(sum(subset_means) / len(subset_means))
    ordered = sorted(distributions)
    low = ordered[max(0, int(0.025 * samples) - 1)]
    high = ordered[min(samples - 1, int(0.975 * samples))]
    nonpositive = sum(value <= 0 for value in distributions)
    nonnegative = sum(value >= 0 for value in distributions)
    p_value = min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1) / (samples + 1))
    return {
        "control_minus_ablation": 100.0 * observed,
        "ci95": [100.0 * low, 100.0 * high],
        "p_value": p_value,
        "samples": samples,
        "seed": seed,
    }


def mcnemar_exact(
    control: dict[str, dict[str, int]],
    treatment: dict[str, dict[str, int]],
) -> dict[str, float | int]:
    control_only = treatment_only = 0
    for subset in SUBSETS:
        if set(control[subset]) != set(treatment[subset]):
            raise ValueError(f"paired IDs differ for {subset}")
        for query_id, control_value in control[subset].items():
            treatment_value = treatment[subset][query_id]
            control_only += int(control_value == 1 and treatment_value == 0)
            treatment_only += int(control_value == 0 and treatment_value == 1)
    discordant = control_only + treatment_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(control_only, treatment_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "control_only_solved": control_only,
        "ablation_only_solved": treatment_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * values[name]))
        adjusted[name] = running
    return adjusted
