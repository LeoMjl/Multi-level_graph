from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES_PATH = PROJECT_ROOT / "data" / "processed" / "m1" / "episodes.jsonl"
SAMPLE_DIR = PROJECT_ROOT / "configs" / "evaluation_samples"
SOURCE_REGRESSION = SAMPLE_DIR / "m1_regression_10.json"
FULL_OUTPUT = SAMPLE_DIR / "m1_no_abstention_294.json"
REGRESSION_OUTPUT = SAMPLE_DIR / "m1_regression_10_no_abstention.json"
SELECTION_SEED = "m1-no-abstention-v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_episodes(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def base_category(category: str) -> str:
    return category.removesuffix("-abs")


def sample_row(episode: dict) -> dict:
    metadata = episode["metadata"]
    return {
        "episode_id": episode["episode_id"],
        "memory_category": metadata["memory_category"],
        "domain": metadata["domain"],
    }


def stable_rank(episode_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{episode_id}".encode()).hexdigest()


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    episodes = read_episodes(EPISODES_PATH)
    by_id = {episode["episode_id"]: episode for episode in episodes}
    ordinary = [
        episode
        for episode in episodes
        if not episode["metadata"]["memory_category"].endswith("-abs")
    ]
    source = read_json(SOURCE_REGRESSION)

    full_payload = {
        "schema": "mlg-m1-sample-manifest-v1",
        "purpose": "no_abstention_subset",
        "dataset": "LongMemEval-V2-Text",
        "dataset_file_sha256": source["dataset_file_sha256"],
        "selection_seed": "",
        "derived_from": "data/processed/m1/episodes.jsonl",
        "filter": "exclude memory_category suffix -abs",
        "samples": [sample_row(episode) for episode in ordinary],
    }

    source_rows = source["samples"]
    target_strata = Counter(
        (base_category(row["memory_category"]), row["domain"])
        for row in source_rows
    )
    retained_ids = [
        row["episode_id"]
        for row in source_rows
        if not row["memory_category"].endswith("-abs")
    ]
    selected_ids = list(retained_ids)
    selected = {episode_id for episode_id in selected_ids}
    retained_strata = Counter(
        (
            base_category(by_id[episode_id]["metadata"]["memory_category"]),
            by_id[episode_id]["metadata"]["domain"],
        )
        for episode_id in retained_ids
    )

    for stratum, target_count in sorted(target_strata.items()):
        needed = target_count - retained_strata[stratum]
        candidates = [
            episode
            for episode in ordinary
            if episode["episode_id"] not in selected
            and (
                base_category(episode["metadata"]["memory_category"]),
                episode["metadata"]["domain"],
            )
            == stratum
        ]
        candidates.sort(key=lambda item: stable_rank(item["episode_id"]))
        replacements = candidates[:needed]
        if len(replacements) != needed:
            raise RuntimeError(f"Not enough candidates for stratum {stratum}")
        selected_ids.extend(item["episode_id"] for item in replacements)
        selected.update(item["episode_id"] for item in replacements)

    regression_payload = {
        "schema": "mlg-m1-sample-manifest-v1",
        "purpose": "regression_no_abstention",
        "dataset": "LongMemEval-V2-Text",
        "dataset_file_sha256": source["dataset_file_sha256"],
        "selection_seed": SELECTION_SEED,
        "derived_from": "configs/evaluation_samples/m1_regression_10.json",
        "filter": "exclude memory_category suffix -abs",
        "replacement_policy": (
            "retain original non-abstention IDs; fill matching base-category/domain "
            "strata by SHA-256 rank"
        ),
        "samples": [sample_row(by_id[episode_id]) for episode_id in selected_ids],
    }

    write_manifest(FULL_OUTPUT, full_payload)
    write_manifest(REGRESSION_OUTPUT, regression_payload)
    print(f"{FULL_OUTPUT}: {len(full_payload['samples'])} samples")
    print(f"{REGRESSION_OUTPUT}: {len(regression_payload['samples'])} samples")


if __name__ == "__main__":
    main()
