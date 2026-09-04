from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Callable

from mlg.config import PROCESSED_DIR, RAW_DIR, ensure_runtime_dirs
from mlg.data.adapters import (
    LONGMEMEVAL_V2_CATEGORY_ORDER,
    adapt_a1_longmemeval_cleaned,
    adapt_a2_hotpotqa,
    adapt_a3_qasper,
    adapt_a4_longbench,
    adapt_a5_needle_source,
    adapt_longbench,
    adapt_m1_longmemeval_v2,
    adapt_m2_locomo_event_summarization,
    adapt_m3_toolbench,
    adapt_stabletoolbench,
)
from mlg.data.sidecar import attach_sidecars, make_sidecar
from mlg.data.validation import assert_episode_method_input_safe
from mlg.experiments.registry import AUXILIARY_EXPERIMENTS, MAIN_EXPERIMENTS
from mlg.schemas import Episode, Message, write_jsonl


ADAPTERS: dict[str, Callable[[int], list[Episode]]] = {
    "m1": adapt_m1_longmemeval_v2,
    "m2": adapt_m2_locomo_event_summarization,
    "m3": adapt_m3_toolbench,
    "a1": adapt_a1_longmemeval_cleaned,
    "a2": adapt_a2_hotpotqa,
    "a3": adapt_a3_qasper,
    "a4": adapt_a4_longbench,
    "a5": adapt_a5_needle_source,
}

ADAPTER_VERSIONS = {
    "m2": "official-locomo-event-summarization-v1",
    "m3": "official-stabletoolbench-v1",
    "m4": "m4-document-qa-v1",
}

EXPERIMENT_SOURCE_MANIFESTS = {
    "m1": ["longmemeval_v2.json"],
    "m2": ["locomo.json"],
    "m3": ["stabletoolbench.json"],
    "a1": ["longmemeval.json"],
    "a2": ["hotpotqa.json"],
    "a3": ["qasper.json"],
    "a4": ["longbench.json"],
    "a5": ["longbench.json"],
    "exp3": ["stabletoolbench.json"],
    "exp4": ["longbench.json"],
    "exp5": ["longbench.json"],
}

LEGACY_ADAPTERS: dict[str, Callable[[int], list[Episode]]] = {
    "exp3": adapt_stabletoolbench,
    "exp4": adapt_longbench,
    "exp5": adapt_longbench,
}


def prepare_experiment(experiment: str, limit: int = 0) -> list[dict]:
    ensure_runtime_dirs()
    normalized = experiment.lower()
    if normalized == "all":
        experiments = list(MAIN_EXPERIMENTS) + list(AUXILIARY_EXPERIMENTS)
    elif normalized in {"main", "core", "ablation"}:
        experiments = list(MAIN_EXPERIMENTS)
    elif normalized == "aux":
        experiments = list(AUXILIARY_EXPERIMENTS)
    elif normalized == "legacy":
        experiments = list(LEGACY_ADAPTERS)
    else:
        experiments = [normalized]
    manifests = []
    for exp in experiments:
        if exp == "m4":
            from mlg.m4_data import prepare_m4_data

            manifest = prepare_m4_data()
            manifests.append({
                "experiment": "m4",
                "episode_count": int(manifest["quality"]["queries"]) + int(manifest["multihop_rag"]["queries"]),
                **manifest,
            })
            continue
        adapter = ADAPTERS.get(exp) or LEGACY_ADAPTERS.get(exp)
        if adapter is None:
            raise ValueError(f"Unknown experiment {experiment}")
        episodes = adapter(limit)
        if exp in {"exp5", "a5"}:
            episodes = make_needle_episodes(episodes, limit)
        data_quality = experiment_data_quality(exp, episodes)
        out_dir = PROCESSED_DIR / exp
        history_store_count = 0
        if exp == "m1":
            history_store_count = externalize_m1_histories(episodes, out_dir)
        if exp != "m2":
            episodes = attach_sidecars(episodes, exp)
        for episode in episodes:
            assert_episode_method_input_safe(episode)
        rows = [episode.to_dict(include_sidecar=False) for episode in episodes]
        sidecars = (
            []
            if exp == "m2"
            else [episode.sidecar or make_sidecar(episode, experiment=exp) for episode in episodes]
        )
        write_jsonl(out_dir / "episodes.jsonl", rows)
        sidecar_path = out_dir / "sidecars.jsonl"
        if exp == "m2":
            sidecar_path.unlink(missing_ok=True)
        else:
            write_jsonl(sidecar_path, sidecars)
        manifest = {
            "experiment": exp,
            "adapter_version": ADAPTER_VERSIONS.get(exp, "v1"),
            "episode_count": len(episodes),
            "sidecar_count": len(sidecars),
            "external_history_store_count": history_store_count,
            "data_quality": data_quality,
            "source_manifests": [
                str(RAW_DIR / "manifests" / name)
                for name in EXPERIMENT_SOURCE_MANIFESTS.get(exp, [])
            ],
            "oracle_policy": "Gold answers/evidence/stages/dependencies are evaluator-only fields; method_input strips them.",
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifests.append(manifest)
    return manifests


def experiment_data_quality(experiment: str, episodes: list[Episode]) -> dict:
    if not episodes:
        return {"episode_count": 0}
    if experiment == "m1":
        return {
            "episode_count": len(episodes),
            "domains": sorted({str(item.metadata.get("domain", "")) for item in episodes}),
            "memory_categories": sorted({str(item.metadata.get("memory_category", "")) for item in episodes}),
            "haystack_tiers": sorted({str(item.metadata.get("haystack_tier", "")) for item in episodes}),
            "trajectory_count_min": min(int(item.metadata.get("trajectory_count", 0)) for item in episodes),
            "trajectory_count_max": max(int(item.metadata.get("trajectory_count", 0)) for item in episodes),
            "multimodal_question_count": sum(bool(item.metadata.get("multimodal_question")) for item in episodes),
            "excluded_multimodal_question_count": max(
                int(item.metadata.get("excluded_multimodal_question_count", 0)) for item in episodes
            ),
            "official_eval_spec_rate": sum(bool(item.metadata.get("eval_function")) for item in episodes) / len(episodes),
        }
    if experiment == "m2":
        parent_ids = {
            str(item.metadata.get("parent_episode_id", ""))
            for item in episodes
        }
        event_counts = [
            len(item.metadata.get("gold_events", []))
            for item in episodes
        ]
        return {
            "episode_count": len(episodes),
            "conversation_count": len(parent_ids),
            "target_speaker_count": len(episodes),
            "gold_event_count": sum(event_counts),
            "gold_events_per_target_min": min(event_counts),
            "gold_events_per_target_max": max(event_counts),
            "source_files": sorted({
                str(item.metadata.get("source_file", ""))
                for item in episodes
            }),
            "official_task": "event_summarization",
        }
    if experiment == "m3":
        return {
            "episode_count": len(episodes),
            "official_subsets": sorted({item.split for item in episodes}),
            "source_files": sorted({str(item.metadata.get("source_file", "")) for item in episodes}),
            "solvable_rate": sum(bool(item.metadata.get("solvable")) for item in episodes) / len(episodes),
            "api_definition_rate": sum(bool(item.metadata.get("tools")) for item in episodes) / len(episodes),
            "api_reference_count": sum(len(item.metadata.get("tools", [])) for item in episodes),
            "official_task": "stabletoolbench_tool_use",
        }
    return {"episode_count": len(episodes)}


def load_prepared(experiment: str, limit: int = 0) -> list[Episode]:
    if experiment.lower() == "m4":
        raise RuntimeError(
            "M4 uses shared document indexes; run scripts/run_m4_document_qa.py."
        )
    path = PROCESSED_DIR / experiment.lower() / "episodes.jsonl"
    manifest_path = path.with_name("manifest.json")
    expected_version = ADAPTER_VERSIONS.get(experiment.lower())
    prepared_version = ""
    if manifest_path.is_file():
        try:
            prepared_version = str(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("adapter_version", "")
            )
        except (json.JSONDecodeError, OSError):
            prepared_version = ""
    if not path.exists() or (expected_version and prepared_version != expected_version):
        prepare_experiment(experiment, limit=limit)
    rows = []
    stratified_m1 = experiment.lower() == "m1" and bool(limit)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(Episode.from_dict(json.loads(line)))
            if limit and not stratified_m1 and len(rows) >= limit:
                break
    if stratified_m1:
        rows = stratified_prepared_m1(rows, limit)
    materialize_external_histories(rows, path.parent)
    sidecar_path = path.with_name("sidecars.jsonl")
    if sidecar_path.exists():
        sidecars: dict[str, dict] = {}
        with sidecar_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sidecars[str(item.get("item_id", ""))] = item
        for episode in rows:
            episode.sidecar = sidecars.get(episode.episode_id, {})
            if experiment.lower() == "m1":
                # M1 sidecars are intentionally lightweight on disk. The graph method
                # rebuilds from the shared, materialized raw haystack at runtime.
                episode.sidecar = {}
    return rows


def stratified_prepared_m1(episodes: list[Episode], limit: int) -> list[Episode]:
    if not limit or limit >= len(episodes):
        return episodes
    order = {category: idx for idx, category in enumerate(LONGMEMEVAL_V2_CATEGORY_ORDER)}
    strata: dict[tuple[str, str], list[Episode]] = {}
    for episode in episodes:
        key = (
            str(episode.metadata.get("memory_category", "")),
            str(episode.metadata.get("domain", "")),
        )
        strata.setdefault(key, []).append(episode)
    for items in strata.values():
        items.sort(key=lambda item: hashlib.sha256(item.episode_id.encode("utf-8")).hexdigest())
    keys = sorted(strata, key=lambda key: (order.get(key[0], len(order)), key[1]))
    selected: list[Episode] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if depth < len(strata[key]):
                selected.append(strata[key][depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def externalize_m1_histories(episodes: list[Episode], out_dir: Path) -> int:
    history_dir = out_dir / "histories"
    history_dir.mkdir(parents=True, exist_ok=True)
    stored: dict[str, Path] = {}
    for episode in episodes:
        trajectory_ids = episode.metadata.get("haystack_trajectory_ids", [])
        signature_payload = json.dumps(
            {
                "tier": episode.metadata.get("haystack_tier", ""),
                "trajectory_ids": trajectory_ids,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()[:20]
        history_path = history_dir / f"{signature}.jsonl"
        if signature not in stored:
            write_jsonl(history_path, [message.__dict__ for message in episode.history])
            stored[signature] = history_path
        episode.history_ref = {
            "schema": "mlg-history-ref-v1",
            "path": history_path.relative_to(out_dir).as_posix(),
            "message_count": len(episode.history),
            "signature": signature,
        }
        episode.history = []
    return len(stored)


def materialize_external_histories(episodes: list[Episode], out_dir: Path) -> None:
    cache: dict[Path, list[Message]] = {}
    for episode in episodes:
        if episode.history or not episode.history_ref:
            continue
        if episode.history_ref.get("schema") != "mlg-history-ref-v1":
            raise ValueError(f"Unsupported history reference for {episode.episode_id}")
        rel_path = Path(str(episode.history_ref.get("path", "")))
        history_path = (out_dir / rel_path).resolve()
        out_resolved = out_dir.resolve()
        if out_resolved != history_path and out_resolved not in history_path.parents:
            raise ValueError(f"External history path escapes processed directory: {rel_path}")
        if history_path not in cache:
            if not history_path.is_file():
                raise FileNotFoundError(f"External history store is missing: {history_path}")
            messages: list[Message] = []
            with history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        messages.append(Message(**json.loads(line)))
            expected_count = int(episode.history_ref.get("message_count", 0) or 0)
            if expected_count and len(messages) != expected_count:
                raise ValueError(
                    f"External history count mismatch for {history_path}: {len(messages)} != {expected_count}"
                )
            cache[history_path] = messages
        episode.history = cache[history_path]


def make_needle_episodes(base: list[Episode], limit: int = 0) -> list[Episode]:
    needles: list[Episode] = []
    source = base[: limit or len(base)]
    if not source:
        return needles
    for idx, episode in enumerate(source):
        answer = f"needle_value_{idx:04d}"
        fact = f"The hidden needle code for this sample is {answer}."
        history_text = "\n".join(msg.content for msg in episode.history)
        needle_position = ("head", "middle", "tail")[idx % 3]
        if needle_position == "head":
            haystack = f"{fact}\n\n{history_text}"
            depth_ratio = 0.0
        elif needle_position == "middle":
            midpoint = len(history_text) // 2
            haystack = f"{history_text[:midpoint]}\n\n{fact}\n\n{history_text[midpoint:]}"
            depth_ratio = 0.5
        else:
            haystack = f"{history_text}\n\n{fact}"
            depth_ratio = 1.0
        needles.append(
            Episode(
                episode_id=f"needle_{episode.episode_id}",
                dataset="Needle-in-a-Haystack",
                split=episode.split,
                history=[episode.history[0].__class__(role="user", content=haystack, turn_index=1)],
                query="What is the hidden needle code for this sample?",
                answers=[answer],
                gold_evidence=[fact],
                metadata={
                    "seed": idx,
                    "source_episode_id": episode.episode_id,
                    "source_subset": episode.metadata.get("subset", ""),
                    "needle_position": needle_position,
                    "needle_depth_ratio": depth_ratio,
                    "context_length_bucket": len(haystack) // 1000,
                },
            )
        )
    return needles
