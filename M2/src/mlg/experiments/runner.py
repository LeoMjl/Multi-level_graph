from __future__ import annotations

import json
import hashlib
import time
from collections import Counter
from pathlib import Path

from mlg.config import (
    RESULTS_DIR,
    RAW_DIR,
    RuntimeConfig,
    ensure_runtime_dirs,
    select_judge_profile,
    select_model_profiles,
)
from mlg.data.prepare import load_prepared
from mlg.experiments.registry import (
    allowed_methods_for_suite_experiment,
    default_methods_for_suite,
    experiment_role,
    experiments_for_suite,
    methods_for_suite_experiment,
    skipped_methods_for_suite_experiment,
)
from mlg.methods import METHODS
from mlg.metrics import aggregate_records, evaluate_prediction, judge_detail_records
from mlg.schemas import Episode, Message, MetricRecord, Prediction, write_jsonl


def run_suite(
    *,
    suite: str,
    experiments: list[str] | None = None,
    methods: list[str],
    limit: int = 0,
    fake_llm: bool = False,
    model: str = "",
    judge_model: str = "",
    model_config: Path | None = None,
    model_profiles: list[str] | None = None,
    judge_profile: str = "",
    run_dir: Path | None = None,
    seeds: list[int] | None = None,
    episode_ids: list[str] | None = None,
) -> dict:
    ensure_runtime_dirs()
    requested_methods = methods or default_methods_for_suite(suite)
    runtime = RuntimeConfig()
    selected_profiles = select_model_profiles(
        model_config=model_config,
        model_profiles=model_profiles,
        fallback_model=model or runtime.model,
    )
    selected_judge_profile = select_judge_profile(
        model_config=model_config,
        judge_profile=judge_profile,
        fallback_judge_model=judge_model or runtime.judge_model,
    )
    judge_runtime = RuntimeConfig(api_key="") if fake_llm else RuntimeConfig.from_profile(selected_judge_profile, judge=True)
    resolved_run_dir = run_dir or RESULTS_DIR / "latest"
    run_seeds = seeds or [0]
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    suite_experiments = experiments_for_suite(suite)
    selected_experiments = [item.lower() for item in experiments] if experiments else suite_experiments
    invalid_experiments = [item for item in selected_experiments if item not in suite_experiments]
    if invalid_experiments:
        raise ValueError(
            f"Experiment(s) not in suite {suite}: {', '.join(invalid_experiments)}. "
            f"Allowed: {', '.join(suite_experiments)}"
        )
    if not selected_experiments:
        raise ValueError("At least one experiment must be selected")
    config = {
        "suite": suite,
        "experiments": selected_experiments,
        "methods": requested_methods,
        "limit": limit,
        "fake_llm": fake_llm,
        "model_config": str(model_config) if model_config else "",
        "model_profiles": list(selected_profiles),
        "judge_profile": selected_judge_profile.name,
        "judge_model": selected_judge_profile.model,
        "seeds": run_seeds,
        "episode_ids": list(episode_ids or []),
        "sample_selection": "episode_ids" if episode_ids else ("limit" if limit else "all"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (resolved_run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics: list[MetricRecord] = []
    all_predictions: list[Prediction] = []
    all_judge_details: list[dict] = []
    all_api_usage: list[dict] = []
    memory_build_records: list[dict] = []
    protocol_manifests: dict[str, dict] = {}
    skipped_methods: dict[str, list[str]] = {}
    for experiment in selected_experiments:
        if experiment == "m3" and not fake_llm:
            raise RuntimeError(
                "Formal M3 runs use scripts/run_m3_toolbench_official.* so "
                "StableToolBench inference and evaluation stay inside the pinned upstream harness."
            )
        if experiment == "m4" and not fake_llm:
            raise RuntimeError(
                "Formal M4 runs use scripts/run_m4_document_qa.py so QuALITY and "
                "MultiHop-RAG share one auditable document index per dataset."
            )
        # Fake mode is a pipeline smoke test, not an evaluation of real benchmark
        # data. Keeping it on compact fixtures prevents 100-trajectory M1 histories
        # from making unit tests look like formal runs and avoids pseudo-scoring the
        # official LLM-judge categories without a judge.
        if fake_llm:
            if episode_ids:
                raise ValueError("episode_ids selection is not supported in fake-LLM smoke mode")
            episodes = smoke_episodes(experiment, limit or 2)
        elif episode_ids:
            episodes = select_episodes_by_id(load_prepared(experiment, limit=0), episode_ids)
        else:
            episodes = load_prepared(experiment, limit=limit)
        if experiment == "m1":
            protocol_manifests[experiment] = validate_m1_protocol(
                episodes,
                fake_llm=fake_llm,
                limit=limit or (len(episode_ids) if episode_ids else 0),
            )
        elif experiment == "m2":
            protocol_manifests[experiment] = validate_m2_protocol(
                episodes,
                fake_llm=fake_llm,
                limit=limit or (len(episode_ids) if episode_ids else 0),
            )
        elif experiment == "m3":
            protocol_manifests[experiment] = validate_m3_protocol(
                episodes,
                fake_llm=fake_llm,
                limit=limit or (len(episode_ids) if episode_ids else 0),
            )
        elif experiment == "m4":
            protocol_manifests[experiment] = validate_m4_protocol(
                episodes,
                fake_llm=fake_llm,
                limit=limit or (len(episode_ids) if episode_ids else 0),
            )
        experiment_methods = methods_for_suite_experiment(suite, experiment, requested_methods)
        skipped = skipped_methods_for_suite_experiment(suite, experiment, requested_methods)
        if skipped:
            skipped_methods[experiment] = skipped
        if not experiment_methods:
            allowed = ", ".join(allowed_methods_for_suite_experiment(suite, experiment))
            raise ValueError(f"No requested methods are enabled for {experiment}. Allowed: {allowed}")
        for profile_name, profile in selected_profiles.items():
            method_runtime = RuntimeConfig.from_profile(profile)
            if experiment == "m1" and not fake_llm:
                validate_m1_runtime(
                    method_runtime,
                    judge_runtime,
                    selected_judge_profile.model,
                    experiment_methods,
                )
            elif experiment == "m2" and not fake_llm:
                validate_m2_runtime(method_runtime, judge_runtime)
            for seed in run_seeds:
                profile_dir = resolved_run_dir / experiment / profile_name / f"seed_{seed}"
                profile_dir.mkdir(parents=True, exist_ok=True)
                for method_name in experiment_methods:
                    if method_name not in METHODS:
                        raise ValueError(f"Unknown method {method_name}. Known: {', '.join(sorted(METHODS))}")
                    method = METHODS[method_name](
                        fake_llm=fake_llm,
                        runtime=method_runtime,
                        model_profile=profile_name,
                        seed=seed,
                    )
                    prediction_path = profile_dir / f"{method_name}_predictions.jsonl"
                    predictions = []
                    for episode in episodes:
                        predictions.append(method.predict(episode))
                        # Incremental and long-context methods can take many minutes.
                        # Persist each completed episode so a later timeout does not
                        # discard already-paid model calls.
                        write_jsonl(
                            prediction_path,
                            [item.to_dict() for item in predictions],
                        )
                    for prediction in predictions:
                        for call_index, call in enumerate(prediction.metadata.get("api_calls", []), start=1):
                            all_api_usage.append({
                                "experiment": experiment,
                                "episode_id": prediction.episode_id,
                                "method": method_name,
                                "model_profile": profile_name,
                                "seed": seed,
                                "call_index": call_index,
                                **call,
                            })
                        if prediction.memory_build_time_ms > 0:
                            memory_build_records.append({
                                "experiment": experiment,
                                "episode_id_trigger": prediction.episode_id,
                                "method": method_name,
                                "model_profile": profile_name,
                                "seed": seed,
                                "haystack_key": prediction.metadata.get("haystack_key", ""),
                                "memory_build_time_ms": prediction.memory_build_time_ms,
                                "memory_build_api_usage": prediction.api_usage.get("memory_build", {}),
                                "vector_cache_enabled": prediction.metadata.get("vector_cache_enabled", False),
                                "vector_cache_hit": prediction.metadata.get("vector_cache_hit", False),
                                "vector_cache_status": prediction.metadata.get("vector_cache_status", ""),
                                "vector_cache_key": prediction.metadata.get("vector_cache_key", ""),
                                "vector_cache_path": prediction.metadata.get("vector_cache_path", ""),
                                "status": prediction.status,
                                "failure_reason": prediction.failure_reason,
                            })
                    # Checkpoint expensive model outputs before invoking optional
                    # LLM judges. A judge configuration/error must not discard all
                    # already-completed predictions for this method.
                    write_jsonl(
                        prediction_path,
                        [item.to_dict() for item in predictions],
                    )
                    run_mode = "fake" if fake_llm else "llm"
                    metrics = []
                    for episode, prediction in zip(episodes, predictions):
                        record = evaluate_prediction(experiment, episode, prediction, runtime=judge_runtime)
                        record.run_mode = run_mode
                        record.model_profile = profile_name
                        record.fallback = prediction_fallback_reason(prediction)
                        record.failure = prediction_failure_reason(prediction)
                        record.seed = seed
                        metrics.append(record)
                        all_judge_details.extend(
                            judge_detail_records(
                                experiment,
                                episode,
                                prediction,
                                record,
                                judge_model=selected_judge_profile.model,
                            )
                        )
                    write_jsonl(profile_dir / f"{method_name}_metrics.jsonl", [item.to_dict() for item in metrics])
                    all_predictions.extend(predictions)
                    all_metrics.extend(metrics)

    write_jsonl(resolved_run_dir / "predictions.jsonl", [item.to_dict() for item in all_predictions])
    write_jsonl(resolved_run_dir / "metrics.jsonl", [item.to_dict() for item in all_metrics])
    write_jsonl(resolved_run_dir / "judge_details.jsonl", all_judge_details)
    for detail in all_judge_details:
        trace = detail.get("judge_trace", {})
        evaluation_mode = trace.get("evaluation_mode", "") if isinstance(trace, dict) else ""
        if isinstance(trace, dict) and evaluation_mode in {
            "official_longmemeval_v2_llm_judge",
            "locomo_adapted_factscore_llm",
        }:
            all_api_usage.append({
                "experiment": detail.get("experiment", ""),
                "episode_id": detail.get("episode_id", ""),
                "method": detail.get("method", ""),
                "model_profile": "judge",
                "seed": detail.get("seed", 0),
                "call_index": 1,
                "phase": "judge",
                "kind": "chat.completions",
                "model": trace.get("model", ""),
                "elapsed_ms": trace.get("latency_ms", 0.0),
                "usage": trace.get("usage"),
                "usage_missing": trace.get("usage_missing", True),
                "error": "",
            })
    write_jsonl(resolved_run_dir / "api_usage.jsonl", all_api_usage)
    write_jsonl(resolved_run_dir / "memory_build.jsonl", memory_build_records)
    summary = aggregate_records(
        all_metrics,
        metadata={
            "suite": suite,
            "requested_methods": requested_methods,
            "experiment_methods": {
                experiment: methods_for_suite_experiment(suite, experiment, requested_methods)
                for experiment in selected_experiments
            },
            "skipped_methods": skipped_methods,
            "model_profiles": list(selected_profiles),
            "judge_profile": selected_judge_profile.name,
            "judge_model": selected_judge_profile.model,
            "seeds": run_seeds,
            "fallback_policy": "allow_smoke_fallback" if fake_llm else "fail_closed",
            "protocols": protocol_manifests,
            "main_experiments": [exp for exp in selected_experiments if experiment_role(exp) == "main"],
            "auxiliary_experiments": [exp for exp in selected_experiments if experiment_role(exp) == "auxiliary"],
            "legacy_experiments": [exp for exp in selected_experiments if experiment_role(exp) == "legacy"],
        },
    )
    (resolved_run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    run_manifest = {
        **config,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_mode": "fake" if fake_llm else "llm",
        "fallback_policy": "allow_smoke_fallback" if fake_llm else "fail_closed",
        "protocols": protocol_manifests,
        "artifacts": {
            "predictions": "predictions.jsonl",
            "metrics": "metrics.jsonl",
            "api_usage": "api_usage.jsonl",
            "memory_build": "memory_build.jsonl",
            "judge_details": "judge_details.jsonl",
            "summary": "summary.json",
        },
    }
    (resolved_run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"run_dir": str(resolved_run_dir), "summary": summary, "metric_records": len(all_metrics)}


def select_episodes_by_id(episodes: list[Episode], episode_ids: list[str]) -> list[Episode]:
    """Select episodes in manifest order and fail on missing or duplicate IDs."""
    normalized = [str(item).strip() for item in episode_ids if str(item).strip()]
    if not normalized:
        raise ValueError("episode_ids must contain at least one ID")
    if len(set(normalized)) != len(normalized):
        raise ValueError("episode_ids contains duplicate IDs")
    by_id = {episode.episode_id: episode for episode in episodes}
    missing = [episode_id for episode_id in normalized if episode_id not in by_id]
    if missing:
        raise ValueError("Unknown episode ID(s): " + ", ".join(missing))
    return [by_id[episode_id] for episode_id in normalized]


def prediction_fallback_reason(prediction: Prediction) -> str:
    explicit = str(prediction.metadata.get("fallback", "")).strip()
    if explicit:
        return explicit
    retrieval = prediction.metadata.get("retrieval", "")
    if isinstance(retrieval, dict):
        retrieval = retrieval.get("retrieval", "")
    retrieval_name = str(retrieval).strip()
    if "lexical_fallback_after_embedding_error" in retrieval_name:
        return retrieval_name
    return ""


def prediction_failure_reason(prediction: Prediction) -> str:
    if prediction.status != "failed":
        return ""
    if prediction.failure_stage:
        return f"{prediction.failure_stage}:{prediction.failure_reason}"
    return prediction.failure_reason or "method_failed"


M1_EXPECTED_QUESTION_TYPES = {
    "dynamic-environment": 86,
    "dynamic-environment-abs": 41,
    "procedure": 74,
    "procedure-abs": 32,
    "static-environment": 134,
    "static-environment-abs": 55,
}
M1_EXPECTED_DOMAINS = {"enterprise": 197, "web": 225}


def validate_m1_protocol(episodes: list[Episode], *, fake_llm: bool, limit: int) -> dict:
    if fake_llm:
        return {
            "status": "smoke",
            "formal_eligible": False,
            "dataset": "synthetic-smoke",
            "sample_count": len(episodes),
            "expected_sample_count": 422,
            "tier": "smoke",
            "lafs_status": "not_reported",
            "lafs_reason": "fake run is not an official protocol run",
        }
    datasets = sorted({episode.dataset for episode in episodes})
    tiers = sorted({str(episode.metadata.get("haystack_tier", "")) for episode in episodes})
    question_types = Counter(str(episode.metadata.get("memory_category", "")) for episode in episodes)
    domains = Counter(str(episode.metadata.get("domain", "")) for episode in episodes)
    haystack_signatures = {
        str(episode.history_ref.get("signature", ""))
        for episode in episodes
        if episode.history_ref.get("signature")
    }
    checks = {
        "dataset": datasets == ["LongMemEval-V2-Text"],
        "tier": tiers == ["small"],
        "sample_count": len(episodes) == 422,
        "unique_question_ids": len({episode.episode_id for episode in episodes}) == 422,
        "question_type_counts": dict(question_types) == M1_EXPECTED_QUESTION_TYPES,
        "domain_counts": dict(domains) == M1_EXPECTED_DOMAINS,
        "shared_haystack_count": len(haystack_signatures) == 2,
    }
    formal_eligible = all(checks.values()) and limit == 0
    if limit == 0 and not formal_eligible:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            "M1 formal protocol validation failed: " + ", ".join(failed)
        )
    manifest = {
        "status": "formal_protocol" if formal_eligible else "pilot_incomplete",
        "formal_eligible": formal_eligible,
        "dataset": datasets,
        "sample_count": len(episodes),
        "expected_sample_count": 422,
        "tier": tiers,
        "question_type_counts": dict(sorted(question_types.items())),
        "domain_counts": dict(sorted(domains.items())),
        "unique_haystack_count": len(haystack_signatures),
        "checks": checks,
        "lafs_status": "not_reported",
        "lafs_reason": (
            "the released reference frontier covers all 451 questions and is not a corresponding "
            "frontier for LongMemEval-V2-Text (422 questions)"
        ),
        "source_hashes": m1_source_hashes(),
    }
    return manifest


def validate_m1_runtime(
    method_runtime: RuntimeConfig,
    judge_runtime: RuntimeConfig,
    judge_model: str,
    methods: list[str],
) -> None:
    if not method_runtime.api_key:
        raise RuntimeError(
            f"M1 formal run requires reader API key {method_runtime.api_key_env}"
        )
    embedding_methods = {
        "ours_full",
        "ours_wo_hier",
        "ours_wo_mainline",
        "ours_wo_dep",
        "ours_wo_active_pool",
        "raw_rag",
        "atomic_rag",
    }
    if embedding_methods.intersection(methods) and not method_runtime.embedding_api_key:
        raise RuntimeError(
            f"M1 formal run requires embedding API key {method_runtime.embedding_api_key_env}"
        )
    if judge_model != "deepseek-v4-flash":
        raise RuntimeError("M1 formal protocol requires judge model deepseek-v4-flash")
    if not judge_runtime.api_key:
        raise RuntimeError(
            f"M1 formal protocol requires judge API key {judge_runtime.api_key_env}"
        )


def validate_m2_protocol(
    episodes: list[Episode],
    *,
    fake_llm: bool,
    limit: int,
) -> dict:
    expected_count = 20
    checks = {
        "official_dataset": all(
            item.dataset == "LoCoMo Event Summarization (official)"
            for item in episodes
        ),
        "official_task": all(
            item.metadata.get("task_type") == "locomo_event_summarization"
            for item in episodes
        ),
        "speaker_level_targets": all(
            item.metadata.get("target_speaker") and item.metadata.get("gold_events")
            for item in episodes
        ),
        "full_count": len(episodes) == expected_count,
    }
    formal = not fake_llm and not limit
    required = ("official_dataset", "official_task", "speaker_level_targets")
    if formal:
        required += ("full_count",)
    failed = [name for name in required if not checks[name]]
    if failed:
        raise RuntimeError(f"M2 protocol validation failed: {', '.join(failed)}")
    source = RAW_DIR / "repos" / "locomo" / "repository" / "data" / "locomo10.json"
    return {
        "protocol": "official_locomo_event_summarization",
        "task_unit": "one target speaker per official conversation",
        "expected_episode_count": expected_count,
        "actual_episode_count": len(episodes),
        "complete": bool(formal and checks["full_count"]),
        "pilot": not formal,
        "official_metrics": [
            "rouge_1",
            "rouge_2",
            "rouge_l",
            "factscore_precision",
            "factscore_recall",
            "factscore_f1",
        ],
        "official_baseline_families": [
            "base",
            "long_context",
            "incremental_summarization",
        ],
        "upstream_event_evaluator": "not_released_in_official_repository",
        "source_hash": sha256_file(source) if source.is_file() else "",
        "checks": checks,
    }


def validate_m3_protocol(
    episodes: list[Episode],
    *,
    fake_llm: bool,
    limit: int,
) -> dict:
    expected_counts = {
        "G1_instruction": 163,
        "G1_category": 153,
        "G1_tool": 158,
        "G2_instruction": 106,
        "G2_category": 124,
        "G3_instruction": 61,
    }
    subset_counts = Counter(item.split for item in episodes)
    query_ids = [str(item.metadata.get("query_id", "")) for item in episodes]
    checks = {
        "official_dataset": bool(episodes) and all(
            item.dataset == "StableToolBench" for item in episodes
        ),
        "solvable_queries_only": bool(episodes) and all(
            bool(item.metadata.get("solvable")) for item in episodes
        ),
        "registered_query_ids": bool(query_ids) and all(query_ids)
        and len(set(query_ids)) == len(query_ids),
        "official_api_definitions": bool(episodes) and all(
            isinstance(item.metadata.get("tools"), list)
            and bool(item.metadata.get("tools"))
            for item in episodes
        ),
        "official_test_subsets": set(subset_counts).issubset(expected_counts),
        "full_query_count": len(episodes) == 765,
        "full_subset_counts": dict(subset_counts) == expected_counts,
    }
    formal = not fake_llm and not limit
    required = [] if fake_llm else [
        "official_dataset",
        "solvable_queries_only",
        "registered_query_ids",
        "official_api_definitions",
        "official_test_subsets",
    ]
    if formal:
        required.extend(["full_query_count", "full_subset_counts"])
    failed = [name for name in required if not checks[name]]
    if failed:
        raise RuntimeError(f"M3 protocol validation failed: {', '.join(failed)}")
    return {
        "protocol": "official_stabletoolbench_solvable_queries",
        "task_unit": "one interactive tool-use query",
        "expected_query_count": 765,
        "actual_query_count": len(episodes),
        "subset_counts": dict(sorted(subset_counts.items())),
        "complete": bool(formal and not failed),
        "pilot": not formal,
        "official_metrics": ["SoPR", "SoWR", "FAC"],
        "aggregation": "macro_average_over_six_official_subsets",
        "execution_harness": "scripts/run_m3_toolbench_official.*",
        "checks": checks,
    }


def validate_m4_protocol(
    episodes: list[Episode],
    *,
    fake_llm: bool,
    limit: int,
) -> dict:
    if not fake_llm:
        raise RuntimeError("Formal M4 validation is owned by scripts/run_m4_document_qa.py")
    return {
        "status": "smoke",
        "formal_eligible": False,
        "dataset": "synthetic-document-qa",
        "sample_count": len(episodes),
        "formal_datasets": ["QuALITY-v1.0.1-dev", "MultiHop-RAG"],
        "execution_harness": "scripts/run_m4_document_qa.py",
    }


def validate_m2_runtime(
    method_runtime: RuntimeConfig,
    judge_runtime: RuntimeConfig,
) -> None:
    if not method_runtime.api_key:
        raise RuntimeError(
            f"M2 formal run requires reader API key {method_runtime.api_key_env}"
        )
    if not judge_runtime.api_key:
        raise RuntimeError(
            f"M2 FactScore evaluation requires judge API key {judge_runtime.api_key_env}"
        )


def m1_source_hashes() -> dict[str, str]:
    base = RAW_DIR / "repos" / "longmemeval_v2"
    dataset = base / "dataset"
    repository = base / "repository"
    paths = {
        "questions_jsonl": dataset / "questions.jsonl",
        "small_haystacks": dataset / "haystacks" / "lme_v2_small.json",
        "official_qa_eval_metrics": repository / "evaluation" / "qa_eval_metrics.py",
        "official_harness": repository / "evaluation" / "harness.py",
        "official_lafs": repository / "leaderboard" / "compute_lafs.py",
    }
    return {
        name: sha256_file(path)
        for name, path in paths.items()
        if path.is_file()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_episodes(experiment: str, limit: int) -> list[Episode]:
    episodes: list[Episode] = []
    for idx in range(limit):
        if experiment in {"m1", "a1"}:
            episodes.append(
                Episode(
                    episode_id=f"smoke_{experiment}_{idx}",
                    dataset="SmokeLongMemEvalV2" if experiment == "m1" else "SmokeLongMemEval",
                    split="smoke",
                    history=[Message(role="user", content=f"Conversation fact: project codename is Atlas{idx}.", turn_index=1)],
                    query="What is the project codename?",
                    answers=[f"Atlas{idx}"],
                    gold_evidence=[f"Conversation fact: project codename is Atlas{idx}."],
                    gold_dependencies=[{"source": f"Conversation fact: project codename is Atlas{idx}.", "target": "memory_retrieval"}],
                    metadata={"memory_category": "long_term_memory", "task_type": "long_agent_memory"},
                )
            )
        elif experiment == "m2":
            fact = f"Alex started a new job on 2 January 2024."
            episodes.append(
                Episode(
                    episode_id=f"smoke_m2_{idx}",
                    dataset="LoCoMo Event Summarization (official)",
                    split="smoke",
                    history=[
                        Message(
                            role="user",
                            content=f"[session_1 | 2 January 2024 | Alex]\n{fact}",
                            turn_index=1,
                            metadata={
                                "session_id": "session_1",
                                "session_date": "2 January 2024",
                                "speaker": "Alex",
                            },
                        )
                    ],
                    query="Summarize Alex's significant life events in chronological order.",
                    answers=[f"- 2 January 2024: {fact}"],
                    metadata={
                        "task_type": "locomo_event_summarization",
                        "target_speaker": "Alex",
                        "timeframe": "2 January 2024",
                        "session_count": 1,
                        "gold_events": [{"date": "2 January 2024", "fact": fact}],
                    },
                )
            )
        elif experiment == "m3":
            episodes.append(
                Episode(
                    episode_id=f"smoke_m3_{idx}",
                    dataset="SmokeStableToolBench",
                    split="smoke",
                    history=[
                        Message(role="system", content="Available API: weather.get_forecast(city)", turn_index=1),
                    ],
                    query="What is the weather in Beijing?",
                    metadata={
                        "query_id": f"smoke-{idx}",
                        "solvable": True,
                        "task_type": "stabletoolbench_tool_use",
                        "tools": [{
                            "category_name": "Weather",
                            "tool_name": "weather",
                            "api_name": "get_forecast",
                            "api_description": "Get a city forecast.",
                            "required_parameters": [{"name": "city", "type": "string"}],
                            "optional_parameters": [],
                        }],
                    },
                )
            )
        elif experiment == "m4":
            episodes.append(
                Episode(
                    episode_id=f"smoke_m4_{idx}",
                    dataset="SmokeDocumentQA",
                    split="smoke",
                    history=[
                        Message(
                            role="system",
                            content="Document: France's capital city is Paris.",
                            turn_index=0,
                        )
                    ],
                    query="What is the capital of France?",
                    answers=["Paris"],
                    metadata={"query_id": str(idx), "task_type": "document_qa"},
                )
            )
        elif experiment == "exp3":
            episodes.append(
                Episode(
                    episode_id=f"smoke_exp3_{idx}",
                    dataset="SmokeMirrorAPI",
                    split="smoke",
                    history=[Message(role="system", content="Available tools: calculator", turn_index=0)],
                    query="Produce the calculator API response.",
                    answers=["5"],
                    metadata={"tools": [{"name": "calculator"}]},
                )
            )
        elif experiment in {"a5", "exp5"}:
            answer = f"needle_value_{idx:04d}"
            episodes.append(
                Episode(
                    episode_id=f"smoke_{experiment}_{idx}",
                    dataset="SmokeNeedle",
                    split="smoke",
                    history=[Message(role="user", content=f"Long haystack. The hidden needle code for this sample is {answer}.", turn_index=1)],
                    query="What is the hidden needle code for this sample?",
                    answers=[answer],
                    metadata={"context_length_bucket": 1},
                )
            )
        else:
            answer = f"LongBench{idx}"
            episodes.append(
                Episode(
                    episode_id=f"smoke_{experiment}_{idx}",
                    dataset="SmokeEvidenceQA" if experiment in {"a2", "a3"} else "SmokeLongBench",
                    split="smoke",
                    history=[Message(role="user", content=f"Document: the answer is {answer}.", turn_index=1)],
                    query="Return the answer.",
                    answers=[answer],
                    gold_evidence=[f"Document: the answer is {answer}."],
                    gold_dependencies=[{"source": f"Document: the answer is {answer}.", "target": "Return the answer."}],
                    metadata={"task_type": "qa", "context_length_bucket": 1},
                )
            )
    from mlg.data.sidecar import attach_sidecars

    return attach_sidecars(episodes, experiment)
