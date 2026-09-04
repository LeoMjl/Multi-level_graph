from __future__ import annotations

from collections import defaultdict
import importlib.util
import itertools
import json
import random
import re
from statistics import mean
import time
from typing import Any

from mlg.config import RAW_DIR, RuntimeConfig
from mlg.metrics.judge import (
    semantic_dependency_recall,
)
from mlg.metrics.locomo import locomo_factscore, official_rouge_scores
from mlg.metrics.text import exact_match, f1, rouge_l
from mlg.methods.base import extract_api_usage
from mlg.schemas import Episode, MetricRecord, Prediction
from mlg.experiments.registry import experiment_role


# exp3 is a legacy MirrorAPI response-generation diagnostic. M4 uses the
# upstream StableToolBench harness and never reuses these local indicators.
EXP3_AUXILIARY_METRICS = ["tool_name_match_rate", "latency_ms"]


def evaluate_prediction(
    experiment: str,
    episode: Episode,
    prediction: Prediction,
    *,
    runtime: RuntimeConfig | None = None,
) -> MetricRecord:
    exp = experiment.lower()
    metrics: dict[str, float] = {}
    extra: dict[str, Any] = {}
    if exp == "m1":
        if prediction.status == "failed":
            official_score, evaluation_mode, official_judge_trace = 0.0, "method_failure", {}
        else:
            official_score, evaluation_mode, official_judge_trace = longmemeval_v2_official_score(
                prediction,
                episode,
                runtime=runtime,
            )
        metrics = {
            "official_qa_accuracy": official_score,
            "memory_build_time_ms": prediction.memory_build_time_ms,
            "memory_query_time_ms": prediction.memory_query_time_ms,
            "reader_generation_time_ms": prediction.reader_generation_time_ms,
            "end_to_end_time_ms": prediction.end_to_end_time_ms,
        }
        if prediction.token_usage is not None:
            metrics["api_total_tokens"] = float(prediction.token_usage)
        if episode.gold_evidence:
            metrics["evidence_recall"] = evidence_recall(prediction.evidence, episode.gold_evidence)
        question_type = str(episode.metadata.get("memory_category", ""))
        extra["question_type"] = question_type
        extra["classification"] = longmemeval_v2_classification(question_type)
        extra["domain"] = str(episode.metadata.get("domain", ""))
        extra["dataset"] = episode.dataset
        extra["haystack_tier"] = str(episode.metadata.get("haystack_tier", ""))
        extra["usage_missing"] = bool(prediction.api_usage.get("usage_missing")) or prediction.token_usage is None
        extra["longmemeval_v2_evaluation_mode"] = evaluation_mode
        if official_judge_trace:
            extra["longmemeval_v2_judge_trace"] = official_judge_trace
    elif exp == "m2":
        reference = episode.answers[0] if episode.answers else ""
        fact_scores, fact_trace = locomo_factscore(
            prediction.answer,
            episode.metadata.get("gold_events", []),
            runtime=runtime,
            allow_llm_judge=prediction.api_usage.get("source") != "not_applicable_fake",
        )
        metrics = official_rouge_scores(prediction.answer, reference)
        metrics.update(fact_scores)
        extra.update(
            {
                "dataset": episode.dataset,
                "target_speaker": str(episode.metadata.get("target_speaker", "")),
                "parent_episode_id": str(episode.metadata.get("parent_episode_id", "")),
                "locomo_factscore_trace": fact_trace,
            }
        )
    elif exp == "exp3":
        # Auxiliary only: did the method name the right tool, and how long did it take?
        metrics = {
            "tool_name_match_rate": tool_call_accuracy(prediction, episode),
            "latency_ms": prediction.end_to_end_time_ms,
        }
        extra["auxiliary_only"] = True
    elif exp == "exp4":
        metrics = {
            "exact_match": exact_match(prediction.answer, episode.answers),
            "f1": f1(prediction.answer, episode.answers),
            "rouge_l": rouge_l(prediction.answer, episode.answers),
            "answer_in_evidence_rate": 1.0 if any(str(ans).lower() in " ".join(prediction.evidence).lower() for ans in episode.answers) else 0.0,
        }
        extra["task_type"] = str(episode.metadata.get("task_type", ""))
    elif exp == "exp5":
        metrics = {
            "needle_answer_accuracy": 1.0 if any(ans.lower() in prediction.answer.lower() for ans in episode.answers) else 0.0,
            "latency_ms": prediction.end_to_end_time_ms,
            "token_usage": float(prediction.token_usage or 0),
        }
        extra["context_length_bucket"] = episode.metadata.get("context_length_bucket", 0)
    elif exp == "m3":
        if episode.dataset.startswith("Smoke"):
            extra["smoke_only"] = True
        else:
            raise RuntimeError(
                "M3 must be scored by the official StableToolEval/FAC harness; "
                "project-local per-query metrics are intentionally disabled."
            )
    elif exp == "m4":
        if episode.dataset.startswith("Smoke"):
            extra["smoke_only"] = True
        else:
            raise RuntimeError(
                "M4 must be scored by the official StableToolEval/FAC harness; "
                "project-local per-episode metrics are intentionally disabled."
            )
    elif exp == "a1":
        metrics = {
            "exact_match": exact_match(prediction.answer, episode.answers),
            "f1": f1(prediction.answer, episode.answers),
            "evidence_recall": evidence_recall(prediction.evidence, episode.gold_evidence),
            "graph_dependency_recall": semantic_dependency_recall(
                prediction.dependencies,
                episode.gold_dependencies,
                prediction_graph=prediction.metadata.get("graph"),
                runtime=runtime,
            )[0],
            "token_usage": float(prediction.token_usage or 0),
            "latency_ms": prediction.end_to_end_time_ms,
        }
    elif exp == "a2":
        dep_score, dep_method = semantic_dependency_recall(
            prediction.dependencies,
            episode.gold_dependencies,
            prediction_graph=prediction.metadata.get("graph"),
            runtime=runtime,
        )
        metrics = {
            "exact_match": exact_match(prediction.answer, episode.answers),
            "f1": f1(prediction.answer, episode.answers),
            "supporting_fact_recall": evidence_recall(prediction.evidence, episode.gold_evidence),
            "dependency_recall": dep_score,
            "token_usage": float(prediction.token_usage or 0),
            "latency_ms": prediction.end_to_end_time_ms,
        }
        extra["dependency_match_method"] = dep_method
    elif exp == "a3":
        metrics = {
            "f1": f1(prediction.answer, episode.answers),
            "evidence_recall": evidence_recall(prediction.evidence, episode.gold_evidence),
            "evidence_precision": evidence_precision(prediction.evidence, episode.gold_evidence),
            "token_usage": float(prediction.token_usage or 0),
            "latency_ms": prediction.end_to_end_time_ms,
        }
    elif exp == "a4":
        metrics = {
            "exact_match": exact_match(prediction.answer, episode.answers),
            "f1": f1(prediction.answer, episode.answers),
            "rouge_l": rouge_l(prediction.answer, episode.answers),
            "retrieval_accuracy": 1.0 if any(str(ans).lower() in " ".join(prediction.evidence).lower() for ans in episode.answers) else 0.0,
            "token_usage": float(prediction.token_usage or 0),
            "latency_ms": prediction.end_to_end_time_ms,
        }
        extra["task_type"] = str(episode.metadata.get("task_type", ""))
    elif exp == "a5":
        metrics = {
            "retrieval_accuracy": 1.0 if any(ans.lower() in prediction.answer.lower() for ans in episode.answers) else 0.0,
            "token_usage": float(prediction.token_usage or 0),
            "latency_ms": prediction.end_to_end_time_ms,
        }
        extra["context_length_bucket"] = episode.metadata.get("context_length_bucket", 0)
    if prediction.token_usage is None:
        metrics.pop("token_usage", None)
    return MetricRecord(
        experiment=exp,
        method=prediction.method,
        episode_id=episode.episode_id,
        metrics=metrics,
        model_profile=prediction.model_profile,
        seed=prediction.seed,
        metadata=extra,
    )


def judge_detail_records(
    experiment: str,
    episode: Episode,
    prediction: Prediction,
    record: MetricRecord,
    *,
    judge_model: str = "",
) -> list[dict[str, Any]]:
    details = []
    dependency_result = record.metrics["dependency_recall"] if "dependency_recall" in record.metrics else record.metrics.get("graph_dependency_recall")
    metric_map = {
        "dependency_match_method": ("dependency", episode.gold_dependencies, prediction.dependencies, dependency_result),
    }
    for method_key, (metric, expected, pred_value, result) in metric_map.items():
        match_method = record.metadata.get(method_key)
        if not match_method:
            continue
        details.append({
            "item_id": f"{record.experiment}:{record.episode_id}:{record.method}:{record.seed}:{metric}",
            "metric": metric,
            "expected": expected,
            "prediction": pred_value,
            "judge_result": result,
            "judge_reason": match_method,
            "experiment": experiment,
            "episode_id": episode.episode_id,
            "method": prediction.method,
            "seed": prediction.seed,
            "judge_model": judge_model,
        })
    for metric in ("step_success_rate",):
        if metric not in record.metrics:
            continue
        details.append({
            "item_id": f"{record.experiment}:{record.episode_id}:{record.method}:{record.seed}:{metric}",
            "metric": metric,
            "expected": (
                episode.sidecar.get("evaluator_gold", {}).get("gold_outputs")
                or episode.sidecar.get("gold_outputs")
                or episode.answers
            ) if episode.sidecar else episode.answers,
            "prediction": prediction.answer,
            "judge_result": record.metrics[metric],
            "judge_reason": "rule_first_or_judge_required_field",
            "experiment": experiment,
            "episode_id": episode.episode_id,
            "method": prediction.method,
            "seed": prediction.seed,
            "judge_model": judge_model,
        })
    official_trace = record.metadata.get("longmemeval_v2_judge_trace")
    if isinstance(official_trace, dict) and official_trace:
        details.append({
            "item_id": f"{record.experiment}:{record.episode_id}:{record.method}:{record.seed}:official_qa_accuracy",
            "metric": "official_qa_accuracy",
            "expected": episode.answers,
            "prediction": prediction.answer,
            "judge_result": record.metrics.get("official_qa_accuracy"),
            "judge_reason": record.metadata.get("longmemeval_v2_evaluation_mode", ""),
            "experiment": experiment,
            "episode_id": episode.episode_id,
            "method": prediction.method,
            "seed": prediction.seed,
            "judge_model": official_trace.get("model", judge_model),
            "judge_trace": official_trace,
        })
    factscore_trace = record.metadata.get("locomo_factscore_trace")
    if isinstance(factscore_trace, dict) and factscore_trace:
        details.append({
            "item_id": f"{record.experiment}:{record.episode_id}:{record.method}:{record.seed}:factscore",
            "metric": "factscore",
            "expected": episode.metadata.get("gold_events", []),
            "prediction": prediction.answer,
            "judge_result": {
                key: record.metrics.get(key)
                for key in ("factscore_precision", "factscore_recall", "factscore_f1")
            },
            "judge_reason": factscore_trace.get("evaluation_mode", ""),
            "experiment": experiment,
            "episode_id": episode.episode_id,
            "method": prediction.method,
            "seed": prediction.seed,
            "judge_model": factscore_trace.get("model", judge_model),
            "judge_trace": factscore_trace,
        })
    return details


def aggregate_records(records: list[MetricRecord], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate metrics by run mode, experiment role, experiment, model, and method.

    Keeping run_mode ("llm" vs "fake") as the top-level key means smoke-test runs
    can never silently blend into publishable LLM-run numbers. Experiment and model
    dimensions are also kept separate so main results cannot silently absorb
    auxiliary diagnostics.
    """
    grouped: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = defaultdict(dict)
    fallback_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    failure_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    usage_missing_counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    group_sizes: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    protocol_values: dict[tuple[str, str, str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"dataset": set(), "haystack_tier": set()}
    )
    judge_counts: dict[tuple[str, str, str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    breakdowns: dict[tuple[str, str, str, str, str, str, str], dict[str, list[float]]] = defaultdict(dict)
    paired_values: dict[tuple[str, str, str, str, str, str], dict[tuple[str, int], float]] = defaultdict(dict)
    for record in records:
        role = experiment_role(record.experiment)
        profile = record.model_profile or "default"
        key = (record.run_mode or "llm", role, record.experiment, profile, record.method)
        group_sizes[key] += 1
        if record.fallback:
            fallback_counts[key] += 1
        if record.failure:
            failure_counts[key] += 1
        if record.metadata.get("usage_missing"):
            usage_missing_counts[key] += 1
        for protocol_key in ("dataset", "haystack_tier"):
            value = str(record.metadata.get(protocol_key, "")).strip()
            if value:
                protocol_values[key][protocol_key].add(value)
        method_metrics = grouped[key]
        for metric_key, value in record.metrics.items():
            numeric_value = float(value)
            method_metrics.setdefault(metric_key, []).append(numeric_value)
            paired_values[(record.run_mode or "llm", role, record.experiment, profile, record.method, metric_key)][
                (record.episode_id, record.seed)
            ] = numeric_value
        for meta_key, meta_value in record.metadata.items():
            if meta_key.endswith("_match_method") and meta_value:
                judge_counts[key][str(meta_value)] += 1
        dimensions: list[tuple[str, str]] = []
        if record.experiment == "exp4" and record.metadata.get("task_type"):
            dimensions.append(("task_type", str(record.metadata["task_type"])))
        elif record.experiment == "m1":
            for dimension in ("classification", "question_type", "domain"):
                if record.metadata.get(dimension):
                    dimensions.append((dimension, str(record.metadata[dimension])))
        elif record.experiment == "m3":
            if record.metadata.get("split"):
                dimensions.append(("split", str(record.metadata["split"])))
        for dimension, dimension_value in dimensions:
            bkey = key + (dimension, dimension_value)
            for metric_key, value in record.metrics.items():
                breakdowns[bkey].setdefault(metric_key, []).append(float(value))

    summary: dict[str, Any] = {"metadata": metadata or {}}
    for (run_mode, role, exp, profile, method), metrics in sorted(grouped.items()):
        block = summary.setdefault(run_mode, {}).setdefault(role, {}).setdefault(exp, {}).setdefault(profile, {})
        method_block = {
            metric: {
                "mean": mean(values) if values else 0.0,
                "n": len(values),
                "ci95": normal_ci95(values),
                "bootstrap_ci95": bootstrap_ci95(values),
            }
            for metric, values in metrics.items()
        }
        key = (run_mode, role, exp, profile, method)
        total = group_sizes[key]
        method_block["_fallback_rate"] = (fallback_counts[key] / total) if total else 0.0
        method_block["_failure_rate"] = (failure_counts[key] / total) if total else 0.0
        method_block["_usage_missing_rate"] = (usage_missing_counts[key] / total) if total else 0.0
        method_block["_sample_count"] = total
        if judge_counts[key]:
            method_block["_judge_method_counts"] = dict(sorted(judge_counts[key].items()))
        for (b_run_mode, b_role, b_exp, b_profile, b_method, dim, dim_value), b_metrics in sorted(breakdowns.items()):
            if (b_run_mode, b_role, b_exp, b_profile, b_method) != key:
                continue
            breakdown_payload = {
                metric: {
                    "mean": mean(values) if values else 0.0,
                    "n": len(values),
                    "ci95": normal_ci95(values),
                    "bootstrap_ci95": bootstrap_ci95(values),
                }
                for metric, values in b_metrics.items()
            }
            method_block.setdefault("_breakdowns", {}).setdefault(dim, {})[dim_value] = breakdown_payload
        if exp == "m1":
            if run_mode == "fake":
                method_block["_api_token_reporting_status"] = "not_applicable_fake"
            elif usage_missing_counts[key]:
                method_block.pop("api_total_tokens", None)
                method_block["_api_token_reporting_status"] = "incomplete_not_reported"
            else:
                method_block["_api_token_reporting_status"] = "complete_api_usage"
            method_block["classification_accuracy"] = m1_accuracy_breakdown(
                breakdowns,
                key,
                "classification",
            )
            method_block["domain_accuracy"] = m1_accuracy_breakdown(
                breakdowns,
                key,
                "domain",
            )
            method_block["_latency_accounting"] = m1_latency_accounting(metrics, total)
            method_block["_lafs"] = m1_lafs_gate(
                run_mode=run_mode,
                sample_count=total,
                fallback_count=fallback_counts[key],
                failure_count=failure_counts[key],
                usage_missing_count=usage_missing_counts[key],
                protocol=protocol_values[key],
            )
        block[method] = method_block
    attach_paired_comparisons(summary, paired_values)
    return summary


def m1_accuracy_breakdown(
    breakdowns: dict[tuple[str, str, str, str, str, str, str], dict[str, list[float]]],
    key: tuple[str, str, str, str, str],
    dimension: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for bkey, metric_values in sorted(breakdowns.items()):
        if bkey[:5] != key or bkey[5] != dimension:
            continue
        values = metric_values.get("official_qa_accuracy", [])
        if not values:
            continue
        output[bkey[6]] = {
            "accuracy": mean(values),
            "correct": int(sum(values)),
            "total": len(values),
            "bootstrap_ci95": bootstrap_ci95(values),
        }
    return output


def m1_latency_accounting(metrics: dict[str, list[float]], sample_count: int) -> dict[str, Any]:
    build_values = metrics.get("memory_build_time_ms", [])
    build_events = [value for value in build_values if value > 0]
    query_values = metrics.get("memory_query_time_ms", [])
    reader_values = metrics.get("reader_generation_time_ms", [])
    return {
        "memory_build_event_count": len(build_events),
        "memory_build_total_ms": sum(build_values),
        "memory_build_cold_mean_ms": mean(build_events) if build_events else 0.0,
        "amortized_end_to_end_time_ms": (
            (sum(build_values) + sum(query_values) + sum(reader_values)) / sample_count
            if sample_count
            else 0.0
        ),
        "definition": "(unique memory builds + all memory queries + all reader generations) / questions",
    }


def m1_lafs_gate(
    *,
    run_mode: str,
    sample_count: int,
    fallback_count: int,
    failure_count: int,
    usage_missing_count: int,
    protocol: dict[str, set[str]],
) -> dict[str, Any]:
    requirements = {
        "run_mode_llm": run_mode == "llm",
        "dataset_longmemeval_v2_text": protocol["dataset"] == {"LongMemEval-V2-Text"},
        "tier_small": protocol["haystack_tier"] == {"small"},
        "complete_422_questions": sample_count == 422,
        "no_fallback": fallback_count == 0,
        "no_failed_questions": failure_count == 0,
        "complete_api_usage": usage_missing_count == 0,
        # The released fixed frontier is for the complete 451-question benchmark.
        # It is not a corresponding reference frontier for this 422-question text subset.
        "matching_text_reference_frontier": False,
    }
    failed = [name for name, passed in requirements.items() if not passed]
    return {
        "status": "not_reported" if failed else "eligible",
        "reason": "requirements_not_met: " + ", ".join(failed) if failed else "all_requirements_met",
        "requirements": requirements,
        "reference_policy": "Do not reuse the released 451-question frontier for LongMemEval-V2-Text.",
    }


def evidence_precision(predicted: list[str], gold: list[str]) -> float:
    if not predicted:
        return 0.0
    return sum(1 for item in predicted if any(evidence_item_match(item, target) for target in gold)) / len(predicted)


def evidence_recall(predicted: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return sum(1 for item in gold if any(evidence_item_match(source, item) for source in predicted)) / len(gold)


def evidence_item_match(predicted: str, gold: str) -> bool:
    pred = normalize_text(predicted)
    target = normalize_text(gold)
    if not pred or not target:
        return False
    shorter, longer = sorted((pred, target), key=len)
    if len(shorter) >= 24 and shorter in longer:
        return True
    pred_terms = content_terms(pred)
    gold_terms = content_terms(target)
    if not pred_terms or not gold_terms:
        return False
    overlap = len(pred_terms & gold_terms)
    return overlap / len(gold_terms) >= 0.35 or overlap / len(pred_terms) >= 0.60


def tool_call_accuracy(prediction: Prediction, episode: Episode) -> float | None:
    tools = episode.metadata.get("tools") or []
    gold_name = str(episode.metadata.get("gold_tool_name", "")).strip()
    gold_api = str(episode.metadata.get("gold_api_name", "")).strip()
    if not gold_name and len(tools) == 1:
        only = tools[0]
        gold_name = str(only.get("name", "") if isinstance(only, dict) else only).strip()
        gold_api = str(only.get("api_name", "") if isinstance(only, dict) else "").strip()
    if not gold_name and not gold_api:
        return None
    predicted_name, predicted_api = extract_tool_identity(prediction, tools)
    if not predicted_name and not predicted_api:
        return 0.0
    name_match = not gold_name or normalize_text(predicted_name) == normalize_text(gold_name)
    api_match = not gold_api or normalize_text(predicted_api) == normalize_text(gold_api)
    return 1.0 if name_match and api_match else 0.0


_LONGMEMEVAL_V2_METRICS_MODULE: Any | None = None

LONGMEMEVAL_V2_CLASSIFICATION_MAP = {
    "static-environment": "static",
    "static-environment-abs": "static",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic",
    "procedure": "procedure",
    "procedure-abs": "procedure",
    "errors-gotchas": "gotchas",
}


def longmemeval_v2_classification(question_type: str) -> str:
    """Map released question types to the official leaderboard categories."""
    return LONGMEMEVAL_V2_CLASSIFICATION_MAP.get(question_type, "unclassified")


def longmemeval_v2_official_score(
    prediction: Prediction,
    episode: Episode,
    *,
    runtime: RuntimeConfig | None,
) -> tuple[float, str, dict[str, Any]]:
    """Evaluate with the released LongMemEval-V2 per-question eval specification."""
    eval_spec = str(episode.metadata.get("eval_function", "")).strip()
    if not eval_spec:
        return exact_match(prediction.answer, episode.answers), "legacy_exact_match", {}
    if not episode.answers:
        raise ValueError(f"LongMemEval-V2 episode {episode.episode_id} has no reference answer")

    official = _load_longmemeval_v2_metrics_module()
    eval_name = str(official.eval_name(eval_spec))
    parsed_prediction = str(official.extract_boxed_answer(prediction.answer))
    # The released qa_eval_metrics.py contains the LLM evaluator functions but the
    # LLM_EVAL_FUNCTIONS registry lives in evaluation/harness.py. Keep the names
    # here as well so direct reuse of the official scoring module still supplies
    # the required evaluator configuration.
    llm_eval_names = set(getattr(official, "LLM_EVAL_FUNCTIONS", set())) | {
        "llm_abstention_checker",
        "llm_gotchas_checker",
    }
    eval_kwargs: dict[str, Any] = {}
    judge_trace: dict[str, Any] = {}
    prediction_for_eval = parsed_prediction
    evaluation_mode = f"official_rule:{eval_name}"
    if eval_name in llm_eval_names:
        if runtime is None or not runtime.model or not runtime.api_key:
            raise RuntimeError(
                f"LongMemEval-V2 evaluator {eval_name} requires a configured judge model and API key"
            )
        prediction_for_eval = prediction.answer
        eval_kwargs = {
            "question_item": {
                "id": episode.episode_id,
                "question": episode.query,
                "question_type": episode.metadata.get("memory_category", ""),
            },
            "parsed_prediction": parsed_prediction,
            "model_response": prediction.answer,
            "evaluator_model": runtime.model,
            "evaluator_base_url": runtime.base_url,
            "evaluator_api_key": runtime.api_key,
            "evaluator_api_key_env": runtime.api_key_env,
            "evaluator_max_completion_tokens": runtime.max_tokens,
        }
        evaluation_mode = f"official_llm_judge:{eval_name}"
    original_call = None
    if eval_name in llm_eval_names:
        original_call = official._call_chat_completion

        def traced_call_chat_completion(**kwargs: Any) -> str:
            started = time.perf_counter()
            captured: dict[str, Any] = {}
            traced_kwargs = dict(kwargs)
            if kwargs.get("client") is not None and hasattr(kwargs["client"], "chat"):
                traced_kwargs["client"] = _UsageCapturingClient(kwargs["client"], captured)
            response_text = original_call(**traced_kwargs)
            parsed_usage = extract_api_usage(captured.get("usage"))
            judge_trace.update({
                "evaluation_mode": "official_longmemeval_v2_llm_judge",
                "eval_name": eval_name,
                "eval_spec": eval_spec,
                "model": str(kwargs.get("model", "")),
                "messages": kwargs.get("messages", []),
                "raw_response": response_text,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": parsed_usage,
                "usage_missing": parsed_usage is None,
            })
            return response_text

        official._call_chat_completion = traced_call_chat_completion
    try:
        score_raw = official.eval_from_spec(
            eval_spec,
            prediction_for_eval,
            episode.answers[0],
            **eval_kwargs,
        )
    finally:
        if original_call is not None:
            official._call_chat_completion = original_call
    score = 1.0 if bool(official.score_to_bool(score_raw)) else 0.0
    if bool(official.is_unknown(parsed_prediction)):
        score = 0.0
    if judge_trace:
        judge_trace["parsed_score"] = score
    return score, evaluation_mode, judge_trace


class _UsageCapturingCompletions:
    def __init__(self, completions: Any, capture: dict[str, Any]) -> None:
        self._completions = completions
        self._capture = capture

    def create(self, **kwargs: Any) -> Any:
        response = self._completions.create(**kwargs)
        self._capture["usage"] = getattr(response, "usage", None)
        return response


class _UsageCapturingChat:
    def __init__(self, chat: Any, capture: dict[str, Any]) -> None:
        self.completions = _UsageCapturingCompletions(chat.completions, capture)


class _UsageCapturingClient:
    def __init__(self, client: Any, capture: dict[str, Any]) -> None:
        self._client = client
        self.chat = _UsageCapturingChat(client.chat, capture)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _load_longmemeval_v2_metrics_module() -> Any:
    global _LONGMEMEVAL_V2_METRICS_MODULE
    if _LONGMEMEVAL_V2_METRICS_MODULE is not None:
        return _LONGMEMEVAL_V2_METRICS_MODULE
    path = (
        RAW_DIR
        / "repos"
        / "longmemeval_v2"
        / "repository"
        / "evaluation"
        / "qa_eval_metrics.py"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Official LongMemEval-V2 evaluator is missing. Clone xiaowu0162/LongMemEval-V2 under "
            f"{path.parents[2]}"
        )
    spec = importlib.util.spec_from_file_location("mlg_longmemeval_v2_official_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official LongMemEval-V2 evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LONGMEMEVAL_V2_METRICS_MODULE = module
    return module


def step_order_accuracy(prediction: Prediction, episode: Episode) -> float:
    sequence = [normalize_action(str(item)) for item in episode.metadata.get("action_sequence", [])]
    sequence = [item for item in sequence if item]
    if not sequence:
        return 1.0
    predicted = extract_actions(prediction)
    if not predicted:
        return 0.0
    return lcs_ratio(predicted, sequence)


def action_f1(prediction: Prediction, episode: Episode) -> float:
    answers = episode.answers or episode.metadata.get("action_sequence", [])
    return f1(prediction.answer, [str(item) for item in answers])


def subgoal_completion_rate(prediction: Prediction, episode: Episode) -> float:
    if prediction.answer and exact_match(prediction.answer, episode.answers):
        return 1.0
    gold_action = normalize_action(episode.answers[0] if episode.answers else "")
    if not gold_action:
        return 1.0 if action_f1(prediction, episode) >= 0.5 else 0.0
    return 1.0 if gold_action in extract_actions(prediction) else 0.0


def extract_tool_identity(prediction: Prediction, tools: list[Any]) -> tuple[str, str]:
    raw = prediction.metadata.get("llm_raw", {})
    if isinstance(raw, dict):
        name = raw.get("tool_name") or raw.get("tool")
        api = raw.get("api_name") or raw.get("api")
        if name or api:
            return str(name or ""), str(api or "")
    name_match = re.search(r"\btool_name\s*[:=]\s*([a-z0-9_.-]+)", prediction.answer, re.I)
    api_match = re.search(r"\bapi_name\s*[:=]\s*([a-z0-9_.-]+)", prediction.answer, re.I)
    if name_match or api_match:
        return name_match.group(1) if name_match else "", api_match.group(1) if api_match else ""
    mentioned = mentioned_candidate_tools(prediction, tools)
    if len(mentioned) == 1:
        return mentioned[0]
    return "", ""


def mentioned_candidate_tools(prediction: Prediction, tools: list[Any]) -> list[tuple[str, str]]:
    text = prediction_text_blob(prediction)
    mentioned: list[tuple[str, str]] = []
    for tool in tools:
        name = str(tool.get("name", "") if isinstance(tool, dict) else tool)
        api = str(tool.get("api_name", "") if isinstance(tool, dict) else "")
        if name and normalize_text(name) in text:
            mentioned.append((name, api))
    return mentioned


def normalize_text(text: object) -> str:
    return " ".join(re.findall(r"[a-z0-9_\-\u4e00-\u9fff]+", str(text).lower()))


def content_terms(text: object) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "you", "your", "are", "was", "were",
        "have", "has", "had", "from", "into", "will", "would", "about", "there", "their",
        "what", "when", "where", "which", "then", "than", "they", "them", "been",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9_\-\u4e00-\u9fff]{3,}", str(text).lower())
        if term not in stop
    }


def normalize_action(text: object) -> str:
    raw = str(text)
    match = re.search(r"\b(click|type|select|hover|press|scroll|navigate|enter|upload)\b[^\n,;]*", raw, re.I)
    if not match:
        return ""
    action = match.group(1).upper()
    element = action_element(match.group(0))
    value = action_value(match.group(0))
    parts = [action]
    if element:
        parts.append(f"element={element}")
    if value:
        parts.append(f"value={value}")
    return " ".join(parts)


def action_element(text: object) -> str:
    raw = str(text)
    match = re.search(r"element\s*=\s*([a-z0-9_\-:.#]+)", raw, re.I)
    if match:
        return match.group(1).lower()
    return ""


def action_type(text: object) -> str:
    action = normalize_action(text)
    return action.split(" ", 1)[0].upper() if action else ""


def action_value(text: object) -> str:
    raw = str(text)
    match = re.search(r"value\s*=\s*([^,;\n]+)", raw, re.I)
    return normalize_text(match.group(1)) if match else ""


def extract_actions(prediction: Prediction) -> list[str]:
    chunks = [prediction.answer, *prediction.evidence]
    raw = prediction.metadata.get("llm_raw", {})
    if isinstance(raw, dict):
        raw_action = raw.get("action_type") or raw.get("action") or raw.get("operation")
        raw_target = raw.get("target_element") or raw.get("element") or raw.get("element_id")
        raw_value = raw.get("value") or raw.get("input_value")
        if raw_action or raw_target or raw_value:
            parts = [str(raw_action or "ACTION").upper()]
            if raw_target:
                parts.append(f"element={raw_target}")
            if raw_value:
                parts.append(f"value={raw_value}")
            chunks.append(" ".join(parts))
    for dep in prediction.dependencies:
        if isinstance(dep, dict):
            chunks.extend(str(dep.get(key, "")) for key in ("source", "target", "source_content", "target_content"))
    actions: list[str] = []
    for chunk in chunks:
        for match in re.finditer(r"\b(click|type|select|hover|press|scroll|navigate|enter|upload)\b[^\n,;]*", str(chunk), re.I):
            action = normalize_action(match.group(0))
            if action:
                actions.append(action)
    return actions


def lcs_ratio(predicted: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    if not predicted:
        return 0.0
    prev = [0] * (len(gold) + 1)
    for pred in predicted:
        cur = [0]
        for idx, target in enumerate(gold, start=1):
            cur.append(prev[idx - 1] + 1 if pred == target else max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1] / len(gold)


def prediction_text_blob(prediction: Prediction) -> str:
    raw = {
        "answer": prediction.answer,
        "evidence": prediction.evidence,
        "dependencies": prediction.dependencies,
        "llm_raw": prediction.metadata.get("llm_raw", {}),
    }
    return normalize_text(json.dumps(raw, ensure_ascii=False))


def normal_ci95(values: list[float]) -> float:
    """Normal-approximation 95% half-width (1.96 * sigma / sqrt(n)).

    This is NOT a bootstrap resampling CI -- it is a plain normal approximation, so
    treat it as a rough guide for small n or skewed 0/1 metrics (n < 30).
    """
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return 1.96 * (variance ** 0.5) / (len(values) ** 0.5)


def bootstrap_ci95(values: list[float], *, samples: int = 1000, seed: int = 13) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(draw))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return [lo, hi]


def cluster_bootstrap_ci95(
    clusters: list[list[float]], *, samples: int = 1000, seed: int = 13
) -> list[float]:
    """Bootstrap source episodes while retaining all dependent decision rounds."""
    nonempty = [cluster for cluster in clusters if cluster]
    if not nonempty:
        return [0.0, 0.0]
    observed = mean([value for cluster in nonempty for value in cluster])
    if len(nonempty) == 1:
        return [observed, observed]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sampled_clusters = [nonempty[rng.randrange(len(nonempty))] for _ in range(len(nonempty))]
        draw = [value for cluster in sampled_clusters for value in cluster]
        means.append(mean(draw))
    means.sort()
    return [
        means[int(0.025 * (samples - 1))],
        means[int(0.975 * (samples - 1))],
    ]


def attach_paired_comparisons(
    summary: dict[str, Any],
    paired_values: dict[tuple[str, str, str, str, str, str], dict[tuple[str, int], float]],
) -> None:
    groups = sorted({key[:4] for key in paired_values})
    for run_mode, role, exp, profile in groups:
        block = summary.get(run_mode, {}).get(role, {}).get(exp, {}).get(profile, {})
        if not isinstance(block, dict) or "ours_full" not in block:
            continue
        methods = [method for method in block if not method.startswith("_") and method != "ours_full"]
        metrics = {
            key[5]
            for key in paired_values
            if key[:5] == (run_mode, role, exp, profile, "ours_full")
        }
        for metric in sorted(metrics):
            ours = paired_values.get((run_mode, role, exp, profile, "ours_full", metric), {})
            for method in methods:
                other = paired_values.get((run_mode, role, exp, profile, method, metric), {})
                comparison = paired_comparison(ours, other)
                if comparison["n"] >= 2:
                    block.setdefault("_comparisons", {}).setdefault(metric, {})[
                        f"ours_full_vs_{method}"
                    ] = comparison


def paired_comparison(
    ours: dict[tuple[str, int], float],
    other: dict[tuple[str, int], float],
) -> dict[str, Any]:
    common = sorted(set(ours) & set(other))
    if len(common) < 2:
        return {"n": len(common), "mean_delta": 0.0, "p_value": None, "test": "insufficient_pairs"}
    deltas = [ours[key] - other[key] for key in common]
    observed = mean(deltas)
    p_value = paired_permutation_p_value(deltas)
    return {
        "n": len(common),
        "mean_delta": observed,
        "p_value": p_value,
        "significant": bool(p_value is not None and p_value < 0.05),
        "test": "paired_sign_flip_permutation",
    }


def paired_permutation_p_value(deltas: list[float], *, max_exact: int = 14, samples: int = 4096) -> float | None:
    if not deltas:
        return None
    observed = abs(mean(deltas))
    if len(deltas) <= max_exact:
        signs_iter = itertools.product((-1, 1), repeat=len(deltas))
    else:
        rng = random.Random(17)
        signs_iter = ([rng.choice((-1, 1)) for _ in deltas] for _ in range(samples))
    count = 0
    extreme = 0
    for signs in signs_iter:
        count += 1
        permuted = abs(mean([delta * sign for delta, sign in zip(deltas, signs)]))
        if permuted >= observed:
            extreme += 1
    return extreme / count if count else None
