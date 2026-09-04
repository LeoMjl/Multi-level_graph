from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlg.experiments.registry import METHOD_DISPLAY_NAMES

# Threshold above which a method's fallback rate is flagged in the report.
# When more than this fraction of a method's LLM-run samples fell back to the
# deterministic keyword path, the headline score is mostly NOT measuring the
# LLM's actual ability, so reviewers must be warned.
FALLBACK_WARN_THRESHOLD = 0.20


def summarize_run(run_dir: Path) -> str:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = ["# Multi-level_graph Run Summary", ""]

    if _is_nested(summary):
        _render_nested_summary(lines, summary)
    elif _is_grouped(summary):
        for run_mode, methods in sorted((k, v) for k, v in summary.items() if k != "metadata"):
            lines.append(f"# Run mode: {run_mode}")
            lines.append("")
            for method, metrics in sorted(methods.items()):
                _render_method(lines, method, metrics, warn_fallback=(run_mode == "llm"))
            lines.append("")
    else:
        for method, metrics in sorted(summary.items()):
            _render_method(lines, method, metrics, warn_fallback=False)
        lines.append("")

    report = "\n".join(lines).strip() + "\n"
    (run_dir / "summary.md").write_text(report, encoding="utf-8")
    return report


def _is_grouped(summary: dict[str, Any]) -> bool:
    """Detect the new {run_mode: {method: {...}}} layout vs legacy {method: {...}}.

    We treat the top-level value as a group if it maps to a dict whose values are
    themselves dicts containing metric entries (dicts with 'mean').
    """
    for value in summary.values():
        if not isinstance(value, dict):
            return False
        for inner in value.values():
            if isinstance(inner, dict) and any(
                isinstance(v, dict) and "mean" in v for v in inner.values()
            ):
                return True
        return False
    return False


def _is_nested(summary: dict[str, Any]) -> bool:
    for run_mode, roles in summary.items():
        if run_mode == "metadata":
            continue
        if not isinstance(roles, dict):
            continue
        if any(role in roles for role in ("main", "auxiliary")):
            return True
    return False


def _render_nested_summary(lines: list[str], summary: dict[str, Any]) -> None:
    metadata = summary.get("metadata", {})
    if metadata:
        lines.append("## Run Metadata")
        for key in ("suite", "requested_methods", "experiment_methods", "skipped_methods", "model_profiles", "judge_profile"):
            if key in metadata:
                lines.append(f"- {key}: {metadata[key]}")
        lines.append("")

    lines.append("## Main Experiment Results")
    _render_role(lines, summary, "main")

    lines.append("## Model Comparison")
    _render_model_comparison(lines, summary)

    lines.append("## Auxiliary Diagnostics")
    _render_role(lines, summary, "auxiliary")


def _render_role(lines: list[str], summary: dict[str, Any], role: str) -> None:
    rendered = False
    for run_mode, roles in sorted((k, v) for k, v in summary.items() if k != "metadata"):
        experiments = roles.get(role, {}) if isinstance(roles, dict) else {}
        if not experiments:
            continue
        rendered = True
        lines.append(f"### Run mode: {run_mode}")
        for experiment, profiles in sorted(experiments.items()):
            lines.append(f"#### {experiment}")
            for profile, methods in sorted(profiles.items()):
                lines.append(f"##### Model profile: {profile}")
                for method, metrics in sorted(methods.items()):
                    if method.startswith("_"):
                        continue
                    _render_method(lines, method, metrics, warn_fallback=(run_mode == "llm"))
    if not rendered:
        lines.append("- No records.")
        lines.append("")


def _render_model_comparison(lines: list[str], summary: dict[str, Any]) -> None:
    key_metrics = (
        "official_qa_accuracy",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "factscore_precision",
        "factscore_recall",
        "factscore_f1",
    )
    rendered = False
    for run_mode, roles in sorted((k, v) for k, v in summary.items() if k != "metadata"):
        main = roles.get("main", {}) if isinstance(roles, dict) else {}
        for experiment, profiles in sorted(main.items()):
            rendered = True
            lines.append(f"### {run_mode} / {experiment}")
            for profile, methods in sorted(profiles.items()):
                for method, metrics in sorted(methods.items()):
                    if method.startswith("_"):
                        continue
                    parts = []
                    for metric in key_metrics:
                        payload = metrics.get(metric)
                        if isinstance(payload, dict):
                            parts.append(f"{metric}={payload.get('mean', 0):.4f}")
                    if parts:
                        lines.append(f"- {profile} / {method}: " + ", ".join(parts))
            lines.append("")
    if not rendered:
        lines.append("- No main experiment records.")
        lines.append("")


def _render_method(lines: list[str], method: str, metrics: dict[str, Any], *, warn_fallback: bool) -> None:
    lines.append(f"## {METHOD_DISPLAY_NAMES.get(method, method)}")
    fallback_rate = metrics.get("_fallback_rate", 0.0)
    failure_rate = metrics.get("_failure_rate", 0.0)
    sample_count = metrics.get("_sample_count", None)
    for metric, payload in sorted(metrics.items()):
        if metric.startswith("_"):
            continue
        if not isinstance(payload, dict):
            lines.append(f"- {metric}: {payload}")
            continue
        if "mean" not in payload:
            lines.append(f"- {metric}: {json.dumps(payload, ensure_ascii=False)}")
            continue
        status = str(payload.get("status", ""))
        if status.startswith("not_reportable"):
            lines.append(
                f"- {metric}: status={status}, "
                f"observed_task_n={payload.get('observed_task_n', 0)}, "
                f"complete_task_n={payload.get('complete_task_n', 0)}"
            )
            continue
        lines.append(
            f"- {metric}: mean={payload.get('mean', 0):.4f}, "
            f"n={payload.get('n', 0)}, ci95={payload.get('ci95', 0):.4f}, "
            f"bootstrap_ci95={payload.get('bootstrap_ci95', [0, 0])}"
        )
    if sample_count is not None:
        lines.append(f"- _sample_count: {sample_count}")
    if metrics.get("_judge_method_counts"):
        lines.append(f"- _judge_method_counts: {metrics['_judge_method_counts']}")
    if "_toolenv_availability_rate" in metrics:
        lines.append(f"- _toolenv_availability_rate: {metrics['_toolenv_availability_rate']:.2%}")
    if "_environment_failure_rate" in metrics:
        lines.append(f"- _environment_failure_rate: {metrics['_environment_failure_rate']:.2%}")
    if metrics.get("_breakdowns"):
        lines.append(f"- _breakdowns: {metrics['_breakdowns']}")
    if warn_fallback and fallback_rate > FALLBACK_WARN_THRESHOLD:
        lines.append(
            f"- _fallback_rate: {fallback_rate:.2%}  "
            f"WARNING: >{FALLBACK_WARN_THRESHOLD:.0%} of samples used keyword fallback; "
            "headline scores may not reflect the LLM's own ability."
        )
    else:
        lines.append(f"- _fallback_rate: {fallback_rate:.2%}")
    lines.append(f"- _failure_rate: {failure_rate:.2%}")
    if "_usage_missing_rate" in metrics:
        lines.append(f"- _usage_missing_rate: {metrics['_usage_missing_rate']:.2%}")
    if "_api_token_reporting_status" in metrics:
        lines.append(f"- _api_token_reporting_status: {metrics['_api_token_reporting_status']}")
    if metrics.get("_lafs"):
        lines.append(f"- _lafs: {json.dumps(metrics['_lafs'], ensure_ascii=False)}")
    lines.append("")
