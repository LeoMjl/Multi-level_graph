from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mlg.m4_data import M4_DIR
from mlg.m4_eval import paired_reliability, score_binary_judgments
from mlg.m4_graphrag import GRAPHRAG_COMMIT
from mlg.m4_pipeline import METHODS, _slug
from mlg.m4_raptor import RAPTOR_COMMIT
from mlg.m4_retrieval import read_jsonl


def build_report(output_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "m4-target-paper-results-v5",
        "protocol": {
            "reader_and_judge": "deepseek-v4-flash", "temperature": 0.0, "thinking": "disabled",
            "query_isolation": "batched JSON with independent query_id/context pairs; Full Text shares context only within one document",
            "embedding": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
            "retrieval": {
                "raptor_memorystream_taskgraph_k": 10,
                "graphrag": "official local search from top-10 entities with mixed graph and source-text context",
                "context_tokens": 8192,
            },
            "memorystream": "LLM-generated semantic keys embedded for flat lookup; original chunks retained as values",
            "full_text": "complete QuALITY article without retrieval truncation",
            "chunking": {"quality": 512, "multihop_rag": 1024, "overlap": 0},
            "quality": "open-ended; source answer options hidden from methods",
            "multihop_rag": "2,255 non-null queries only",
            "oracle_policy": "answers, difficulty, types, and evidence are evaluator-only",
            "official_code": {"raptor": RAPTOR_COMMIT, "graphrag": GRAPHRAG_COMMIT},
        },
        "datasets": {}, "reliability": {}, "artifacts": {},
    }
    scored: dict[str, dict[str, dict]] = {}
    for dataset, methods in METHODS.items():
        labels = read_jsonl(M4_DIR / f"{dataset}_labels.jsonl")
        expected = len(labels)
        scored[dataset] = {}
        for method in methods:
            prediction_path = output_dir / f"{dataset}_{_slug(method)}_predictions.jsonl"
            judgment_path = output_dir / f"{dataset}_{_slug(method)}_judgments.jsonl"
            predictions = read_jsonl(prediction_path)
            judgments = read_jsonl(judgment_path)
            if len(predictions) != expected or len(judgments) != expected:
                raise ValueError(f"{dataset}/{method} is incomplete")
            result = score_binary_judgments(dataset, predictions, judgments, labels)
            scored[dataset][method] = result
            for path in (prediction_path, judgment_path):
                report["artifacts"][path.name] = {"rows": expected, "sha256": _sha(path)}
        report["datasets"][dataset] = {
            method: {key: value for key, value in result.items() if key != "correctness"}
            for method, result in scored[dataset].items()
        }
        strongest = max(
            (method for method in methods if method != "TaskGraph"),
            key=lambda method: scored[dataset][method]["binary_accuracy"]["value"],
        )
        label_by_id = {row["query_id"]: row for row in labels}
        groups = ["easy", "hard"] if dataset == "quality" else ["inference", "comparison", "temporal"]
        group_ids = {
            group: {
                query_id for query_id, label in label_by_id.items()
                if ("hard" if dataset == "quality" and label["difficult"] else "easy" if dataset == "quality"
                    else label["question_type"].replace("_query", "")) == group
            }
            for group in groups
        }
        comparisons = {}
        for baseline in (method for method in methods if method != "TaskGraph"):
            comparison = {
                "overall": paired_reliability(
                    scored[dataset][baseline]["correctness"], scored[dataset]["TaskGraph"]["correctness"]
                )
            }
            for group, ids in group_ids.items():
                comparison[group] = paired_reliability(
                    {key: value for key, value in scored[dataset][baseline]["correctness"].items() if key in ids},
                    {key: value for key, value in scored[dataset]["TaskGraph"]["correctness"].items() if key in ids},
                )
            comparisons[baseline] = comparison
        report["reliability"][dataset] = {
            "strongest_baseline": strongest, "comparisons": comparisons,
        }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# M4 target-paper protocol results", "", "## QuALITY", "",
        "| Method | Easy | Hard | Overall | ROUGE-L (R) |", "|---|---:|---:|---:|---:|",
    ]
    for method, result in report["datasets"]["quality"].items():
        groups = result["by_group"]
        lines.append(f"| {method} | {_pct(groups['easy']['binary_accuracy']['value'])} | {_pct(groups['hard']['binary_accuracy']['value'])} | {_pct(result['binary_accuracy']['value'])} | {_pct(result['rouge_l_recall']['value'])} |")
    lines += [
        "", "## MultiHop-RAG", "",
        "| Method | Inference | Comparison | Temporal | Overall | ROUGE-L (R) |", "|---|---:|---:|---:|---:|---:|",
    ]
    for method, result in report["datasets"]["multihop_rag"].items():
        groups = result["by_group"]
        lines.append(f"| {method} | {_pct(groups['inference']['binary_accuracy']['value'])} | {_pct(groups['comparison']['binary_accuracy']['value'])} | {_pct(groups['temporal']['binary_accuracy']['value'])} | {_pct(result['binary_accuracy']['value'])} | {_pct(result['rouge_l_recall']['value'])} |")
    for dataset, item in report["reliability"].items():
        lines += ["", f"## {dataset} paired reliability", "", f"Strongest baseline: {item['strongest_baseline']}", ""]
        for baseline, comparison in item["comparisons"].items():
            lines.append(f"### TaskGraph vs. {baseline}")
            for subset, values in comparison.items():
                lines.append(f"- {subset}: delta={_pct(values['accuracy_delta'])}, bootstrap 95% CI={_pct(values['bootstrap_ci95'][0])} to {_pct(values['bootstrap_ci95'][1])}, McNemar p={values['mcnemar_exact_p']:.6g}")
    return "\n".join(lines) + "\n"


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
