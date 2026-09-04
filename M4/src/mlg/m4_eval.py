from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Callable

from scipy.stats import binomtest
import numpy as np


def quality_correct(prediction: str, gold: str) -> bool:
    match = re.search(r"\b([ABCD])\b", str(prediction).upper())
    return bool(match and match.group(1) == gold)


def multihop_official_correct(prediction: str, gold: str) -> bool:
    return bool(set(prediction.lower().split()).intersection(gold.lower().split()))


def normalized_exact_correct(prediction: str, gold: str) -> bool:
    normalize = lambda text: " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))
    return normalize(prediction) == normalize(gold)


def rouge_l_recall(prediction: str, gold: str) -> float:
    predicted = re.findall(r"\w+", prediction.lower(), flags=re.UNICODE)
    reference = re.findall(r"\w+", gold.lower(), flags=re.UNICODE)
    if not reference:
        return float(not predicted)
    previous = [0] * (len(predicted) + 1)
    for gold_token in reference:
        current = [0]
        for index, predicted_token in enumerate(predicted, 1):
            current.append(previous[index - 1] + 1 if gold_token == predicted_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1] / len(reference)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _accuracy(rows: list[bool]) -> dict[str, Any]:
    successes, total = sum(rows), len(rows)
    return {"value": successes / total if total else 0.0, "correct": successes, "n": total, "ci95": wilson_interval(successes, total)}


def score_binary_judgments(
    dataset: str, predictions: list[dict], judgments: list[dict], labels: list[dict],
) -> dict[str, Any]:
    prediction_by_id = {row["query_id"]: row for row in predictions}
    judgment_by_id = {row["query_id"]: row for row in judgments}
    label_by_id = {row["query_id"]: row for row in labels}
    if set(prediction_by_id) != set(judgment_by_id) or set(prediction_by_id) != set(label_by_id):
        raise ValueError(f"{dataset} predictions, judgments, and labels must contain identical query IDs")
    correctness, rouge = {}, {}
    grouped_correct: dict[str, list[bool]] = defaultdict(list)
    grouped_rouge: dict[str, list[float]] = defaultdict(list)
    for query_id, prediction in prediction_by_id.items():
        judgment = judgment_by_id[query_id]
        digest = hashlib.sha256(str(prediction.get("answer", "")).encode("utf-8")).hexdigest()
        if judgment.get("prediction_sha256") != digest:
            raise ValueError(f"Stale judgment for {query_id}")
        correct = bool(int(judgment["correct"]))
        score = rouge_l_recall(str(prediction.get("answer", "")), str(label_by_id[query_id]["answer"]))
        group = (
            "hard" if dataset == "quality" and label_by_id[query_id]["difficult"]
            else "easy" if dataset == "quality"
            else str(label_by_id[query_id]["question_type"]).replace("_query", "")
        )
        correctness[query_id] = correct
        rouge[query_id] = score
        grouped_correct[group].append(correct)
        grouped_rouge[group].append(score)
    return {
        "binary_accuracy": _accuracy(list(correctness.values())),
        "rouge_l_recall": _mean_metric(list(rouge.values())),
        "by_group": {
            group: {"binary_accuracy": _accuracy(values), "rouge_l_recall": _mean_metric(grouped_rouge[group])}
            for group, values in sorted(grouped_correct.items())
        },
        "correctness": correctness,
    }


def _mean_metric(values: list[float]) -> dict[str, Any]:
    return {"value": float(np.mean(values)) if values else 0.0, "n": len(values)}


def score_quality(predictions: list[dict], labels: list[dict]) -> dict[str, Any]:
    label_by_id = {row["query_id"]: row for row in labels}
    correctness = {}
    hard = []
    for row in predictions:
        label = label_by_id[row["query_id"]]
        correct = quality_correct(str(row.get("answer", "")), label["answer"])
        correctness[row["query_id"]] = correct
        if label["difficult"]:
            hard.append(correct)
    return {"accuracy": _accuracy(list(correctness.values())), "hard_accuracy": _accuracy(hard), "correctness": correctness}


def score_multihop(predictions: list[dict], labels: list[dict]) -> dict[str, Any]:
    label_by_id = {row["query_id"]: row for row in labels}
    official, exact = {}, {}
    by_type: dict[str, list[bool]] = defaultdict(list)
    for row in predictions:
        label = label_by_id[row["query_id"]]
        answer = str(row.get("answer", ""))
        correct = multihop_official_correct(answer, label["answer"])
        official[row["query_id"]] = correct
        exact[row["query_id"]] = normalized_exact_correct(answer, label["answer"])
        by_type[label["question_type"]].append(correct)
    result = {
        "official_qa_accuracy": _accuracy(list(official.values())),
        "normalized_exact_match": _accuracy(list(exact.values())),
        "by_type": {kind: _accuracy(values) for kind, values in sorted(by_type.items())},
        "correctness": official,
    }
    if predictions and predictions[0].get("retrieval"):
        result["official_retrieval"] = official_retrieval_metrics(predictions, label_by_id)
    return result


def official_retrieval_metrics(predictions: list[dict], label_by_id: dict[str, dict]) -> dict[str, Any]:
    hits10 = hits4 = 0
    average_precisions, reciprocal_ranks = [], []
    for row in predictions:
        label = label_by_id[row["query_id"]]
        if label["question_type"] == "null_query":
            continue
        gold = [_compact(item) for item in label["evidence_facts"]]
        retrieved = [_compact(item.get("text", "")) for item in row.get("retrieval", [])]
        found, ap_sum, first = [], 0.0, None
        for rank, item in enumerate(retrieved[:10], 1):
            matched = [fact for fact in gold if fact in item]
            if not matched:
                continue
            hits10 += int(first is None)
            if first is None:
                first = rank
                hits4 += int(rank <= 4)
            new = [fact for fact in matched if fact not in found]
            found.extend(new)
            ap_sum += len(new) / rank
        average_precisions.append(ap_sum / min(len(gold), 10))
        reciprocal_ranks.append(1 / first if first else 0.0)
    total = len(average_precisions)
    return {
        "Hits@10": hits10 / total,
        "Hits@4": hits4 / total,
        "MAP@10": sum(average_precisions) / total,
        "MRR@10": sum(reciprocal_ranks) / total,
        "n": total,
    }


def paired_reliability(
    baseline: dict[str, bool], taskgraph: dict[str, bool], *, seed: int = 20260801, samples: int = 10000,
) -> dict[str, Any]:
    ids = sorted(set(baseline) & set(taskgraph))
    baseline_only = sum(baseline[item] and not taskgraph[item] for item in ids)
    taskgraph_only = sum(taskgraph[item] and not baseline[item] for item in ids)
    discordant = baseline_only + taskgraph_only
    p_value = binomtest(min(baseline_only, taskgraph_only), discordant, 0.5).pvalue if discordant else 1.0
    deltas = np.asarray([float(taskgraph[item]) - float(baseline[item]) for item in ids])
    negative = int(np.sum(deltas < 0))
    zero = int(np.sum(deltas == 0))
    positive = int(np.sum(deltas > 0))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), [negative / len(ids), zero / len(ids), positive / len(ids)], size=samples)
    boots = np.sort((counts[:, 2] - counts[:, 0]) / len(ids))
    return {
        "n": len(ids),
        "accuracy_delta": float(np.mean(deltas)),
        "bootstrap_ci95": [float(boots[int(samples * 0.025)]), float(boots[int(samples * 0.975)])],
        "mcnemar_exact_p": p_value,
        "discordant": {"baseline_only": baseline_only, "taskgraph_only": taskgraph_only},
        "bootstrap_samples": samples,
        "seed": seed,
    }


def _compact(text: str) -> str:
    return str(text).replace(" ", "").replace("\n", "")
