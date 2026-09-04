from __future__ import annotations

import json
import re
import time
from typing import Any

from mlg.config import RuntimeConfig
from mlg.methods.base import extract_api_usage


def official_rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """Return the default `rouge` package F scores used by the LoCoMo paper."""
    if not prediction.strip() or not reference.strip():
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    try:
        from rouge import Rouge
    except ImportError as exc:  # pragma: no cover - packaging error
        raise RuntimeError("LoCoMo evaluation requires the project dependency `rouge>=1.0.1`") from exc
    scores = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"]).get_scores(
        prediction,
        reference,
    )[0]
    return {
        "rouge_1": float(scores["rouge-1"]["f"]),
        "rouge_2": float(scores["rouge-2"]["f"]),
        "rouge_l": float(scores["rouge-l"]["f"]),
    }


def locomo_factscore(
    prediction: str,
    reference_events: list[dict[str, Any]],
    *,
    runtime: RuntimeConfig | None = None,
    allow_llm_judge: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Apply LoCoMo's adapted FactScore precision/recall definition."""
    references = [
        f"{str(item.get('date', '')).strip()}: {str(item.get('fact', '')).strip()}".strip(": ")
        for item in reference_events
        if isinstance(item, dict) and str(item.get("fact", "")).strip()
    ]
    predicted_facts = _atomic_prediction_facts(prediction)
    if not references or not predicted_facts:
        zeros = {
            "factscore_precision": 0.0,
            "factscore_recall": 0.0,
            "factscore_f1": 0.0,
        }
        return zeros, {
            "evaluation_mode": "empty",
            "reference_fact_count": len(references),
            "predicted_fact_count": len(predicted_facts),
        }

    runtime = runtime or RuntimeConfig()
    judged = None
    if allow_llm_judge and runtime.api_key:
        judged = _judge_fact_alignment(predicted_facts, references, runtime)
        if judged is None:
            raise RuntimeError("LoCoMo FactScore judge failed; formal evaluation cannot use lexical fallback")
    if judged is not None:
        judged_facts, supported, covered, trace = judged
        predicted_facts = judged_facts or predicted_facts
        precision = len(supported) / len(predicted_facts) if predicted_facts else 0.0
        recall = len(covered) / len(references)
        mode = "locomo_adapted_factscore_llm"
    else:
        supported, covered = _lexical_fact_alignment(predicted_facts, references)
        precision = len(supported) / len(predicted_facts)
        recall = len(covered) / len(references)
        trace = {}
        mode = "locomo_adapted_factscore_lexical_fallback"
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "factscore_precision": precision,
        "factscore_recall": recall,
        "factscore_f1": f1,
    }, {
        "evaluation_mode": mode,
        "reference_fact_count": len(references),
        "predicted_fact_count": len(predicted_facts),
        "supported_predicted_count": len(supported),
        "covered_reference_count": len(covered),
        **trace,
    }


def _judge_fact_alignment(
    predicted_facts: list[str],
    references: list[str],
    runtime: RuntimeConfig,
) -> tuple[list[str], set[int], set[int], dict[str, Any]] | None:
    try:
        import httpx
        from openai import OpenAI

        client = OpenAI(
            api_key=runtime.api_key,
            base_url=runtime.base_url,
            http_client=httpx.Client(
                verify=runtime.verify_ssl,
                trust_env=not runtime.ignore_proxy,
                timeout=runtime.request_timeout_seconds,
            ),
        )
        system = (
            "You are implementing LoCoMo's adapted FactScore. The candidate summary and reference "
            "event graph have already been split into event facts. A predicted fact is supported "
            "only if a reference fact entails it. Treat dates, "
            "entities, negation, and causal claims strictly. Represent every valid correspondence "
            "as one predicted-fact/reference-fact pair so precision and recall use the same matches."
        )
        user = (
            "Reference event graph:\n"
            + "\n".join(f"[{index}] {fact}" for index, fact in enumerate(references))
            + "\n\nCandidate event facts:\n"
            + "\n".join(f"[{index}] {fact}" for index, fact in enumerate(predicted_facts))
            + '\n\nReturn JSON only: {"matched_pairs":'
            '[{"predicted_index":0,"reference_index":0}]}.'
        )
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=runtime.judge_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max(2_048, runtime.max_tokens),
            extra_body=(
                {"thinking": {"type": runtime.thinking_mode}}
                if runtime.thinking_mode in {"enabled", "disabled"}
                else None
            ),
        )
        raw = json.loads(response.choices[0].message.content or "{}")
        if "matched_pairs" not in raw:
            return None
        matched_pairs = _valid_fact_pairs(
            raw.get("matched_pairs"),
            len(predicted_facts),
            len(references),
        )
        supported = {predicted_index for predicted_index, _ in matched_pairs}
        covered = {reference_index for _, reference_index in matched_pairs}
        return predicted_facts, supported, covered, {
            "model": runtime.judge_model,
            "latency_ms": (time.perf_counter() - started) * 1_000,
            "usage": extract_api_usage(getattr(response, "usage", None)),
            "usage_missing": getattr(response, "usage", None) is None,
            "matched_pairs": [
                {
                    "predicted_index": predicted_index,
                    "reference_index": reference_index,
                }
                for predicted_index, reference_index in sorted(matched_pairs)
            ],
        }
    except Exception:
        return None


def _valid_fact_pairs(
    value: Any,
    predicted_count: int,
    reference_count: int,
) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        return set()
    pairs: set[tuple[int, int]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        predicted_index = item.get("predicted_index")
        reference_index = item.get("reference_index")
        if (
            isinstance(predicted_index, int)
            and not isinstance(predicted_index, bool)
            and 0 <= predicted_index < predicted_count
            and isinstance(reference_index, int)
            and not isinstance(reference_index, bool)
            and 0 <= reference_index < reference_count
        ):
            pairs.add((predicted_index, reference_index))
    return pairs


def _atomic_prediction_facts(text: str) -> list[str]:
    lines = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    if len(lines) <= 1:
        lines = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    return list(dict.fromkeys(line for line in lines if len(line.split()) >= 3))


def _lexical_fact_alignment(
    predicted: list[str],
    references: list[str],
) -> tuple[set[int], set[int]]:
    predicted_tokens = [_content_tokens(item) for item in predicted]
    reference_tokens = [_content_tokens(item) for item in references]
    supported: set[int] = set()
    covered: set[int] = set()
    for pred_index, pred in enumerate(predicted_tokens):
        for ref_index, ref in enumerate(reference_tokens):
            if _token_f1(pred, ref) >= 0.5:
                supported.add(pred_index)
                covered.add(ref_index)
    return supported, covered


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "with", "from", "that", "this"}
    }


def _token_f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return 0.0 if overlap == 0 else 2 * overlap / (len(left) + len(right))
