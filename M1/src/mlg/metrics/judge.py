"""Semantic dependency matching used by auxiliary graph diagnostics."""
from __future__ import annotations

from typing import Any

from mlg.config import RuntimeConfig


def _thinking_extra_body(runtime: RuntimeConfig) -> dict[str, Any] | None:
    if runtime.thinking_mode in {"enabled", "disabled"}:
        return {"thinking": {"type": runtime.thinking_mode}}
    return None


def _parse_json_bool_key(content: str, key: str) -> bool:
    import json

    try:
        value = json.loads(content)
        if isinstance(value, dict):
            return bool(value.get(key, False))
    except json.JSONDecodeError:
        pass
    return '"true"' in content.lower()


def semantic_dependency_recall(
    prediction_dependencies: list,
    gold_dependencies: list,
    prediction_graph: dict | None = None,
    *,
    runtime: RuntimeConfig | None = None,
) -> tuple[float, str]:
    """Judge whether the method recovered the gold dependency edges, semantically.

    The old metric required both ``dep["source"]`` and ``dep["target"]`` to appear as
    literal substrings in the prediction. That failed for ours, which emits graph node
    IDs (``L4_Item1288``) instead of conversation text -- the format simply never
    matched gold's verbatim quotes.

    This variant resolves node IDs back to their text via ``prediction_graph`` (the
    serialized TaskGraph carried in prediction.metadata['graph']), then asks the judge
    whether each gold fact is covered by some predicted edge's source text.

    Returns ``(score in [0,1], match_method)``.
    """
    if not gold_dependencies:
        return 1.0, "no_gold"
    if not prediction_dependencies:
        return 0.0, "empty"

    if runtime is None:
        runtime = RuntimeConfig()

    # Build node_id -> text lookup from the serialized graph, so L4_Item1288 etc.
    # can be resolved to the conversation fragment they were extracted from.
    node_text = _build_node_text_index(prediction_graph)

    # Resolve each predicted edge into a human-readable (source_text, target_text) pair.
    pred_pairs = _resolve_dependency_pairs(prediction_dependencies, node_text)

    if runtime.api_key:
        results = []
        for gold in gold_dependencies:
            gold_source = str(gold.get("source", "") if isinstance(gold, dict) else gold)
            gold_target = str(gold.get("target", "") if isinstance(gold, dict) else "")
            results.append(_judge_dependency_covered(gold_source, gold_target, pred_pairs, runtime))
        # Only fall back to literal if EVERY judge call failed (returned None). A legitimate
        # False (no predicted edge covers the gold fact) must count as a judge_llm decision.
        if any(r is not None for r in results):
            hits = sum(1 for r in results if r)
            return hits / len(gold_dependencies), "judge_llm"

    # Fallback: literal substring match against resolved text (fake_llm / no-key).
    return _literal_dependency_recall(pred_pairs, gold_dependencies), "judge_fallback"


def _build_node_text_index(graph: dict | None) -> dict[str, str]:
    """Map node_id -> best available text (value, else content) from a serialized graph."""
    index: dict[str, str] = {}
    if not isinstance(graph, dict):
        return index
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id", "")
        if not node_id:
            continue
        text = node.get("value") or node.get("content") or ""
        if text:
            index[node_id] = str(text)
    return index


def _resolve_dependency_pairs(deps: list, node_text: dict[str, str]) -> list[tuple[str, str]]:
    """Turn a list of dependency dicts into (source_text, target_text) pairs.

    Handles two dependency shapes:
      - LLM/baseline shape: {"source": <text or id>, "target": <text or id>}
      - ours shape: {"source_id": <node id>, "target_id": <node id>,
                     "source_content": <merged text>, "target_content": <text>}

    Node IDs are resolved via node_text when available; explicit *_content fields
    (ours' merged conversation-fact text) take priority since they carry the full-turn
    fragment that gold quotes match against.
    """
    pairs: list[tuple[str, str]] = []
    for dep in deps or []:
        if not isinstance(dep, dict):
            continue
        # Prefer explicit content fields (ours' merged turn text), then fall back to
        # source/target which may be node IDs resolvable via node_text.
        src = str(dep.get("source_content") or dep.get("source") or dep.get("source_id") or "")
        tgt = str(dep.get("target_content") or dep.get("target") or dep.get("target_id") or "")
        # If src/tgt look like node IDs, try resolving them via the graph index.
        if src in node_text:
            src = node_text[src]
        if tgt in node_text:
            tgt = node_text[tgt]
        pairs.append((src, tgt))
    return pairs


def _judge_dependency_covered(
    gold_source: str,
    gold_target: str,
    pred_pairs: list[tuple[str, str]],
    runtime: RuntimeConfig,
) -> bool | None:
    """Ask the judge whether any predicted edge covers the gold fact."""
    try:
        import httpx
        from openai import OpenAI

        http_client = httpx.Client(
            verify=runtime.verify_ssl,
            trust_env=not runtime.ignore_proxy,
            timeout=60,
        )
        client = OpenAI(api_key=runtime.api_key, base_url=runtime.base_url, http_client=http_client)
        pred_lines = "\n".join(
            f"- source: {s[:200]} | target: {t[:120]}" for s, t in pred_pairs[:8]
        ) or "(none)"
        system = (
            "You are evaluating dependency recovery for a task-memory benchmark. "
            "A gold dependency points from a conversation FACT to the stage it belongs to. "
            "Decide whether any predicted edge covers the SAME fact (same content, possibly "
            "paraphrased or referenced by node text). Answer only JSON: "
            "{\"covered\": true} or {\"covered\": false}."
        )
        user = (
            f"Gold dependency:\n  source (fact): {gold_source[:300]}\n"
            f"  target (stage): {gold_target}\n\n"
            f"Predicted edges:\n{pred_lines}\n\n"
            "Is the gold fact covered by any predicted edge? Return JSON only."
        )
        response = client.chat.completions.create(
            model=runtime.judge_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=32,
            extra_body=_thinking_extra_body(runtime),
        )
        content = (response.choices[0].message.content or "").strip()
        return _parse_json_bool_key(content, "covered")
    except Exception:
        return None


def _literal_dependency_recall(
    pred_pairs: list[tuple[str, str]],
    gold_dependencies: list,
) -> float:
    """Fallback dependency recall over resolved text (no LLM)."""
    hits = 0
    for gold in gold_dependencies:
        gold_source = str(gold.get("source", "") if isinstance(gold, dict) else gold).lower()
        gold_target = str(gold.get("target", "") if isinstance(gold, dict) else "").lower()
        for src, tgt in pred_pairs:
            s, t = src.lower(), tgt.lower()
            if gold_source and gold_source in s:
                if not gold_target or gold_target in t:
                    hits += 1
                    break
    return hits / len(gold_dependencies) if gold_dependencies else 1.0
