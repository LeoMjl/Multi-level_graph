from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from mlg.m5.dataset import ChapterPrompt
from mlg.m5.memory import count_tokens


WRITEBACK_REGISTRY_TOKEN_BUDGET = 4000
WRITEBACK_REGISTRY_ITEM_LIMIT = 96
MAX_EXTRACTED_L4_FACTS = 12
LIFECYCLE_KEY_SUGGESTION_MIN_RATIO = 0.75
LIFECYCLE_KEY_SUGGESTION_LIMIT = 3


@dataclass(frozen=True)
class RegistrySelection:
    rows: tuple[dict[str, Any], ...]
    total_count: int
    prompt_tokens: int
    dropped_count: int
    token_budget: int
    item_limit: int

    def audit(self) -> dict[str, int]:
        return {
            "total_count": self.total_count,
            "selected_count": len(self.rows),
            "dropped_count": self.dropped_count,
            "prompt_tokens": self.prompt_tokens,
            "token_budget": self.token_budget,
            "item_limit": self.item_limit,
        }


def memory_system_prompt(*, registry_provided: bool = False) -> str:
    registry_rule = (
        "输入包含STATE_KEY_REGISTRY（active/resolved最新版本的有界相关视图，可能省略无关旧槽）。"
        "更新既有状态槽时operation必须为update，key与prior_key都必须原样复用登记key；"
        "只有正文确实出现全新状态槽时才能用operation=create创建新key，且prior_key必须为null；"
        "不得仅因某个旧槽未出现在有界视图中就另造同义key。"
        if registry_provided else ""
    )
    return (
        "你是长篇小说TaskGraph原子状态写回器。只根据刚完成正文抽取可被后文依赖的事实，"
        "不得预测未来。输出严格JSON：summary字符串；facts必须包含1至12条原子事实。"
        "每个fact必须且只能明确给出"
        "operation(create/update)、跨章节稳定key、prior_key、value、entities数组、importance(1-5)、"
        "final_status(active/resolved)、source_quote。绝不能输出superseded；旧版本由系统自动标记。"
        "相同状态槽在不同章节必须复用完全相同的key。重点保存具体数字、形状、方向、损耗、持有人、"
        "知识来源、承诺和未决线索；每项只表达一个可核验原子事实。final_status=active只用于正文结束时"
        "仍未解决、仍在持有或仍会约束后文的状态；一次性完成的事件和已经闭合的问题用resolved；"
        "若可抽取事实超过12条，只保留对后文影响最大的12条，优先保留未决约束、精确属性、"
        "人物持有/知识状态和既有stable key更新；"
        "source_quote必须是支持该事实的非空简短证据，可逐字引用，也可忠实释义；"
        "只能陈述本章正文实际发生或确认的内容，不得推断未来或编造。"
        f"{registry_rule}"
    )


def memory_user_prompt(
    prompt: ChapterPrompt,
    text: str,
    active_registry: list[dict[str, Any]] | None = None,
) -> str:
    registry = json.dumps(active_registry or [], ensure_ascii=False)
    return (
        f"STATE_KEY_REGISTRY（有界相关视图；只能用于key复用，事实仍须由本章正文支持）：\n{registry}\n\n"
        f"第{prompt.chapter_id}章《{prompt.title}》正文：\n{text}"
    )


def active_state_registry(graph: Any, fact_by_key: dict[str, str]) -> list[dict[str, Any]]:
    """Expose each stable key's latest active or resolved state, never old versions."""
    rows: list[dict[str, Any]] = []
    for key, node_id in sorted(fact_by_key.items()):
        node = graph.nodes.get(node_id)
        if (
            node is None
            or key.startswith("chapter_summary:")
            or node.metadata.get("state_status") not in {"active", "resolved"}
        ):
            continue
        rows.append({
            "key": key,
            "value": node.value,
            "entities": list(node.metadata.get("entities", [])),
            "chapter_id": int(node.metadata.get("chapter_id", 0)),
            "final_status": str(node.metadata.get("state_status", "active")),
            "version": int(node.metadata.get("state_version", 1)),
        })
    return rows


def select_writeback_registry(
    graph: Any,
    fact_by_key: dict[str, str],
    source_text: str,
    *,
    token_budget: int = WRITEBACK_REGISTRY_TOKEN_BUDGET,
    item_limit: int = WRITEBACK_REGISTRY_ITEM_LIMIT,
) -> RegistrySelection:
    """Rank the full controller registry, then expose only a bounded detailed view."""
    if token_budget < 0 or item_limit < 0:
        raise ValueError("Writeback registry limits must be non-negative")
    full = active_state_registry(graph, fact_by_key)
    source_terms = _bigrams(source_text)
    ranked: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in full:
        node = graph.nodes[fact_by_key[str(row["key"])]]
        entities = [str(item) for item in row.get("entities", [])]
        entity_hit = any(entity and entity in source_text for entity in entities)
        row_terms = _bigrams(" ".join([str(row.get("value", "")), *entities]))
        overlap = len(source_terms & row_terms) / max(1, len(row_terms))
        rank = (
            -int(entity_hit),
            -overlap,
            -int(row.get("final_status") == "active"),
            -int(node.metadata.get("importance", 1)),
            -int(row.get("chapter_id", 0)),
            str(row.get("key", "")),
        )
        ranked.append((rank, row))
    selected: list[dict[str, Any]] = []
    for _, row in sorted(ranked, key=lambda item: item[0]):
        if len(selected) >= item_limit:
            break
        trial = [*selected, row]
        if count_tokens(json.dumps(trial, ensure_ascii=False)) <= token_budget:
            selected.append(row)
    tokens = count_tokens(json.dumps(selected, ensure_ascii=False))
    return RegistrySelection(
        rows=tuple(selected),
        total_count=len(full),
        prompt_tokens=tokens,
        dropped_count=len(full) - len(selected),
        token_budget=token_budget,
        item_limit=item_limit,
    )


def _bigrams(text: str) -> set[str]:
    compact = "".join(char.casefold() for char in text if not char.isspace())
    return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}


def summary_fact(prompt: ChapterPrompt, payload: dict[str, Any]) -> dict[str, Any]:
    summary = str(payload.get("summary") or "本章已完成，细节以正文为准。").strip()
    return {
        "operation": "create",
        "key": f"chapter_summary:{prompt.chapter_id}",
        "prior_key": None,
        "value": summary,
        "entities": [],
        "importance": 3,
        "final_status": "resolved",
        "source_quote": "",
    }


def atomic_facts(
    prompt: ChapterPrompt,
    payload: dict[str, Any],
    *,
    source_text: str | None = None,
    registered_keys: set[str] | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Parse once and validate the extractor's stable-key lifecycle contract."""
    if strict and set(payload) != {"summary", "facts"}:
        missing = sorted({"summary", "facts"} - set(payload))
        extra = sorted(set(payload) - {"summary", "facts"})
        detail = ", ".join([*(f"missing:{item}" for item in missing),
                            *(f"forbidden:{item}" for item in extra)])
        raise ValueError(f"Writeback top-level schema mismatch: {detail}")
    if strict and (
        not isinstance(payload.get("summary"), str)
        or not payload["summary"].strip()
    ):
        raise ValueError("Writeback summary must be a non-empty string")
    if strict and source_text is None:
        raise ValueError("Strict writeback validation requires chapter source text")
    facts: list[dict[str, Any]] = []
    raw_facts = payload.get("facts", [])
    if not isinstance(raw_facts, list):
        raise ValueError("Writeback facts must be an array")
    if not raw_facts:
        raise ValueError("Writeback facts must be non-empty")
    if len(raw_facts) > MAX_EXTRACTED_L4_FACTS:
        raise ValueError(
            "Writeback facts must contain at most "
            f"{MAX_EXTRACTED_L4_FACTS} items; received {len(raw_facts)}"
        )
    known = set(registered_keys or ())
    seen_keys: set[str] = set()
    quote_errors: list[str] = []
    for index, raw in enumerate(raw_facts):
        if strict and not isinstance(raw, dict):
            raise ValueError(f"Writeback fact {index} must be an object")
        fact = raw if isinstance(raw, dict) else {"value": str(raw)}
        required = {
            "operation", "key", "prior_key", "value", "entities",
            "importance", "final_status", "source_quote",
        }
        if strict:
            missing = sorted(required - set(fact))
            if missing:
                raise ValueError(
                    f"Writeback fact {index} missing fields: {', '.join(missing)}"
                )
            extra = sorted(set(fact) - required)
            if extra:
                raise ValueError(
                    f"Writeback fact {index} has forbidden fields: {', '.join(extra)}"
                )
            for field in ("operation", "key", "value", "final_status", "source_quote"):
                if not isinstance(fact[field], str):
                    raise ValueError(
                        f"Writeback fact {index} field {field} must be a string"
                    )
            if fact["prior_key"] is not None and not isinstance(fact["prior_key"], str):
                raise ValueError(f"Writeback fact {index} prior_key must be string or null")
            if not isinstance(fact["entities"], list) or any(
                not isinstance(entity, str) for entity in fact["entities"]
            ):
                raise ValueError(f"Writeback entities must be a string array at {index}")
        value = str(fact.get("value") or fact.get("fact") or "").strip()
        if not value:
            if strict:
                raise ValueError(f"Writeback fact {index} has empty value")
            continue
        key = str(
            fact.get("key") or f"chapter_{prompt.chapter_id}_fact_{index}"
        ).strip()
        if not key:
            raise ValueError(f"Writeback fact {index} has empty key")
        if key.startswith("chapter_summary:"):
            raise ValueError(f"Reserved state key prefix is not allowed: {key}")
        if key in seen_keys:
            raise ValueError(f"Duplicate state key in writeback: {key}")
        seen_keys.add(key)
        operation_supplied = bool(str(fact.get("operation") or "").strip())
        operation = str(fact.get("operation") or (
            "update" if key in known else "create"
        )).strip().lower()
        if operation not in {"create", "update"}:
            raise ValueError(f"Invalid lifecycle operation for {key}: {operation}")
        prior_raw = fact.get("prior_key")
        prior_key = str(prior_raw).strip() if prior_raw is not None else None
        if strict and operation == "create" and prior_raw is not None:
            raise ValueError(f"create prior_key must be null: {key}")
        if not strict and not operation_supplied and operation == "update":
            prior_key = key
        _validate_lifecycle(operation, key, prior_key, known)
        status = str(
            fact.get("final_status", fact.get("status", "active"))
        ).strip().lower()
        if status not in {"active", "resolved"}:
            if strict:
                raise ValueError(f"Invalid final_status for {key}: {status}")
            status = "active"
        importance = _importance(fact.get("importance", 3), key, strict)
        entities = fact.get("entities", [])
        if strict and not isinstance(entities, list):
            raise ValueError(f"Writeback entities must be an array for {key}")
        if not isinstance(entities, list):
            entities = [entities]
        source_quote = str(fact.get("source_quote", "")).strip()
        quote_valid: bool | None = None
        quote_exact: bool | None = None
        if source_text is not None:
            quote_exact = bool(source_quote) and source_quote in source_text
            if strict:
                quote_valid = bool(source_quote)
                if not quote_valid:
                    quote_errors.append(
                        f"key={json.dumps(key[:160], ensure_ascii=False)} has empty source_quote"
                    )
            else:
                quote_valid = bool(source_quote) and _contains_quote(
                    source_text, source_quote,
                )
            if not quote_valid and not strict:
                source_quote = ""
        facts.append({
            "operation": operation,
            "key": key,
            "prior_key": prior_key,
            "value": value,
            "entities": sorted({str(item).strip() for item in entities if str(item).strip()}),
            "importance": importance,
            "final_status": status,
            "source_quote": source_quote,
            "source_quote_valid": quote_valid,
            "source_quote_exact": quote_exact,
        })
    if quote_errors:
        raise ValueError(
            "Missing source_quote evidence. Provide a non-empty factual statement "
            "grounded in this chapter; faithful paraphrase is allowed. "
            + " | ".join(quote_errors)
        )
    return facts


def _validate_lifecycle(
    operation: str, key: str, prior_key: str | None, known: set[str],
) -> None:
    if operation == "create":
        if key in known:
            raise ValueError(f"create cannot reuse registered key: {key}")
        if prior_key:
            raise ValueError(f"create must not provide prior_key: {key}")
        return
    if key not in known:
        suggestions = _registered_key_suggestions(key, known)
        detail = ""
        if suggestions:
            rendered = json.dumps(suggestions, ensure_ascii=False)
            detail = (
                "; nearest registered keys (diagnostic only, not auto-repair): "
                + rendered
            )
        raise ValueError(f"update requires a registered key: {key}{detail}")
    if prior_key != key:
        raise ValueError(f"update prior_key must exactly match registered key: {key}")


def _registered_key_suggestions(key: str, known: set[str]) -> list[str]:
    ranked = sorted(
        (
            (SequenceMatcher(None, key, candidate).ratio(), candidate)
            for candidate in known
        ),
        key=lambda row: (-row[0], row[1]),
    )
    return [
        candidate for score, candidate in ranked
        if score >= LIFECYCLE_KEY_SUGGESTION_MIN_RATIO
    ][:LIFECYCLE_KEY_SUGGESTION_LIMIT]


def _importance(value: Any, key: str, strict: bool) -> int:
    if strict and (not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5):
        raise ValueError(f"importance must be an integer from 1 to 5 for {key}")
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return 3


def _contains_quote(text: str, quote: str) -> bool:
    return "".join(quote.split()) in "".join(text.split())
