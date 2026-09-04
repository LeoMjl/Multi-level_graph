from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from mlg.m5.backend import GenerationResult, TextBackend
from mlg.m5.dataset import ChapterPrompt


@dataclass
class MemoryContext:
    text: str = ""
    selected_chapters: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStrategy:
    name = "base"

    def __init__(self, *, token_budget: int = 12000) -> None:
        self.token_budget = token_budget

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        return MemoryContext()

    def observe(
        self, prompt: ChapterPrompt, text: str, backend: TextBackend,
    ) -> list[GenerationResult]:
        return []

    def to_state(self) -> dict[str, Any]:
        return {"name": self.name, "token_budget": self.token_budget}

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("name") != self.name:
            raise ValueError(f"Cannot restore {state.get('name')} into {self.name}")


class CurrentOnlyMemory(MemoryStrategy):
    name = "current_only"


class RecencyWindowMemory(MemoryStrategy):
    name = "recency_window"

    def __init__(self, *, token_budget: int = 12000) -> None:
        super().__init__(token_budget=token_budget)
        self.chapters: list[tuple[int, str]] = []

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        blocks: list[str] = []
        selected: list[int] = []
        remaining = self.token_budget
        for chapter_id, text in reversed(self.chapters):
            if chapter_id == prompt.chapter_id - 1:
                continue
            block = f"既有事实片段：\n{text}"
            tokens = count_tokens(block)
            if tokens <= remaining:
                blocks.append(block)
                selected.append(chapter_id)
                remaining -= tokens
                continue
            if not blocks and remaining > 200:
                blocks.append(truncate_tokens(block, remaining, keep_end=True))
                selected.append(chapter_id)
                remaining = 0
            break
        blocks.reverse()
        selected.reverse()
        return MemoryContext("\n\n".join(blocks), selected, {
            "remaining_tokens": remaining,
            "excluded_local_chapter": prompt.chapter_id - 1 if prompt.chapter_id > 1 else None,
        })

    def observe(self, prompt: ChapterPrompt, text: str, backend: TextBackend) -> list[GenerationResult]:
        self.chapters.append((prompt.chapter_id, text))
        return []

    def to_state(self) -> dict[str, Any]:
        return {**super().to_state(), "chapters": self.chapters}

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self.chapters = [(int(item[0]), str(item[1])) for item in state.get("chapters", [])]


class RunningSummaryMemory(MemoryStrategy):
    name = "running_summary"

    def __init__(self, *, token_budget: int = 12000) -> None:
        super().__init__(token_budget=token_budget)
        self.summary = ""
        self.covered_through = 0
        self.prior_summary = ""
        self.prior_covered_through = 0

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        text = truncate_tokens(self.prior_summary, self.token_budget, keep_end=True)
        selected = list(range(1, self.prior_covered_through + 1)) if text else []
        return MemoryContext(text, selected, {
            "covered_through": self.prior_covered_through,
            "excluded_local_chapter": prompt.chapter_id - 1 if prompt.chapter_id > 1 else None,
        })

    def observe(self, prompt: ChapterPrompt, text: str, backend: TextBackend) -> list[GenerationResult]:
        system = (
            "你是长篇小说连续性记忆器。只依据既有摘要和刚完成正文更新摘要。"
            "保留人物知识边界、时间地点、物品位置/数量/损耗、承诺、未决线索和因果；"
            "不得预测未来。只输出JSON，字段summary。"
        )
        user = f"既有摘要：\n{self.summary or '无'}\n\n第{prompt.chapter_id}章正文：\n{text}"
        result = backend.generate(system, user, purpose="memory")
        payload = parse_json_object(result.text)
        self.prior_summary = self.summary
        self.prior_covered_through = self.covered_through
        self.summary = str(payload.get("summary") or result.text).strip()
        self.summary = truncate_tokens(self.summary, self.token_budget, keep_end=True)
        self.covered_through = prompt.chapter_id
        return [result]

    def to_state(self) -> dict[str, Any]:
        return {
            **super().to_state(),
            "summary": self.summary,
            "covered_through": self.covered_through,
            "prior_summary": self.prior_summary,
            "prior_covered_through": self.prior_covered_through,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self.summary = str(state.get("summary", ""))
        self.covered_through = int(state.get("covered_through", 0))
        self.prior_summary = str(state.get("prior_summary", ""))
        self.prior_covered_through = int(state.get("prior_covered_through", 0))


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def truncate_tokens(text: str, budget: int, *, keep_end: bool) -> str:
    if budget <= 0:
        return ""
    tokens = _encoder().encode(text)
    if len(tokens) <= budget:
        return text
    selected = tokens[-budget:] if keep_end else tokens[:budget]
    return _encoder().decode(selected)


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if 0 <= start < end:
            try:
                payload = json.loads(value[start:end + 1])
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def _encoder():
    return tiktoken.get_encoding("o200k_base")
