from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlg.m5.backend import GenerationResult, TextBackend
from mlg.m5.dataset import ChapterPrompt
from mlg.m5.memory import MemoryContext, MemoryStrategy, parse_json_object, truncate_tokens


class HierarchicalSummaryMemory(MemoryStrategy):
    name = "hierarchical_summary"

    def __init__(self, *, token_budget: int = 12000) -> None:
        super().__init__(token_budget=token_budget)
        self.chapter_summaries: dict[int, str] = {}
        self.volume_summaries: dict[int, str] = {}

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        blocks: list[str] = []
        for volume_id in sorted(self.volume_summaries):
            blocks.append(f"[第{volume_id}卷累计摘要]\n{self.volume_summaries[volume_id]}")
        recent = sorted(self.chapter_summaries)[-8:]
        for chapter_id in recent:
            blocks.append(f"[第{chapter_id}章摘要]\n{self.chapter_summaries[chapter_id]}")
        text = truncate_tokens("\n\n".join(blocks), self.token_budget, keep_end=True)
        covered = sorted(self.chapter_summaries)
        return MemoryContext(text, covered, {"recent_summary_ids": recent})

    def observe(self, prompt: ChapterPrompt, text: str, backend: TextBackend) -> list[GenerationResult]:
        prior_volume = self.volume_summaries.get(prompt.volume_id, "")
        system = (
            "你是分层小说记忆器。根据刚完成正文，输出严格JSON："
            "chapter_summary为本章事实摘要；volume_summary为合并既有卷摘要后的本卷累计摘要。"
            "必须保留人物知识边界、物品状态、时间地点、因果与未决线索，不得预测未来。"
        )
        user = (
            f"卷号：{prompt.volume_id}\n既有本卷摘要：\n{prior_volume or '无'}"
            f"\n\n第{prompt.chapter_id}章正文：\n{text}"
        )
        result = backend.generate(system, user, purpose="memory")
        payload = parse_json_object(result.text)
        chapter_summary = str(payload.get("chapter_summary") or payload.get("summary") or result.text)
        volume_summary = str(payload.get("volume_summary") or chapter_summary)
        self.chapter_summaries[prompt.chapter_id] = truncate_tokens(chapter_summary, 900, keep_end=False)
        self.volume_summaries[prompt.volume_id] = truncate_tokens(volume_summary, 2600, keep_end=False)
        return [result]

    def to_state(self) -> dict[str, Any]:
        return {
            **super().to_state(),
            "chapter_summaries": self.chapter_summaries,
            "volume_summaries": self.volume_summaries,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self.chapter_summaries = {int(k): str(v) for k, v in state.get("chapter_summaries", {}).items()}
        self.volume_summaries = {int(k): str(v) for k, v in state.get("volume_summaries", {}).items()}


class OracleRetrievalMemory(MemoryStrategy):
    """Evaluation-only upper bound: retrieve plant text, never hidden gold predicates."""

    name = "oracle_retrieval"

    def __init__(self, schedule_path: Path, *, token_budget: int = 12000) -> None:
        super().__init__(token_budget=token_budget)
        self.schedule_path = schedule_path.resolve()
        self.schedule = [
            json.loads(line) for line in self.schedule_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        self.chapters: dict[int, str] = {}

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        selected: list[int] = []
        for row in self.schedule:
            start, end = [int(item) for item in row["evaluation_window"]]
            if start <= prompt.chapter_id <= end:
                plant = int(row["plant_chapter"])
                if plant in self.chapters:
                    selected.append(plant)
        selected = sorted(set(selected))
        blocks = [f"[历史第{chapter_id}章]\n{self.chapters[chapter_id]}" for chapter_id in selected]
        text = truncate_tokens("\n\n".join(blocks), self.token_budget, keep_end=False)
        return MemoryContext(text, selected, {"oracle": True, "schedule_only": True})

    def observe(self, prompt: ChapterPrompt, text: str, backend: TextBackend) -> list[GenerationResult]:
        self.chapters[prompt.chapter_id] = text
        return []

    def to_state(self) -> dict[str, Any]:
        return {**super().to_state(), "chapters": self.chapters}

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self.chapters = {int(k): str(v) for k, v in state.get("chapters", {}).items()}
