from __future__ import annotations

from typing import Any

from mlg.m5.dataset import ChapterPrompt
from mlg.m5.memory import MemoryContext, MemoryStrategy, count_tokens, truncate_tokens


CHAPTER_SUMMARY_TOKENS = 600
VOLUME_SUMMARY_TOKENS = 800
RECENT_CHAPTER_SUMMARIES = 8


class HierarchicalSummaryMemory(MemoryStrategy):
    name = "hierarchical_summary"

    def __init__(self, *, token_budget: int = 12000) -> None:
        super().__init__(token_budget=token_budget)
        self.chapter_summaries: dict[int, str] = {}
        self.volume_summaries: dict[int, str] = {}
        self.volume_chapters: dict[int, list[int]] = {}

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        candidates: list[tuple[str, int, str, list[int]]] = []
        for volume_id in sorted(self.volume_summaries):
            candidates.append((
                "volume", volume_id,
                f"[第{volume_id}卷累计摘要]\n{self.volume_summaries[volume_id]}",
                list(self.volume_chapters.get(volume_id, [])),
            ))
        recent = sorted(self.chapter_summaries)[-RECENT_CHAPTER_SUMMARIES:]
        for chapter_id in recent:
            candidates.append((
                "chapter", chapter_id,
                f"[第{chapter_id}章摘要]\n{self.chapter_summaries[chapter_id]}",
                [chapter_id],
            ))
        blocks: list[str] = []
        included: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        selected: set[int] = set()
        used = 0
        for level, identifier, block, chapters in candidates:
            tokens = count_tokens(block)
            row = {"level": level, "id": identifier, "tokens": tokens}
            if used + tokens <= self.token_budget:
                blocks.append(block)
                included.append(row)
                selected.update(chapters)
                used += tokens
            else:
                dropped.append(row)
        return MemoryContext("\n\n".join(blocks), sorted(selected), {
            **self.protocol_config(),
            "memory_tokens": used,
            "included_blocks": included,
            "dropped_blocks": dropped,
            "recent_summary_ids": recent,
            "volume_coverage": {
                str(key): list(value) for key, value in self.volume_chapters.items()
            },
        })

    def apply_update(
        self, prompt: ChapterPrompt, *, chapter_summary: str,
        volume_summary: str,
    ) -> None:
        chapter_summary = chapter_summary.strip()
        volume_summary = volume_summary.strip()
        if not chapter_summary or not volume_summary:
            raise ValueError("Hierarchical summary fields must both be non-empty")
        self.chapter_summaries[prompt.chapter_id] = truncate_tokens(
            chapter_summary, CHAPTER_SUMMARY_TOKENS, keep_end=False,
        )
        self.volume_summaries[prompt.volume_id] = truncate_tokens(
            volume_summary, VOLUME_SUMMARY_TOKENS, keep_end=False,
        )
        chapters = self.volume_chapters.setdefault(prompt.volume_id, [])
        if prompt.chapter_id not in chapters:
            chapters.append(prompt.chapter_id)
            chapters.sort()

    def protocol_config(self) -> dict[str, Any]:
        return {
            "chapter_summary_tokens": CHAPTER_SUMMARY_TOKENS,
            "volume_summary_tokens": VOLUME_SUMMARY_TOKENS,
            "recent_chapter_summaries": RECENT_CHAPTER_SUMMARIES,
            "packing": "all_volume_summaries_then_recent_chapter_summaries",
        }

    def to_state(self) -> dict[str, Any]:
        return {
            **super().to_state(),
            "config": self.protocol_config(),
            "chapter_summaries": self.chapter_summaries,
            "volume_summaries": self.volume_summaries,
            "volume_chapters": self.volume_chapters,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        if state.get("config") != self.protocol_config():
            raise ValueError("Hierarchical-summary configuration changed during resume")
        self.chapter_summaries = {
            int(key): str(value)
            for key, value in state.get("chapter_summaries", {}).items()
        }
        self.volume_summaries = {
            int(key): str(value)
            for key, value in state.get("volume_summaries", {}).items()
        }
        self.volume_chapters = {
            int(key): [int(item) for item in value]
            for key, value in state.get("volume_chapters", {}).items()
        }
