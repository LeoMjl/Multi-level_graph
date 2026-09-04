from __future__ import annotations

import json
from pathlib import Path

from mlg.m5.backend import TextBackend
from mlg.m5.dataset import M5Dataset
from mlg.m5.memory import CurrentOnlyMemory, MemoryStrategy, RecencyWindowMemory, RunningSummaryMemory
from mlg.m5.memory_extra import HierarchicalSummaryMemory, OracleRetrievalMemory
from mlg.m5.rag_memory import (
    FakeEmbeddingBackend,
    FlatVectorRAGMemory,
    OpenAIEmbeddingBackend,
)
from mlg.m5.taskgraph_memory import TaskGraphNovelMemory
from mlg.m5.taskgraph_config import PaperDependencyConfig


CONDITIONS = (
    "current_only",
    "recency_window",
    "running_summary",
    "flat_vector_rag",
    "hierarchical_summary",
    "oracle_retrieval",
    "taskgraph",
)


def make_memory(
    condition: str,
    m5_root: Path,
    *,
    token_budget: int,
    fake: bool,
    relationship_backend: TextBackend | None = None,
    dependency_config: PaperDependencyConfig | None = None,
) -> MemoryStrategy:
    if condition == "current_only":
        return CurrentOnlyMemory(token_budget=token_budget)
    if condition == "recency_window":
        return RecencyWindowMemory(token_budget=token_budget)
    if condition == "running_summary":
        return RunningSummaryMemory(token_budget=token_budget)
    if condition == "flat_vector_rag":
        embeddings = FakeEmbeddingBackend() if fake else OpenAIEmbeddingBackend()
        return FlatVectorRAGMemory(embeddings, token_budget=token_budget)
    if condition == "hierarchical_summary":
        return HierarchicalSummaryMemory(token_budget=token_budget)
    if condition == "oracle_retrieval":
        return OracleRetrievalMemory(m5_root / "hook_schedule.jsonl", token_budget=token_budget)
    if condition == "taskgraph":
        if relationship_backend is None:
            raise ValueError("TaskGraph requires a relationship judgment backend")
        embeddings = FakeEmbeddingBackend() if fake else OpenAIEmbeddingBackend()
        shape = json.loads((m5_root / "global_task.json").read_text(encoding="utf-8-sig"))
        dataset = M5Dataset(m5_root)
        return TaskGraphNovelMemory(
            embeddings,
            relationship_backend,
            token_budget=token_budget,
            total_volumes=int(shape["total_volumes"]),
            chapters_per_volume=int(shape["chapters_per_volume"]),
            volume_briefs=dataset.volume_briefs(),
            dependency_config=dependency_config,
        )
    raise ValueError(f"Unknown M5 condition: {condition}; choose from {', '.join(CONDITIONS)}")
