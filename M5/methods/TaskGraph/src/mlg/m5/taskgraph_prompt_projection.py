from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mlg.graph import Node, NodeLevel, NodeStatus
from mlg.m5.continuity import LocalContinuity
from mlg.m5.dataset import ChapterPrompt, M5Dataset
from mlg.m5.length_policy import WRITER_REQUESTED_HAN_CHARS
from mlg.m5.memory import MemoryContext, count_tokens
from mlg.m5.prompts import build_writer_prompts


class PromptProjectionError(ValueError):
    """A graph node cannot safely be exposed to the prose writer."""


class SupersededNodeError(PromptProjectionError):
    """A stale state must never be serialized into the writer context."""


@dataclass(frozen=True)
class ProjectedMemoryBlock:
    """Writer-safe prose plus controller-only provenance."""

    text: str
    source_node_ids: tuple[str, ...]
    chapter_id: int
    category: str
    information_score: int


@dataclass(frozen=True)
class PackedMemory:
    text: str
    blocks: tuple[ProjectedMemoryBlock, ...]
    memory_tokens: int
    dropped_count: int

    @property
    def selected_node_ids(self) -> tuple[str, ...]:
        return tuple(
            node_id for block in self.blocks for node_id in block.source_node_ids
        )


_FORBIDDEN_WRITER_MARKERS = re.compile(
    r"TaskGraph|伏笔关联候选|dependency[_ ]?type|stable[_ ]?key|"
    r"node[_ ]?id|L[1-4](?:节点)?",
    re.IGNORECASE,
)
_COMMAND_LINE = re.compile(r"(?m)^\s*(?:必须包含|必须避免|撰写第\d+章)[：:].*$")
_NORMALIZE = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)


def project_node(
    node: Node,
    volume_briefs: Mapping[int | str, str],
) -> ProjectedMemoryBlock:
    """Project an internal L2/L3/L4 node into writer-safe natural language."""
    chapter_id = int(node.metadata.get("chapter_id", 0))
    if node.level == NodeLevel.L2:
        if node.status not in {NodeStatus.ACTIVE, NodeStatus.DONE}:
            raise PromptProjectionError("Unreleased volume cannot enter a writer prompt")
        volume_id = int(node.metadata.get("volume_id", 0))
        brief = volume_briefs.get(volume_id) or volume_briefs.get(str(volume_id))
        if not brief or not str(brief).strip():
            raise PromptProjectionError(
                f"Missing substantive brief for volume {volume_id}"
            )
        text = f"当前卷的阶段目标：{str(brief).strip()}"
        category = "volume_brief"
    elif node.level == NodeLevel.L3:
        if node.status != NodeStatus.DONE:
            raise PromptProjectionError("Only completed historical chapters may be projected")
        summary = str(node.metadata.get("result_summary", "")).strip()
        summary = _COMMAND_LINE.sub("", summary).strip()
        if not summary:
            raise PromptProjectionError(
                "Completed chapter node has no result_summary; old instructions are unsafe"
            )
        text = f"已成立情节事实：{summary}"
        category = "chapter_result"
    elif node.level == NodeLevel.L4:
        state = str(node.metadata.get("state_status", "active")).strip().lower()
        if state == "superseded":
            raise SupersededNodeError("Superseded state cannot enter a writer prompt")
        value = str(node.value).strip()
        if not value:
            raise PromptProjectionError("State node has no observable value")
        is_summary = str(node.metadata.get("state_key", "")).startswith(
            "chapter_summary:"
        )
        if is_summary:
            text = f"已成立情节事实：{value}"
        elif state == "resolved":
            text = f"已闭合事实：{value}"
        elif state == "active":
            text = f"当前有效事实：{value}"
        else:
            raise PromptProjectionError(f"Unknown state_status: {state}")
        category = "chapter_summary" if is_summary else "atomic_state"
    else:
        raise PromptProjectionError("Only volume, completed-chapter, and state nodes project")

    text = _validate_writer_text(text, node)
    return ProjectedMemoryBlock(
        text=text,
        source_node_ids=(node.node_id,),
        chapter_id=chapter_id,
        category=category,
        information_score=_information_score(text, category),
    )


def project_nodes(
    nodes: Sequence[Node],
    volume_briefs: Mapping[int | str, str],
) -> list[ProjectedMemoryBlock]:
    """Project nodes in controller priority order, rejecting stale states."""
    return [project_node(node, volume_briefs) for node in nodes]


def deduplicate_blocks(
    blocks: Sequence[ProjectedMemoryBlock],
) -> list[ProjectedMemoryBlock]:
    """Collapse exact, near-identical, and same-chapter summary/result overlap."""
    kept: list[ProjectedMemoryBlock] = []
    for candidate in blocks:
        duplicate_index = next(
            (
                index
                for index, prior in enumerate(kept)
                if _hierarchical_duplicate(prior, candidate)
                or _textually_duplicate(prior.text, candidate.text)
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(candidate)
            continue
        prior = kept[duplicate_index]
        winner = max(
            (prior, candidate),
            key=lambda block: (block.information_score, len(block.text)),
        )
        kept[duplicate_index] = winner
    return kept


def pack_projected_memory(
    blocks: Sequence[ProjectedMemoryBlock],
    token_budget: int,
) -> PackedMemory:
    """Pack whole blocks in priority order without ever exceeding the budget."""
    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    selected: list[ProjectedMemoryBlock] = []
    for index, block in enumerate(blocks):
        trial = "\n\n".join(item.text for item in (*selected, block))
        if count_tokens(trial) > token_budget:
            return _packed(selected, len(blocks) - index)
        selected.append(block)
    return _packed(selected, 0)


def build_projected_writer_prompts(
    dataset: M5Dataset,
    chapter: ChapterPrompt,
    memory: PackedMemory,
    *,
    local_continuity: LocalContinuity | None = None,
    repeat_current_contract: bool = True,
    retry_feedback: str = "",
) -> tuple[str, str]:
    """Build prompts from projected prose, never from graph/audit representations."""
    system, user = build_writer_prompts(
        dataset,
        chapter,
        MemoryContext(text=memory.text),
        local_continuity=local_continuity,
        repeat_current_contract=repeat_current_contract,
        retry_feedback=retry_feedback,
    )
    _assert_clean_writer_output(system, user)
    return system, user


def sanitized_writer_payload(
    chapter: ChapterPrompt,
    system: str,
    user: str,
) -> dict[str, Any]:
    """Return the complete writer-visible packet with no controller metadata."""
    _assert_clean_writer_output(system, user)
    return {
        "schema": "m5-writer-packet-v2",
        "system": system,
        "user": user,
        "generation": {
            "chapter_id": chapter.chapter_id,
            "target_han_chars": chapter.target_chars,
            "allowed_han_chars": list(WRITER_REQUESTED_HAN_CHARS),
            "output_format": "chapter_title_then_continuous_body",
        },
    }


def _packed(
    selected: Sequence[ProjectedMemoryBlock], dropped_count: int,
) -> PackedMemory:
    text = "\n\n".join(block.text for block in selected)
    return PackedMemory(
        text=text,
        blocks=tuple(selected),
        memory_tokens=count_tokens(text),
        dropped_count=dropped_count,
    )


def _validate_writer_text(text: str, node: Node) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        raise PromptProjectionError("Projected writer text is empty")
    secrets = (
        node.node_id,
        str(node.metadata.get("state_key", "")),
    )
    if any(secret and secret in clean for secret in secrets):
        raise PromptProjectionError("Projected text contains an internal identifier")
    if _FORBIDDEN_WRITER_MARKERS.search(clean):
        raise PromptProjectionError("Projected text contains internal graph metadata")
    return clean


def _assert_clean_writer_output(*parts: str) -> None:
    joined = "\n".join(parts)
    if _FORBIDDEN_WRITER_MARKERS.search(joined):
        raise PromptProjectionError("Writer-visible prompt contains internal metadata")


def _information_score(text: str, category: str) -> int:
    normalized = _normalized_body(text)
    detail_bonus = {
        "atomic_state": 4,
        "chapter_summary": 3,
        "chapter_result": 2,
        "volume_brief": 1,
    }.get(category, 0)
    return len(normalized) * 10 + detail_bonus


def _hierarchical_duplicate(
    left: ProjectedMemoryBlock, right: ProjectedMemoryBlock,
) -> bool:
    summary_categories = {"chapter_result", "chapter_summary"}
    return (
        left.chapter_id > 0
        and left.chapter_id == right.chapter_id
        and left.category in summary_categories
        and right.category in summary_categories
    )


def _textually_duplicate(left: str, right: str) -> bool:
    a, b = _normalized_body(left), _normalized_body(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 8 and shorter in longer:
        return True
    shingles_a, shingles_b = _shingles(a), _shingles(b)
    union = shingles_a | shingles_b
    return bool(union) and len(shingles_a & shingles_b) / len(union) >= 0.72


def _normalized_body(text: str) -> str:
    body = text.split("：", 1)[-1]
    return _NORMALIZE.sub("", body).casefold()


def _shingles(text: str) -> set[str]:
    if len(text) < 2:
        return {text}
    return {text[index : index + 2] for index in range(len(text) - 1)}
