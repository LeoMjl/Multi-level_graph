from __future__ import annotations

import re
from dataclasses import dataclass

from mlg.methods.base import EMBEDDING_INPUT_MAX_CHARS, trim_text


RAW_CHUNK_MAX_CHARS = 7_000
RAW_CHUNK_OVERLAP_CHARS = 500
ATOMIC_FRAGMENT_MAX_CHARS = EMBEDDING_INPUT_MAX_CHARS
ATOMIC_FRAGMENT_BODY_CHARS = 6_400


@dataclass(frozen=True)
class MemoryUnit:
    unit_id: str
    text: str
    embedding_text: str
    trajectory_id: str
    state_index: str
    turn_index: int
    kind: str
    fragment_index: int = 0
    fragment_count: int = 1


def atomic_units_from_history(history: list[dict]) -> list[MemoryUnit]:
    """Create deterministic, lossless trajectory/state fragments.

    LongMemEval-V2 UI states are often much longer than one embedding request.  A
    head-tail view silently drops fields and actions in the middle of a state, so it
    cannot support the benchmark's static, dynamic, procedure, and abstention
    questions uniformly.  We instead split every long state on source line
    boundaries.  Each fragment carries compact trajectory/state/page provenance and
    stays within the embedding provider limit.  Atomic RAG and Ours call this same
    function, keeping graph relations as their controlled difference.
    """
    units: list[MemoryUnit] = []
    current_trajectory = ""
    for ordinal, message in enumerate(history, start=1):
        content = str(message.get("content", ""))
        if not content.strip():
            continue
        trajectory_match = re.search(r"\[trajectory_id=([^\]\s]+)", content)
        if trajectory_match:
            current_trajectory = trajectory_match.group(1)
        trajectory_id = current_trajectory or f"unknown-{ordinal}"
        state_match = re.search(r"\bstate_index=([^\]\s]+)", content[:240])
        state_index = state_match.group(1) if state_match else ""
        kind = "trajectory_state" if state_index else "trajectory_header"
        turn_index = int(message.get("turn_index") or ordinal)
        if kind == "trajectory_header" or len(content) <= ATOMIC_FRAGMENT_MAX_CHARS:
            bounded = trim_text(content, max_chars=ATOMIC_FRAGMENT_MAX_CHARS)
            units.append(MemoryUnit(
                unit_id=f"{trajectory_id}:{state_index or 'header'}:{ordinal}",
                text=bounded,
                embedding_text=bounded,
                trajectory_id=trajectory_id,
                state_index=state_index,
                turn_index=turn_index,
                kind=kind,
            ))
            continue

        fragments = _line_fragments(content, max_chars=ATOMIC_FRAGMENT_BODY_CHARS)
        provenance = _state_provenance(content, trajectory_id, state_index)
        fragment_count = len(fragments)
        for fragment_index, fragment in enumerate(fragments, start=1):
            prefix = (
                f"[memory_fragment trajectory_id={trajectory_id} state_index={state_index} "
                f"part={fragment_index}/{fragment_count}]\n{provenance}\n"
            )
            text = prefix + fragment
            if len(text) > ATOMIC_FRAGMENT_MAX_CHARS:
                # The body budget leaves ample room for provenance, but fail closed
                # if an unusually long URL/page title consumes that reserve.
                text = prefix + fragment[: max(0, ATOMIC_FRAGMENT_MAX_CHARS - len(prefix))]
            units.append(MemoryUnit(
                unit_id=f"{trajectory_id}:{state_index}:part{fragment_index:03d}:{ordinal}",
                text=text,
                embedding_text=text,
                trajectory_id=trajectory_id,
                state_index=state_index,
                turn_index=turn_index,
                kind="trajectory_state_fragment",
                fragment_index=fragment_index,
                fragment_count=fragment_count,
            ))
    return units


def _line_fragments(text: str, *, max_chars: int) -> list[str]:
    """Split text without discarding source characters or cutting normal tree lines."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    fragments: list[str] = []
    buffer = ""
    for line in text.splitlines(keepends=True):
        remaining = line
        while remaining:
            capacity = max_chars - len(buffer)
            if capacity == 0:
                fragments.append(buffer)
                buffer = ""
                capacity = max_chars
            take = remaining[:capacity]
            buffer += take
            remaining = remaining[len(take):]
            if len(buffer) == max_chars:
                fragments.append(buffer)
                buffer = ""
    if buffer:
        fragments.append(buffer)
    return fragments or [""]


def _state_provenance(content: str, trajectory_id: str, state_index: str) -> str:
    """Return generic page/action context repeated on every state fragment."""
    lines = content.splitlines()
    selected: list[str] = []
    for label in ("URL:", "Action:", "Thought:"):
        match = next((line.strip() for line in lines if line.startswith(label)), "")
        if match:
            selected.append(match)
    root = next((line.strip() for line in lines if "RootWebArea " in line), "")
    if root:
        selected.append(f"Page: {root}")
    if not selected:
        selected.append(f"trajectory={trajectory_id} state={state_index}")
    return trim_text("\n".join(selected), max_chars=900)


def raw_chunks_from_history(
    history: list[dict],
    *,
    max_chars: int = RAW_CHUNK_MAX_CHARS,
    overlap_chars: int = RAW_CHUNK_OVERLAP_CHARS,
) -> list[MemoryUnit]:
    """Chunk the serialized raw haystack without respecting semantic boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars")
    stride = max_chars - overlap_chars
    units: list[MemoryUnit] = []
    buffer = ""

    def append_chunk(chunk: str) -> None:
        index = len(units)
        units.append(MemoryUnit(
            unit_id=f"raw:{index}",
            text=chunk,
            embedding_text=chunk,
            trajectory_id="",
            state_index="",
            turn_index=index,
            kind="raw_chunk",
        ))

    for message in history:
        item = (
            f"[turn={message.get('turn_index', '')} role={message.get('role', 'user')}]\n"
            f"{message.get('content', '')}"
        )
        buffer = f"{buffer}\n\n{item}" if buffer else item
        while len(buffer) >= max_chars:
            append_chunk(buffer[:max_chars])
            buffer = buffer[stride:]
    if buffer and (not units or len(buffer) > overlap_chars):
        append_chunk(buffer)
    return units
