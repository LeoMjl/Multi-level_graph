from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import tiktoken

from mlg.m4_data import TOKENIZER
from mlg.m4_retrieval import normalized


SCHEMA_PROTOCOL = "m4-unified-schema-v1"
TARGET_UNIT_TOKENS = 220
MAX_UNIT_TOKENS = 320
MAX_EVENT_UNITS = 3
EVENT_SIMILARITY = 0.40

_ENCODER = tiktoken.get_encoding(TOKENIZER)
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=(?:[\"'“‘(]*[A-Z0-9]))"
)
_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=[ \t]*\n[ \t]*\n|\s*\Z)", re.DOTALL)


def _sentence_spans(body: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for paragraph in _PARAGRAPH.finditer(body):
        text = paragraph.group(0)
        cursor = 0
        for boundary in _SENTENCE_BOUNDARY.finditer(text):
            if boundary.start() > cursor:
                spans.append((paragraph.start() + cursor, paragraph.start() + boundary.start()))
            cursor = boundary.end()
        if cursor < len(text):
            spans.append((paragraph.start() + cursor, paragraph.end()))
    return spans or ([(0, len(body))] if body.strip() else [])


def _trimmed_span(body: str, start: int, end: int) -> tuple[int, int]:
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return start, end


def build_atomic_units(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for document in documents:
        body = str(document.get("body", ""))
        sentences = _sentence_spans(body)
        groups: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        current_tokens = 0
        for span in sentences:
            sentence_tokens = len(_ENCODER.encode(body[span[0]:span[1]]))
            if current and current_tokens + sentence_tokens > MAX_UNIT_TOKENS:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(span)
            current_tokens += sentence_tokens
            if current_tokens >= TARGET_UNIT_TOKENS:
                groups.append(current)
                current, current_tokens = [], 0
        if current:
            if groups and sum(len(_ENCODER.encode(body[a:b])) for a, b in current) < 48:
                groups[-1].extend(current)
            else:
                groups.append(current)

        doc_id = str(document["doc_id"])
        header = "\n".join(
            f"{key}: {document[key]}"
            for key in ("title", "author", "source", "category", "published_at")
            if document.get(key)
        )
        for unit_index, group in enumerate(groups):
            start, end = _trimmed_span(body, group[0][0], group[-1][1])
            fact_text = body[start:end]
            units.append({
                "unit_id": f"taskgraph:{doc_id}:fact:{unit_index:04d}",
                "doc_id": doc_id,
                "unit_index": unit_index,
                "body_char_start": start,
                "body_char_end": end,
                "fact_text": fact_text,
                "text": f"{header}\n\n{fact_text}" if header else fact_text,
                "title": str(document.get("title", "")),
                "source": str(document.get("source", "")),
                "published_at": str(document.get("published_at", "")),
            })
    return units


def build_report_events(
    units: list[dict[str, Any]], unit_vectors: Any,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    vectors = normalized(unit_vectors)
    by_doc: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        by_doc[str(unit["doc_id"])].append(index)
    events: list[dict[str, Any]] = []
    event_vectors: list[np.ndarray] = []
    for doc_id, indices in by_doc.items():
        indices.sort(key=lambda index: int(units[index]["unit_index"]))
        groups: list[list[int]] = []
        current: list[int] = []
        for index in indices:
            similarity = 1.0 if not current else float(
                vectors[index] @ normalized(np.mean(vectors[current], axis=0, keepdims=True))[0]
            )
            if current and (len(current) >= MAX_EVENT_UNITS or similarity < EVENT_SIMILARITY):
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)
        for event_index, group in enumerate(groups):
            first = units[group[0]]
            body = "\n\n".join(str(units[index]["fact_text"]) for index in group)
            events.append({
                "event_id": f"taskgraph:{doc_id}:event:{event_index:04d}",
                "doc_id": doc_id,
                "event_index": event_index,
                "unit_indices": group,
                "text": f"Title: {first['title']}\n\n{body}",
                "source_text": body,
                "title": first["title"],
                "source": first["source"],
                "published_at": first["published_at"],
            })
            event_vectors.append(normalized(np.mean(vectors[group], axis=0, keepdims=True))[0])
    return events, normalized(np.vstack(event_vectors))
