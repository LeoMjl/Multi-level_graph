from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from mlg.m4_retrieval import normalized


MAX_SCHEMA_EVENTS = 12
PAIR_CANDIDATES = 8
MAX_SCENE_EVENTS = 3
MIN_SCENE_EVENTS = 2
SCENE_SIMILARITY = 0.42
MAX_EVENT_DEPENDENCIES = 3
_STOP = {
    "about", "after", "again", "against", "also", "among", "another", "article", "because",
    "been", "before", "being", "between", "could", "from", "have", "into", "more", "most",
    "news", "only", "other", "over", "report", "said", "says", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those", "through", "title", "under",
    "very", "what", "when", "where", "which", "while", "with", "would", "your",
}
_PROPER = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z0-9'’.-]+)(?:\s+(?:(?:of|the|and)\s+)?"
    r"(?:[A-Z]{2,}|[A-Z][A-Za-z0-9'’.-]+)){0,4}\b"
)


def _terms(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if len(token) >= 4 and token not in _STOP and not token.isdigit()
    }


def _entity_aliases(text: str) -> set[str]:
    aliases: set[str] = set()
    for match in _PROPER.finditer(str(text)):
        tokens = re.findall(r"[A-Za-z0-9]+", match.group(0))
        content = [token for token in tokens if token.casefold() not in {"the", "of", "and"}]
        if not content:
            continue
        normalized_phrase = " ".join(token.casefold() for token in content)
        if len(content) > 1:
            aliases.add(normalized_phrase)
            aliases.add("".join(token[0].casefold() for token in content))
            if len(content[-1]) >= 5:
                aliases.add(content[-1].casefold())
        elif len(content[0]) >= 4 and content[0].casefold() not in _STOP:
            aliases.add(content[0].casefold())
    return aliases


class _Components:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return True
        if self.size[left] + self.size[right] > MAX_SCHEMA_EVENTS:
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return True


def _quality_scene_specs(
    events: list[dict[str, Any]],
    vectors: np.ndarray,
    term_sets: list[set[str]],
    salient: list[set[str]],
    entities: list[set[str]],
    term_df: Counter,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    by_doc: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        by_doc.setdefault(str(event["doc_id"]), []).append(index)

    schemas: list[dict[str, Any]] = []
    schema_vectors: list[np.ndarray] = []
    scene_by_event: dict[int, int] = {}
    for doc_id, indices in by_doc.items():
        indices.sort(key=lambda index: int(events[index]["event_index"]))
        groups: list[list[int]] = []
        current: list[int] = []
        for index in indices:
            similarity = 1.0 if not current else float(
                vectors[index] @ normalized(np.mean(vectors[current], axis=0, keepdims=True))[0]
            )
            if current and (
                len(current) >= MAX_SCENE_EVENTS
                or (len(current) >= MIN_SCENE_EVENTS and similarity < SCENE_SIMILARITY)
            ):
                groups.append(current)
                current = []
            current.append(index)
        if current:
            groups.append(current)
        if len(groups) >= 2 and len(groups[-1]) == 1 and len(groups[-2]) < MAX_SCENE_EVENTS:
            groups[-2].extend(groups.pop())

        for scene_index, group in enumerate(groups):
            schema_index = len(schemas)
            anchors = Counter(term for event_index in group for term in salient[event_index])
            labels = [term for term, _ in anchors.most_common(8)]
            schemas.append({
                "schema_id": f"taskgraph:{doc_id}:scene:{scene_index:04d}",
                "event_indices": group,
                "scene_index": scene_index,
                "text": "Narrative scene: " + ", ".join(labels),
                "anchors": labels,
                "doc_ids": [doc_id],
            })
            schema_vectors.append(normalized(np.mean(vectors[group], axis=0, keepdims=True))[0])
            for event_index in group:
                scene_by_event[event_index] = schema_index

    matrix = vectors @ vectors.T
    candidates: list[dict[str, Any]] = []
    for left, event in enumerate(events):
        for right in by_doc[str(event["doc_id"])]:
            if right <= left or scene_by_event[left] == scene_by_event[right]:
                continue
            if abs(int(events[left]["event_index"]) - int(events[right]["event_index"])) <= 1:
                continue
            score = float(matrix[left, right])
            shared_entities = entities[left] & entities[right]
            shared_terms = salient[left] & salient[right]
            entity_link = bool(shared_entities) and score >= 0.58 and (bool(shared_terms) or score >= 0.70)
            thematic_link = len(shared_terms) >= 2 and score >= 0.66
            if not (entity_link or thematic_link):
                continue
            candidates.append({
                "left": left, "right": right, "relation": "narrative_dependency",
                "cosine": score, "strength": score,
                "entities": sorted(shared_entities)[:5],
                "anchors": sorted(shared_terms, key=lambda term: (term_df[term], term))[:6],
            })
    degree: Counter = Counter()
    relations: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: float(item["strength"]), reverse=True):
        left, right = int(row["left"]), int(row["right"])
        if degree[left] >= MAX_EVENT_DEPENDENCIES or degree[right] >= MAX_EVENT_DEPENDENCIES:
            continue
        relations.append(row)
        degree[left] += 1
        degree[right] += 1
    return schemas, normalized(np.vstack(schema_vectors)), relations, {
        "relation_scope": "within_document",
        "schema_strategy": "contiguous_narrative_scenes",
        "schemas": len(schemas),
        "multi_document_schemas": 0,
        "same_entity_event_pairs": 0,
        "same_attribute_pairs": 0,
        "narrative_dependency_pairs": len(relations),
        "max_dependency_degree": max(degree.values(), default=0),
    }


def build_schema_specs(
    events: list[dict[str, Any]], event_vectors: Any, *, relation_scope: str = "cross_document",
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    if relation_scope not in {"within_document", "cross_document", "hybrid"}:
        raise ValueError(f"Unsupported schema relation scope: {relation_scope}")
    vectors = normalized(event_vectors)
    term_sets = [_terms(f"{event['title']} {event['source_text']}") for event in events]
    term_df = Counter(term for terms in term_sets for term in terms)
    rare_ceiling = max(12, int(len(events) * 0.08))
    salient = [
        set(sorted(
            (term for term in terms if 1 < term_df[term] <= rare_ceiling),
            key=lambda term: (term_df[term], term),
        )[:18])
        for terms in term_sets
    ]
    entities = [
        _entity_aliases(f"{event['title']}\n{event['source_text'][:1600]}")
        for event in events
    ]
    entity_df = Counter(alias for values in entities for alias in values)
    entity_ceiling = max(16, int(len(events) * 0.08))
    entities = [
        {alias for alias in values if entity_df[alias] <= entity_ceiling}
        for values in entities
    ]
    if relation_scope == "within_document":
        return _quality_scene_specs(events, vectors, term_sets, salient, entities, term_df)
    matrix = vectors @ vectors.T
    relations: dict[tuple[int, int], dict[str, Any]] = {}
    for left, event in enumerate(events):
        candidates_in_scope = [
            right for right, candidate in enumerate(events)
            if right != left and (
                relation_scope == "hybrid"
                or (relation_scope == "within_document" and candidate["doc_id"] == event["doc_id"])
                or (relation_scope == "cross_document" and candidate["doc_id"] != event["doc_id"])
            )
        ]
        count = min(PAIR_CANDIDATES, len(candidates_in_scope))
        if not count:
            continue
        scoped_scores = matrix[left, candidates_in_scope]
        selected = np.argpartition(scoped_scores, -count)[-count:]
        for candidate_position in selected:
            right = int(candidates_in_scope[int(candidate_position)])
            score = float(matrix[left, right])
            if score < 0.43:
                continue
            pair = tuple(sorted((left, right)))
            shared_entities = entities[left] & entities[right]
            shared_terms = salient[left] & salient[right]
            relation = ""
            if shared_entities and (shared_terms or score >= 0.68):
                relation = "same_entity_event"
            elif shared_terms and score >= 0.58:
                relation = "same_attribute"
            if not relation:
                continue
            previous = relations.get(pair)
            if previous is None or score > previous["cosine"]:
                relations[pair] = {
                    "left": pair[0], "right": pair[1], "relation": relation,
                    "cosine": score, "strength": score,
                    "entities": sorted(shared_entities)[:5],
                    "anchors": sorted(shared_terms, key=lambda term: (term_df[term], term))[:6],
                }

    components = _Components(len(events))
    same_event = sorted(
        (row for row in relations.values() if row["relation"] == "same_entity_event"),
        key=lambda row: row["cosine"], reverse=True,
    )
    for row in same_event:
        components.union(int(row["left"]), int(row["right"]))
    members: dict[int, list[int]] = {}
    for index in range(len(events)):
        members.setdefault(components.find(index), []).append(index)

    schemas: list[dict[str, Any]] = []
    schema_vectors: list[np.ndarray] = []
    for schema_index, indices in enumerate(sorted(members.values(), key=lambda rows: rows[0])):
        anchors = Counter(term for index in indices for term in salient[index])
        labels = [term for term, _ in anchors.most_common(8)]
        titles = list(dict.fromkeys(str(events[index]["title"]) for index in indices))[:3]
        schemas.append({
            "schema_id": f"taskgraph:schema:{schema_index:04d}",
            "event_indices": indices,
            "text": "Schema: " + ", ".join(labels or sorted(_terms(" ".join(titles))))
            + "\nReports: " + " | ".join(titles),
            "anchors": labels,
            "doc_ids": sorted({str(events[index]["doc_id"]) for index in indices}),
        })
        schema_vectors.append(normalized(np.mean(vectors[indices], axis=0, keepdims=True))[0])
    metadata = {
        "relation_scope": relation_scope,
        "schemas": len(schemas),
        "multi_document_schemas": sum(len(spec["doc_ids"]) > 1 for spec in schemas),
        "same_entity_event_pairs": sum(row["relation"] == "same_entity_event" for row in relations.values()),
        "same_attribute_pairs": sum(row["relation"] == "same_attribute" for row in relations.values()),
    }
    return schemas, normalized(np.vstack(schema_vectors)), list(relations.values()), metadata
