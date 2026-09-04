from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np

from mlg.graph.task_graph import EdgeType, NodeLevel, NodeStatus, TaskGraph
from mlg.m4_retrieval import normalized


def _source_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _date_key(value: str) -> str:
    match = re.search(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", str(value))
    return "" if match is None else "-".join(match.groups())


def _source_temporal_pairs(documents: list[dict[str, Any]]) -> list[tuple[int, int, str, str, str]]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    source_names: dict[str, str] = {}
    for index, document in enumerate(documents):
        source = str(document.get("source", "")).strip()
        source_key = _source_key(source)
        date = _date_key(str(document.get("published_at", "")))
        if source_key and date:
            grouped[source_key].append((date, index))
            source_names.setdefault(source_key, source)
    pairs = []
    for source_key, dated in grouped.items():
        dated.sort()
        for (left_date, left), (right_date, right) in zip(dated, dated[1:]):
            pairs.append((left, right, source_names[source_key], left_date, right_date))
    return pairs


def _dependency_pairs(
    indices: list[int], vectors: np.ndarray, *, threshold: float = 0.25,
) -> list[tuple[int, int, float]]:
    if len(indices) < 3:
        return []
    matrix = vectors[indices] @ vectors[indices].T
    pairs: dict[tuple[int, int], float] = {}
    for position, source in enumerate(indices):
        candidates = [item for item in range(len(indices)) if abs(item - position) > 1]
        candidates.sort(key=lambda item: float(matrix[position, item]), reverse=True)
        for target_position in candidates[:2]:
            score = float(matrix[position, target_position])
            if score < threshold:
                continue
            target = indices[target_position]
            pair = tuple(sorted((source, target)))
            pairs[pair] = max(pairs.get(pair, -1.0), score)
    return [(source, target, score) for (source, target), score in sorted(pairs.items())]


def _cross_document_pairs(
    indices: list[int], vectors: np.ndarray, doc_ids: list[str], *, count: int, threshold: float = 0.25,
) -> list[tuple[int, int, float]]:
    matrix = vectors[indices] @ vectors[indices].T
    pairs: dict[tuple[int, int], float] = {}
    for position, source in enumerate(indices):
        candidates = [
            item for item, target in enumerate(indices)
            if doc_ids[target] != doc_ids[source]
        ]
        candidates.sort(key=lambda item: float(matrix[position, item]), reverse=True)
        for target_position in candidates[:count]:
            score = float(matrix[position, target_position])
            if score < threshold:
                continue
            target = indices[target_position]
            pair = tuple(sorted((source, target)))
            pairs[pair] = max(pairs.get(pair, -1.0), score)
    return [(source, target, score) for (source, target), score in sorted(pairs.items())]


def assemble_taskgraph(
    dataset: str,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    chunk_vectors: Any,
    topic_specs: list[dict[str, Any]],
    topic_vectors: Any,
    doc_summaries: dict[str, str],
    doc_vectors: Any,
) -> tuple[TaskGraph, list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    graph = TaskGraph()
    root = graph.add_node(
        NodeLevel.L1, f"{dataset} document QA", 0, dataset,
        hint="Corpus", status=NodeStatus.ACTIVE, metadata={"dataset": dataset},
    )
    chunk_array = normalized(chunk_vectors)
    topic_array = normalized(topic_vectors)
    doc_array = normalized(doc_vectors)
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    chunks_by_doc: dict[str, list[int]] = defaultdict(list)
    topics_by_doc: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        chunks_by_doc[str(chunk["doc_id"])].append(index)
    for index, spec in enumerate(topic_specs):
        topics_by_doc[str(spec["doc_id"])].append(index)
    for indices in chunks_by_doc.values():
        indices.sort(key=lambda item: int(chunks[item]["chunk_index"]))
    for indices in topics_by_doc.values():
        indices.sort(key=lambda item: int(topic_specs[item]["topic_index"]))

    chunk_nodes: dict[int, str] = {}
    topic_nodes: dict[int, str] = {}
    doc_nodes: dict[int, str] = {}
    for doc_position, document in enumerate(documents):
        doc_id = str(document["doc_id"])
        doc_retrieval_id = f"taskgraph:{doc_id}:document"
        doc_node = graph.add_node(
            NodeLevel.L2, doc_summaries[doc_id], 0, f"{dataset}/{doc_id}",
            hint="Doc", status=NodeStatus.ACTIVE,
            metadata={
                "doc_id": doc_id, "retrieval_id": doc_retrieval_id,
                "source": str(document.get("source", "")),
                "published_at": str(document.get("published_at", "")),
            },
        )
        doc_nodes[doc_position] = doc_node
        graph.add_edge(root, doc_node, EdgeType.INCLUSION)
        records.append({
            "node_id": doc_node, "retrieval_id": doc_retrieval_id, "doc_id": doc_id,
            "level": NodeLevel.L2.value, "text": doc_summaries[doc_id], "order": 0,
            "source": str(document.get("source", "")),
            "published_at": str(document.get("published_at", "")),
        })
        vectors.append(doc_array[doc_position])

        previous_topic = ""
        for topic_global in topics_by_doc[doc_id]:
            spec = topic_specs[topic_global]
            topic_node = graph.add_node(
                NodeLevel.L3, str(spec["summary"]), int(spec["topic_index"]),
                f"{dataset}/{doc_id}/topic/{spec['topic_index']}", hint="Topic",
                metadata={"doc_id": doc_id, "retrieval_id": spec["item_id"]},
            )
            topic_nodes[topic_global] = topic_node
            graph.add_edge(doc_node, topic_node, EdgeType.INCLUSION)
            if previous_topic:
                graph.add_edge(previous_topic, topic_node, EdgeType.MAINLINE)
            previous_topic = topic_node
            records.append({
                "node_id": topic_node, "retrieval_id": spec["item_id"], "doc_id": doc_id,
                "level": NodeLevel.L3.value, "text": spec["summary"],
                "order": int(spec["topic_index"]),
            })
            vectors.append(topic_array[topic_global])
            for chunk_global in spec["chunk_indices"]:
                chunk = chunks[chunk_global]
                chunk_node = graph.add_node(
                    NodeLevel.L4, str(chunk["fact_text"]), int(chunk["chunk_index"]),
                    f"{dataset}/{doc_id}/chunk/{chunk['chunk_index']}", hint="Chunk",
                    metadata={"doc_id": doc_id, "chunk_id": chunk["chunk_id"], "retrieval_id": chunk["chunk_id"]},
                )
                chunk_nodes[chunk_global] = chunk_node
                graph.add_edge(topic_node, chunk_node, EdgeType.INCLUSION)
                records.append({
                    "node_id": chunk_node, "retrieval_id": chunk["chunk_id"], "doc_id": doc_id,
                    "level": NodeLevel.L4.value, "text": chunk["text"], "order": int(chunk["chunk_index"]),
                })
                vectors.append(chunk_array[chunk_global])

        ordered_chunks = chunks_by_doc[doc_id]
        for left, right in zip(ordered_chunks, ordered_chunks[1:]):
            graph.add_edge(chunk_nodes[left], chunk_nodes[right], EdgeType.MAINLINE)
        for left, right, score in _dependency_pairs(ordered_chunks, chunk_array):
            graph.add_edge(chunk_nodes[left], chunk_nodes[right], EdgeType.DEPENDENCY, {
                "relation": "semantic_nonlocal", "cosine": score, "bidirectional": True,
            })
        topic_indices = topics_by_doc[doc_id]
        for left, right, score in _dependency_pairs(topic_indices, topic_array):
            graph.add_edge(topic_nodes[left], topic_nodes[right], EdgeType.DEPENDENCY, {
                "relation": "topic_nonlocal", "cosine": score, "bidirectional": True,
            })

    if dataset == "multihop_rag":
        for left, right, source, left_date, right_date in _source_temporal_pairs(documents):
            graph.add_edge(doc_nodes[left], doc_nodes[right], EdgeType.MAINLINE, {
                "relation": "source_temporal_next", "source": source,
                "left_date": left_date, "right_date": right_date,
            })
        document_ids = [str(document["doc_id"]) for document in documents]
        for left, right, score in _cross_document_pairs(
            list(range(len(documents))), doc_array, document_ids, count=3,
        ):
            graph.add_edge(doc_nodes[left], doc_nodes[right], EdgeType.DEPENDENCY, {
                "relation": "document_crossdoc", "cosine": score, "bidirectional": True,
            })
        topic_doc_ids = [str(spec["doc_id"]) for spec in topic_specs]
        for left, right, score in _cross_document_pairs(
            list(range(len(topic_specs))), topic_array, topic_doc_ids, count=2,
        ):
            graph.add_edge(topic_nodes[left], topic_nodes[right], EdgeType.DEPENDENCY, {
                "relation": "topic_crossdoc", "cosine": score, "bidirectional": True,
            })
        chunk_doc_ids = [str(chunk["doc_id"]) for chunk in chunks]
        for left, right, score in _cross_document_pairs(
            list(range(len(chunks))), chunk_array, chunk_doc_ids, count=1,
        ):
            graph.add_edge(chunk_nodes[left], chunk_nodes[right], EdgeType.DEPENDENCY, {
                "relation": "chunk_crossdoc", "cosine": score, "bidirectional": True,
            })

    edge_counts: dict[str, int] = defaultdict(int)
    relation_counts: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        edge_counts[edge.edge_type.value] += 1
        relation_counts[str(edge.metadata.get("relation", edge.edge_type.value.lower()))] += 1
    return graph, records, normalized(np.vstack(vectors)), {
        "retrieval_nodes": len(records), "level_counts": dict(_counts(row["level"] for row in records)),
        "edge_type_counts": dict(edge_counts), "edge_relation_counts": dict(relation_counts),
    }


def _counts(values) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for value in values:
        output[str(value)] += 1
    return output
