from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mlg.graph.task_graph import EdgeType, NodeLevel, NodeStatus, TaskGraph
from mlg.m4_model import M4Model
from mlg.m4_retrieval import normalized


TASKGRAPH_PROTOCOL = "m4-taskgraph-semantic-v1"
SUMMARY_BATCH_SIZE = 8


class _SummaryCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, str]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("protocol") == TASKGRAPH_PROTOCOL:
                    self.rows[str(row["item_id"])] = row

    def get(self, item_id: str, text: str) -> str:
        row = self.rows.get(item_id, {})
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return str(row.get("summary", "")) if row.get("source_sha256") == digest else ""

    def put(self, item_id: str, text: str, summary: str) -> None:
        row = {
            "protocol": TASKGRAPH_PROTOCOL,
            "item_id": item_id,
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "summary": summary,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.rows[item_id] = row


def _topic_groups(indices: list[int], vectors: np.ndarray) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for index in indices:
        if not current:
            current = [index]
            continue
        centroid = normalized(np.mean(vectors[current], axis=0, keepdims=True))[0]
        similarity = float(vectors[index] @ centroid)
        if len(current) < 4 and (len(current) < 2 or similarity >= 0.38):
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    if current:
        groups.append(current)
    if len(groups) >= 2 and len(groups[-1]) == 1 and len(groups[-2]) < 4:
        groups[-2].extend(groups.pop())
    return groups


def _summarize(
    items: list[dict[str, str]], model: M4Model, cache: _SummaryCache,
) -> tuple[dict[str, str], dict[str, Any]]:
    values = {item["item_id"]: cache.get(item["item_id"], item["text"]) for item in items}
    pending = [item for item in items if not values[item["item_id"]]]
    calls = []
    for start in range(0, len(pending), SUMMARY_BATCH_SIZE):
        batch = pending[start:start + SUMMARY_BATCH_SIZE]
        summaries, call = model.taskgraph_summaries_batch(batch)
        calls.append(call)
        for item in batch:
            summary = summaries[item["item_id"]]
            cache.put(item["item_id"], item["text"], summary)
            values[item["item_id"]] = summary
        print(
            f"[M4] TaskGraph summaries {min(start + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )
    return values, {"requested": len(items), "generated": len(pending), "reused": len(items) - len(pending), "calls": calls}


def prepare_semantic_nodes(
    dataset: str,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    chunk_vectors: Any,
    model: M4Model,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], np.ndarray, np.ndarray, dict[str, Any]]:
    vectors = normalized(chunk_vectors)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        grouped[str(chunk["doc_id"])].append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: int(chunks[index]["chunk_index"]))

    topic_specs: list[dict[str, Any]] = []
    topic_inputs: list[dict[str, str]] = []
    for document in documents:
        doc_id = str(document["doc_id"])
        for topic_index, indices in enumerate(_topic_groups(grouped[doc_id], vectors)):
            item_id = f"taskgraph:{doc_id}:topic:{topic_index:04d}"
            body = "\n\n".join(str(chunks[index]["fact_text"]) for index in indices)
            topic_specs.append({
                "item_id": item_id,
                "doc_id": doc_id,
                "topic_index": topic_index,
                "chunk_indices": indices,
                "source_text": body,
            })
            topic_inputs.append({
                "item_id": item_id,
                "text": f"Title: {document.get('title', doc_id)}\n\n{body}",
            })

    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache = _SummaryCache(artifact_dir / "summaries.jsonl")
    topic_summaries, topic_meta = _summarize(topic_inputs, model, cache)
    for spec in topic_specs:
        spec["summary"] = topic_summaries[spec["item_id"]]

    doc_inputs = []
    doc_summaries: dict[str, str] = {}
    topics_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in topic_specs:
        topics_by_doc[spec["doc_id"]].append(spec)
    for document in documents:
        doc_id = str(document["doc_id"])
        if dataset == "multihop_rag" and len(topics_by_doc[doc_id]) == 1:
            doc_summaries[doc_id] = str(topics_by_doc[doc_id][0]["summary"])
            continue
        doc_inputs.append({
            "item_id": f"taskgraph:{doc_id}:document",
            "text": f"Title: {document.get('title', doc_id)}\n\n" + "\n".join(
                spec["summary"] for spec in topics_by_doc[doc_id]
            ),
        })
    doc_summaries_by_item, doc_meta = _summarize(doc_inputs, model, cache)
    doc_summaries.update({
        str(document["doc_id"]): doc_summaries_by_item[f"taskgraph:{document['doc_id']}:document"]
        for document in documents if f"taskgraph:{document['doc_id']}:document" in doc_summaries_by_item
    })
    doc_meta["inherited_single_topic"] = len(documents) - len(doc_inputs)

    topic_ids = [spec["item_id"] for spec in topic_specs]
    topic_vectors, topic_vector_meta = model.cached_embeddings(
        [spec["summary"] for spec in topic_specs], topic_ids,
        dataset=dataset, kind=f"taskgraph_topics_{TASKGRAPH_PROTOCOL}",
    )
    doc_ids = [f"taskgraph:{document['doc_id']}:document" for document in documents]
    doc_vectors, doc_vector_meta = model.cached_embeddings(
        [doc_summaries[str(document["doc_id"])] for document in documents], doc_ids,
        dataset=dataset, kind=f"taskgraph_documents_{TASKGRAPH_PROTOCOL}",
    )
    metadata = {
        "protocol": TASKGRAPH_PROTOCOL,
        "topics": len(topic_specs),
        "topic_summaries": topic_meta,
        "document_summaries": doc_meta,
        "topic_vector_cache": topic_vector_meta,
        "document_vector_cache": doc_vector_meta,
    }
    return topic_specs, doc_summaries, normalized(topic_vectors), normalized(doc_vectors), metadata
