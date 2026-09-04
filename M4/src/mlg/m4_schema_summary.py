from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mlg.m4_retrieval import normalized


QUALITY_NARRATIVE_PROTOCOL = "m4-quality-narrative-schema-v2"
SUMMARY_BATCH_SIZE = 8


class _NarrativeSummaryCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, str]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("protocol") == QUALITY_NARRATIVE_PROTOCOL:
                    self.rows[str(row["item_id"])] = row

    def get(self, item_id: str, text: str) -> str:
        row = self.rows.get(item_id, {})
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return str(row.get("summary", "")) if row.get("source_sha256") == digest else ""

    def put(self, item_id: str, text: str, summary: str) -> None:
        row = {
            "protocol": QUALITY_NARRATIVE_PROTOCOL,
            "item_id": item_id,
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "summary": summary,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.rows[item_id] = row


def _summarize(items: list[dict[str, str]], model: Any, cache: _NarrativeSummaryCache) -> tuple[dict[str, str], dict[str, Any]]:
    values = {item["item_id"]: cache.get(item["item_id"], item["text"]) for item in items}
    pending = [item for item in items if not values[item["item_id"]]]
    calls: list[dict[str, Any]] = []
    for start in range(0, len(pending), SUMMARY_BATCH_SIZE):
        batch = pending[start:start + SUMMARY_BATCH_SIZE]
        summaries, call = model.taskgraph_summaries_batch(batch)
        calls.append(call)
        for item in batch:
            summary = str(summaries[item["item_id"]]).strip()
            cache.put(item["item_id"], item["text"], summary)
            values[item["item_id"]] = summary
        print(f"[M4] QuALITY narrative summaries {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    return values, {"requested": len(items), "generated": len(pending), "reused": len(items) - len(pending), "calls": calls}


def summarize_quality_hierarchy(
    documents: list[dict[str, Any]],
    events: list[dict[str, Any]],
    schemas: list[dict[str, Any]],
    model: Any,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, str], np.ndarray, dict[str, Any]]:
    cache = _NarrativeSummaryCache(artifact_dir / "narrative_summaries.jsonl")
    scene_inputs: list[dict[str, str]] = []
    for schema in schemas:
        source = "\n\n".join(str(events[index]["source_text"]) for index in schema["event_indices"])
        scene_inputs.append({
            "item_id": str(schema["schema_id"]),
            "text": f"Title: {events[schema['event_indices'][0]]['title']}\n\n{source}",
        })
    scene_summaries, scene_meta = _summarize(scene_inputs, model, cache)
    for schema in schemas:
        schema["summary"] = scene_summaries[str(schema["schema_id"])]
        schema["text"] = f"Narrative scene {int(schema.get('scene_index', 0)) + 1}: {schema['summary']}"

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for schema in schemas:
        by_doc.setdefault(str(schema["doc_ids"][0]), []).append(schema)
    doc_inputs = [{
        "item_id": f"taskgraph:{document['doc_id']}:document",
        "text": f"Title: {document.get('title', document['doc_id'])}\n\n" + "\n".join(
            str(schema["summary"]) for schema in by_doc[str(document["doc_id"])]
        ),
    } for document in documents]
    document_items, document_meta = _summarize(doc_inputs, model, cache)
    document_summaries = {
        str(document["doc_id"]): document_items[f"taskgraph:{document['doc_id']}:document"]
        for document in documents
    }
    schema_vectors, schema_vector_meta = model.cached_embeddings(
        [str(schema["text"]) for schema in schemas],
        [str(schema["schema_id"]) for schema in schemas],
        dataset="quality", kind=f"taskgraph_scenes_{QUALITY_NARRATIVE_PROTOCOL}",
    )
    document_vectors, document_vector_meta = model.cached_embeddings(
        [document_summaries[str(document["doc_id"])] for document in documents],
        [f"taskgraph:{document['doc_id']}:document" for document in documents],
        dataset="quality", kind=f"taskgraph_documents_{QUALITY_NARRATIVE_PROTOCOL}",
    )
    return schemas, normalized(schema_vectors), document_summaries, normalized(document_vectors), {
        "scene_summaries": scene_meta,
        "document_summaries": document_meta,
        "scene_vector_cache": schema_vector_meta,
        "document_vector_cache": document_vector_meta,
    }
