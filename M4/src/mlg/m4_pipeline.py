from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import tiktoken

from mlg.m4_data import M4_DIR, prepare_m4_data
from mlg.m4_graphrag import build_graphrag, retrieve_graphrag
from mlg.m4_memorystream import build_memorystream
from mlg.m4_model import M4Model
from mlg.m4_raptor import build_raptor, retrieve_raptor
from mlg.m4_retrieval import RetrievalIndex, read_jsonl
from mlg.m4_taskgraph import build_taskgraph_index


METHODS = {
    "quality": ["Full Text", "RAPTOR", "GraphRAG", "MemoryStream", "TaskGraph"],
    "multihop_rag": ["RAPTOR", "GraphRAG", "MemoryStream", "TaskGraph"],
}
CONTEXT_TOKENS = 8192
TASKGRAPH_CONTEXT_TOKENS = 8000
TASKGRAPH_CANDIDATES = 32
QUALITY_TASKGRAPH_CANDIDATES = 32
PROTOCOL_VERSION = "m4-target-paper-v19-quality-narrative-schema"
ENCODER = tiktoken.get_encoding("o200k_base")


def _query_text(row: dict[str, Any]) -> str:
    return row["question"]


def _document_metadata(text: str) -> dict[str, str]:
    metadata = {}
    for line in str(text).split("\n\n", 1)[0].splitlines():
        key, separator, value = line.partition(":")
        normalized_key = key.strip().casefold()
        if separator and normalized_key in {"title", "source", "published_at"}:
            metadata[normalized_key] = value.strip()
    return metadata


def _context(method: str, retrieval: list[dict[str, Any]]) -> str:
    if method == "TaskGraph":
        document_lines = []
        seen_docs = set()
        for row in retrieval:
            doc_id = str(row.get("doc_id", ""))
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            metadata = _document_metadata(str(row.get("text", "")))
            fields = [f"DOC {len(document_lines) + 1}", f"doc={doc_id}"]
            source = str(row.get("source", "") or metadata.get("source", ""))
            published_at = str(row.get("published_at", "") or metadata.get("published_at", ""))
            if source:
                fields.append(f"source={source}")
            if published_at:
                fields.append(f"date={published_at}")
            document_lines.append(f"[{' | '.join(fields)}]")
        document_map = "[DOCUMENT MAP]\n" + "\n".join(document_lines)
        blocks = []
        for index, row in enumerate(retrieval, 1):
            node_kind = str(row.get("node_kind", ""))
            block_kind = (
                "DOCUMENT OVERVIEW" if node_kind == "document_overview"
                else "NARRATIVE SCENE" if node_kind == "narrative_scene"
                else "EVIDENCE"
            )
            fields = [
                f"{block_kind} {index}", f"node={row['chunk_id']}",
                f"doc={row.get('doc_id', '')}", f"origin={row.get('origin', 'direct')}",
            ]
            if row.get("schema_id"):
                fields.append(f"scene={row['schema_id']}")
            if node_kind == "atomic_evidence":
                fields.append(f"document_order={row.get('order', 0)}")
            if row.get("origin") == "edge_routed":
                fields.extend([
                    f"edge={row.get('edge_type', '')}",
                    f"relation={row.get('route_relation', '')}",
                    f"direction={row.get('route_direction', '')}",
                    f"seed={row.get('seed_node_id', '')}",
                ])
                if row.get("route_label"):
                    fields.append(f"route_terms={row['route_label']}")
                if row.get("route_source"):
                    fields.append(f"route_source={row['route_source']}")
                if row.get("route_date"):
                    fields.append(f"route_date={row['route_date']}")
                if row.get("route_bundle_id"):
                    fields.append(f"bundle={row['route_bundle_id']}")
                    fields.append(f"bundle_role={row.get('route_bundle_role', '')}")
                if row.get("route_pair_dates"):
                    fields.append(f"pair_dates={row['route_pair_dates']}")
                if row.get("route_path_relation"):
                    fields.append(f"path={row['route_path_relation']}")
            blocks.append(f"[{' | '.join(fields)}]\n{row['text']}")
        return f"{document_map}\n\n" + "\n\n".join(blocks)
    return "\n\n".join(f"[{row['chunk_id']}]\n{row['text']}" for row in retrieval)


def _budget(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    seen_text: set[str] = set()
    token_limit = TASKGRAPH_CONTEXT_TOKENS if method == "TaskGraph" else CONTEXT_TOKENS
    bundles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if method == "TaskGraph":
        for row in rows:
            if row.get("route_bundle_id"):
                bundles[str(row["route_bundle_id"])].append(row)
    handled_bundles: set[str] = set()
    for row in rows:
        bundle_id = str(row.get("route_bundle_id", ""))
        if bundle_id:
            if bundle_id in handled_bundles:
                continue
            handled_bundles.add(bundle_id)
            additions = sorted(
                bundles[bundle_id],
                key=lambda item: 0 if item.get("route_bundle_role") == "earlier" else 1,
            )
        else:
            additions = [row]
        unique_additions = []
        local_texts = set(seen_text)
        for addition in additions:
            text_key = " ".join(str(addition["text"]).split())
            if method == "TaskGraph" and text_key in local_texts:
                continue
            unique_additions.append(addition)
            local_texts.add(text_key)
        if not unique_additions:
            continue
        candidate = selected + unique_additions
        if len(ENCODER.encode(_context(method, candidate))) > token_limit:
            continue
        selected.extend(unique_additions)
        seen_text = local_texts
    return selected


def _truncate(text: str) -> str:
    tokens = ENCODER.encode(text)
    return text if len(tokens) <= CONTEXT_TOKENS else ENCODER.decode(tokens[:CONTEXT_TOKENS])


def build_inputs(
    dataset: str, model: M4Model, output_dir: Path, config_path: Path,
    *, methods: list[str] | None = None, query_limit: int = 0,
) -> tuple[list[dict], dict[str, str], dict[str, Any]]:
    selected_methods = methods or METHODS[dataset]
    taskgraph_candidate_limit = QUALITY_TASKGRAPH_CANDIDATES if dataset == "quality" else TASKGRAPH_CANDIDATES
    invalid = sorted(set(selected_methods) - set(METHODS[dataset]))
    if invalid:
        raise ValueError(f"Unsupported {dataset} methods: {invalid}")
    all_documents = read_jsonl(M4_DIR / f"{dataset}_documents.jsonl")
    all_chunks = read_jsonl(M4_DIR / f"{dataset}_chunks.jsonl")
    chunk_vectors, chunk_meta = model.cached_embeddings(
        [row["text"] for row in all_chunks], [row["chunk_id"] for row in all_chunks], dataset=dataset, kind="chunks",
    )
    all_queries = read_jsonl(M4_DIR / f"{dataset}_queries.jsonl")
    all_query_vectors, query_meta = model.cached_embeddings(
        [_query_text(row) for row in all_queries], [row["query_id"] for row in all_queries], dataset=dataset, kind="queries",
    )
    selected_ids = set(sorted(row["query_id"] for row in all_queries)[:query_limit]) if query_limit else set()
    query_positions = [
        index for index, row in enumerate(all_queries)
        if not selected_ids or row["query_id"] in selected_ids
    ]
    queries = [all_queries[index] for index in query_positions]
    query_vectors = [all_query_vectors[index] for index in query_positions]
    selected_doc_ids = {row.get("doc_id", "") for row in queries} if dataset == "quality" else set()
    documents = [row for row in all_documents if not selected_doc_ids or row["doc_id"] in selected_doc_ids]
    chunk_positions = [
        index for index, row in enumerate(all_chunks)
        if not selected_doc_ids or row["doc_id"] in selected_doc_ids
    ]
    chunks = [all_chunks[index] for index in chunk_positions]
    chunk_vectors = [chunk_vectors[index] for index in chunk_positions]
    index = RetrievalIndex(dataset, documents, chunks, chunk_vectors)
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / f"{dataset}_taskgraph.json"
    graph = None
    taskgraph_index = None
    taskgraph_meta = None
    memory_index = None
    memory_meta = None
    if "MemoryStream" in selected_methods:
        memory_vectors, memory_meta = build_memorystream(
            dataset, chunks, model, output_dir / "indexes" / "memorystream"
        )
        memory_index = RetrievalIndex(dataset, documents, chunks, memory_vectors)
    if "TaskGraph" in selected_methods:
        taskgraph_index, taskgraph_meta = build_taskgraph_index(
            dataset, documents, chunks, chunk_vectors, model,
            output_dir / "indexes" / "taskgraph" / dataset,
        )
        graph = taskgraph_index.graph
        graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    document_text = {row["doc_id"]: row["body"] for row in documents}
    retrieved: dict[str, list[list[dict]]] = {}
    retrieval_meta: dict[str, dict[str, Any]] = {}
    for method in selected_methods:
        if method == "MemoryStream":
            assert memory_index is not None and memory_meta is not None
            retrieved[method] = [
                _budget(method, memory_index.flat(
                    vector, doc_id=query.get("doc_id", ""), policy="memorystream_llm_key"
                ))
                for query, vector in zip(queries, query_vectors)
            ]
            retrieval_meta[method] = memory_meta
        elif method == "TaskGraph":
            assert taskgraph_index is not None and taskgraph_meta is not None
            retrieved[method] = [
                _budget(method, taskgraph_index.retrieve(
                    vector, doc_id=query.get("doc_id", ""), k=taskgraph_candidate_limit,
                    query_text=_query_text(query),
                ))
                for query, vector in zip(queries, query_vectors)
            ]
            taskgraph_audit = taskgraph_index.audit(
                retrieved[method], query_vectors, [str(query.get("doc_id", "")) for query in queries],
            )
            if not taskgraph_audit["passed"]:
                raise RuntimeError(f"TaskGraph retrieval audit failed: {taskgraph_audit}")
            retrieval_meta[method] = {
                **taskgraph_meta, "retrieval_audit": taskgraph_audit,
                "candidate_limit": taskgraph_candidate_limit,
                "context_token_target": TASKGRAPH_CONTEXT_TOKENS,
            }
        elif method == "RAPTOR":
            raptor_dir = output_dir / "indexes" / "raptor" / dataset
            build_raptor(dataset, documents, chunks, chunk_vectors, model, raptor_dir)
            retrieved[method] = [
                _budget(method, rows)
                for rows in retrieve_raptor(dataset, queries, query_vectors, raptor_dir)
            ]
        elif method == "GraphRAG":
            graph_rag_dir = output_dir / "indexes" / "graphrag"
            build_graphrag(dataset, graph_rag_dir, config_path)
            graph_rows, graph_meta = retrieve_graphrag(
                dataset, queries, query_vectors, graph_rag_dir, model
            )
            retrieved[method] = [_budget(method, rows) for rows in graph_rows]
            retrieval_meta[method] = graph_meta
    prepared = []
    retrieval_paths = {}
    for method in selected_methods:
        rows = []
        for query_index, query in enumerate(queries):
            item = dict(query)
            if method == "Full Text":
                item["context"] = document_text[query["doc_id"]]
                item["retrieval"] = []
            else:
                retrieval = retrieved[method][query_index]
                item["retrieval"] = retrieval
                item["context"] = _context(method, retrieval)
            item["method"] = method
            rows.append(item)
        path = output_dir / f"{dataset}_{_slug(method)}_retrieval.jsonl"
        _write_rows(path, rows)
        retrieval_paths[method] = str(path)
        prepared.extend(rows)
    audit = {
        "source_documents": len(all_documents), "source_chunks": len(all_chunks), "source_queries": len(all_queries),
        "documents": len(documents), "chunks": len(chunks), "queries": len(queries),
        "methods": selected_methods,
        "graph_nodes": len(graph.nodes) if graph else 0, "graph_edges": len(graph.edges) if graph else 0,
        "graph_path": str(graph_path) if graph else "",
        "chunk_vector_cache": chunk_meta, "query_vector_cache": query_meta, "retrieval_files": retrieval_paths,
        "retrieval_meta": retrieval_meta,
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest() if graph else "",
    }
    return prepared, document_text, audit


def run_generation(
    dataset: str, prepared: list[dict], document_text: dict[str, str], output_dir: Path,
    config_path: Path, *, methods: list[str] | None = None,
    workers: int = 6, batch_size: int = 8, limit: int = 0,
) -> dict[str, Any]:
    selected_methods = methods or METHODS[dataset]
    selected_ids = {row["query_id"] for row in prepared if row["method"] == selected_methods[0]}
    if limit:
        selected_ids = set(sorted(selected_ids)[:limit])
    summaries = {}
    for method in selected_methods:
        rows = [row for row in prepared if row["method"] == method and row["query_id"] in selected_ids]
        output_path = output_dir / f"{dataset}_{_slug(method)}_predictions.jsonl"
        current = {row["query_id"]: row for row in rows}
        valid = {}
        if output_path.exists():
            for row in read_jsonl(output_path):
                query_id = row.get("query_id")
                expected = current.get(query_id)
                if (
                    expected is not None
                    and row.get("method") == method
                    and row.get("status") == "ok"
                    and bool(str(row.get("answer", "")).strip())
                    and row.get("protocol_version") == PROTOCOL_VERSION
                    and row.get("context_sha256") == hashlib.sha256(expected["context"].encode("utf-8")).hexdigest()
                ):
                    valid[query_id] = row
            _write_rows(output_path, [valid[key] for key in sorted(valid)])
        completed = set(valid)
        pending = [row for row in rows if row["query_id"] not in completed]
        batches = _make_batches(pending, method, batch_size)
        failures = _generate_batches(dataset, method, batches, output_path, config_path, workers, document_text)
        total = len(read_jsonl(output_path)) if output_path.exists() else 0
        summaries[method] = {"expected": len(rows), "completed": total, "failures": failures, "path": str(output_path)}
        if failures or total != len(rows):
            raise RuntimeError(f"{dataset}/{method} incomplete: {total}/{len(rows)}, failures={len(failures)}")
    return summaries


def _make_batches(rows: list[dict], method: str, batch_size: int) -> list[list[dict]]:
    if method == "Full Text":
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["doc_id"]].append(row)
        return [
            group[index : index + batch_size]
            for group in grouped.values()
            for index in range(0, len(group), batch_size)
        ]
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _generate_batches(dataset, method, batches, output_path, config_path, workers, document_text):
    local = threading.local()
    failure_path = output_path.with_suffix(".failures.jsonl")
    if failure_path.exists():
        failure_path.unlink()

    def execute(batch_id: int, batch: list[dict]):
        if not hasattr(local, "model"):
            local.model = M4Model.from_config(config_path)
        shared = document_text[batch[0]["doc_id"]] if method == "Full Text" else ""
        answers, call = local.model.answer_batch(dataset, batch, shared_context=shared)
        output = []
        for item in batch:
            output.append({
                "dataset": dataset, "method": method, "query_id": item["query_id"], "answer": answers[item["query_id"]],
                "status": "ok", "protocol_version": PROTOCOL_VERSION,
                "model": local.model.model, "retrieval": item["retrieval"],
                "context_sha256": hashlib.sha256(item["context"].encode("utf-8")).hexdigest(),
                "batch_id": batch_id, "api_call": call,
            })
        return output
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, index, batch): index for index, batch in enumerate(batches)}
        for completed, future in enumerate(as_completed(futures), 1):
            batch_id = futures[future]
            try:
                _append_rows(output_path, future.result())
            except Exception as exc:
                failure = {"batch_id": batch_id, "error": str(exc)}
                failures.append(failure)
                _append_rows(failure_path, [failure])
                print(
                    f"[M4] {dataset}/{method}: batch {batch_id} failed: {str(exc)[:240]}",
                    flush=True,
                )
            if completed == len(batches) or completed % 20 == 0:
                print(f"[M4] {dataset}/{method}: batches {completed}/{len(batches)}, failures={len(failures)}", flush=True)
    if not failures and failure_path.exists():
        failure_path.unlink()
    return failures


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _slug(method: str) -> str:
    return method.lower().replace(" ", "_")
