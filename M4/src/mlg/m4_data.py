from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import tiktoken

from mlg.config import PROCESSED_DIR, RAW_DIR


M4_DIR = PROCESSED_DIR / "m4_document_qa_target"
CHUNK_TOKENS = {"quality": 512, "multihop_rag": 1024}
CHUNK_OVERLAP = 0
TOKENIZER = "o200k_base"


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    payload = "".join(_stable(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return payload.count(b"\n"), _sha(payload)


def _chunks(document: dict[str, Any]) -> list[dict[str, Any]]:
    encoder = tiktoken.get_encoding(TOKENIZER)
    body_tokens = encoder.encode(str(document["body"]))
    chunk_tokens = CHUNK_TOKENS[str(document["dataset"])]
    step = chunk_tokens - CHUNK_OVERLAP
    header_fields = ["title", "author", "source", "category", "published_at"]
    header = "\n".join(
        f"{key}: {document[key]}" for key in header_fields if document.get(key)
    )
    output = []
    for chunk_index, start in enumerate(range(0, len(body_tokens), step)):
        token_slice = body_tokens[start : start + chunk_tokens]
        body = encoder.decode(token_slice)
        text = f"{header}\n\n{body}" if header else body
        output.append(
            {
                "chunk_id": f"{document['doc_id']}:chunk:{chunk_index:04d}",
                "doc_id": document["doc_id"],
                "chunk_index": chunk_index,
                "body_token_start": start,
                "body_token_count": len(token_slice),
                "text": text,
                "fact_text": body,
                "text_sha256": _sha(text.encode("utf-8")),
            }
        )
        if start + chunk_tokens >= len(body_tokens):
            break
    return output


def _prepare_quality() -> tuple[list[dict], list[dict], list[dict]]:
    path = RAW_DIR / "quality" / "QuALITY.v1.0.1.htmlstripped.dev"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    documents: dict[str, dict[str, Any]] = {}
    queries, labels = [], []
    for row in rows:
        doc_id = f"quality:{row['set_unique_id']}"
        document = {
            "dataset": "quality",
            "doc_id": doc_id,
            "source_article_id": str(row["article_id"]),
            "title": str(row.get("title", "")),
            "author": str(row.get("author", "")),
            "source": str(row.get("source", "")),
            "category": str(row.get("topic", "")),
            "published_at": str(row.get("year", "")),
            "body": str(row.get("article", "")),
        }
        documents[doc_id] = document
        for index, question in enumerate(row.get("questions", [])):
            query_id = f"quality:{row['set_unique_id']}:{index:03d}"
            options = [str(item) for item in question.get("options", [])]
            label = int(question.get("gold_label", 0))
            if len(options) != 4 or label not in {1, 2, 3, 4}:
                raise ValueError(f"Invalid QuALITY question: {query_id}")
            queries.append({
                "dataset": "quality",
                "query_id": query_id,
                "doc_id": doc_id,
                "question": str(question["question"]),
            })
            labels.append({
                "query_id": query_id,
                "answer": options[label - 1],
                "difficult": bool(question.get("difficult", 0)),
                "source_options": options,
                "source_gold_label": label,
            })
    return list(documents.values()), queries, labels


def _prepare_multihop() -> tuple[list[dict], list[dict], list[dict]]:
    corpus = json.loads((RAW_DIR / "multihop_rag" / "corpus.json").read_text(encoding="utf-8"))
    questions = json.loads((RAW_DIR / "multihop_rag" / "MultiHopRAG.json").read_text(encoding="utf-8"))
    documents, queries, labels = [], [], []
    for index, row in enumerate(corpus):
        documents.append({
            "dataset": "multihop_rag",
            "doc_id": f"multihop:{index:04d}",
            "title": str(row.get("title", "")),
            "author": str(row.get("author", "") or ""),
            "source": str(row.get("source", "")),
            "category": str(row.get("category", "")),
            "published_at": str(row.get("published_at", "")),
            "url": str(row.get("url", "")),
            "body": str(row.get("body", "")),
        })
    for index, row in enumerate(questions):
        if str(row.get("question_type", "")) == "null_query":
            continue
        query_id = f"multihop:{index:04d}"
        queries.append({
            "dataset": "multihop_rag",
            "query_id": query_id,
            "question": str(row["query"]),
        })
        labels.append({
            "query_id": query_id,
            "answer": str(row["answer"]),
            "question_type": str(row["question_type"]),
            "evidence_facts": [str(item["fact"]) for item in row.get("evidence_list", [])],
        })
    return documents, queries, labels


def prepare_m4_data(output_dir: Path = M4_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "m4-target-paper-v2",
        "chunking": {
            "tokenizer": TOKENIZER,
            "tokens": CHUNK_TOKENS,
            "overlap": CHUNK_OVERLAP,
            "boundary": "contiguous_token_windows",
        },
        "quality_protocol": "Open-ended generation without source answer options.",
        "multihop_protocol": "Only inference, comparison, and temporal queries; null queries excluded.",
        "oracle_policy": "Queries are method input; answers, difficulty, types, and evidence are evaluator-only.",
        "sources": {
            "quality": {"version": "v1.0.1", "split": "dev", "revision": "f84977c40dbfef70c9cab48037b7becfc8e45f73"},
            "multihop_rag": {"revision": "71ac0d0", "official_code_revision": "cde8e844af14b3012f20158abc2854fe8458212a"},
        },
        "files": {},
    }
    for dataset, prepared in (("quality", _prepare_quality()), ("multihop_rag", _prepare_multihop())):
        documents, queries, labels = prepared
        chunks = [chunk for document in documents for chunk in _chunks(document)]
        for kind, rows in (("documents", documents), ("chunks", chunks), ("queries", queries), ("labels", labels)):
            name = f"{dataset}_{kind}.jsonl"
            count, digest = _write_jsonl(output_dir / name, rows)
            manifest["files"][name] = {"count": count, "sha256": digest}
        manifest[dataset] = {
            "documents": len(documents),
            "chunks": len(chunks),
            "queries": len(queries),
            "types": dict(sorted(Counter(row.get("question_type", "all") for row in labels).items())),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
