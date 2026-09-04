from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from graphrag.config import create_graphrag_config
from graphrag.index.api import build_index
from graphrag.llm.openai.openai_chat_llm import OpenAIChatLLM
from graphrag.llm.openai.openai_embeddings_llm import OpenAIEmbeddingsLLM
from graphrag.llm.openai.utils import get_completion_llm_args


REQUIRED = [
    "create_final_nodes.parquet",
    "create_final_entities.parquet",
    "create_final_community_reports.parquet",
    "create_final_text_units.parquet",
    "create_final_relationships.parquet",
]
INDEX_PROTOCOL = "m4-graphrag-v3-no-gleaning"
DOCUMENT_FIELDS = ["doc_id", "title", "author", "source", "category", "published_at", "body"]


async def _float_embeddings(self, input, **kwargs):
    args = {"model": self.configuration.model, **(kwargs.get("model_parameters") or {})}
    response = await self.client.embeddings.create(input=input, encoding_format="float", **args)
    return [item.embedding for item in response.data]


async def _thinking_disabled_chat(self, input, **kwargs):
    args = get_completion_llm_args(kwargs.get("model_parameters"), self.configuration)
    history = kwargs.get("history") or []
    messages = [*history, {"role": "user", "content": input}]
    completion = await self.client.chat.completions.create(
        messages=messages, extra_body={"thinking": {"type": "disabled"}}, **args
    )
    return completion.choices[0].message.content


OpenAIChatLLM._execute_llm = _thinking_disabled_chat
OpenAIEmbeddingsLLM._execute_llm = _float_embeddings

_report_module = importlib.import_module("graphrag.index.verbs.graph.report.create_community_reports")
_original_get_levels = _report_module.get_levels
_active_dataset = ""


def _target_query_levels(nodes):
    levels = _original_get_levels(nodes)
    return [level for level in levels if level <= 2] if _active_dataset == "multihop_rag" else levels


_report_module.get_levels = _target_query_levels


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def graph_config(root: Path, model_config: Path, chunk_size: int):
    raw = json.loads(model_config.read_text(encoding="utf-8-sig"))["profiles"]["deepseek_v4_flash"]
    chat_key = os.getenv(raw["api_key_env"], "")
    embed_key = os.getenv(raw["embedding_api_key_env"], "")
    if not chat_key or not embed_key:
        raise RuntimeError("DeepSeek or embedding API key is missing")
    values = {
        "encoding_model": "o200k_base",
        "llm": {
            "api_key": chat_key, "type": "openai_chat", "model": raw["model"],
            "api_base": raw["base_url"], "model_supports_json": True,
            "temperature": 0.0, "request_timeout": 300.0, "max_retries": 5,
            "concurrent_requests": 8,
        },
        "parallelization": {"num_threads": 8, "stagger": 0.1},
        "async_mode": "threaded",
        "embeddings": {
            "batch_size": 32, "batch_max_tokens": 8192,
            "llm": {
                "api_key": embed_key, "type": "openai_embedding", "model": raw["embedding_model"],
                "api_base": raw["embedding_base_url"], "request_timeout": 300.0,
                "max_retries": 5, "concurrent_requests": 4,
            },
        },
        "chunks": {"size": chunk_size, "overlap": 0, "group_by_columns": ["id"]},
        "input": {
            "type": "file", "file_type": "text", "base_dir": "input",
            "file_encoding": "utf-8", "file_pattern": ".*\\.txt$",
        },
        "cache": {"type": "file", "base_dir": "cache"},
        "storage": {"type": "file", "base_dir": "output"},
        "reporting": {"type": "file", "base_dir": "logs"},
        "entity_extraction": {"max_gleanings": 0},
        "snapshots": {"graphml": False, "raw_entities": False, "top_level_nodes": False},
    }
    config = create_graphrag_config(values, root_dir=str(root))
    # GraphRAG 0.3.5's loader treats numeric zero as missing and restores its
    # default overlap. Correct the resolved object to match the paper protocol.
    config.chunks.overlap = 0
    config.entity_extraction.max_gleanings = 0
    return config


def write_inputs(root: Path, documents: list[dict]) -> None:
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for row in documents:
        name = hashlib.sha256(row["doc_id"].encode()).hexdigest()[:20] + ".txt"
        header = "\n".join(
            f"{field}: {row[field]}" for field in DOCUMENT_FIELDS[1:-1] if row.get(field)
        )
        text = f"{header}\n\n{row.get('body', '')}" if header else str(row.get("body", ""))
        (input_dir / name).write_text(text, encoding="utf-8")


def _input_digest(documents: list[dict]) -> str:
    payload = [
        {field: row.get(field, "") for field in DOCUMENT_FIELDS}
        for row in sorted(documents, key=lambda item: item["doc_id"])
    ]
    raw = json.dumps({"protocol": INDEX_PROTOCOL, "documents": payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def terminal(root: Path, digest: str) -> bool:
    for name in ("complete.json", "empty.json"):
        marker = root / name
        if marker.exists():
            row = json.loads(marker.read_text(encoding="utf-8"))
            return row.get("input_sha256") == digest and row.get("protocol") == INDEX_PROTOCOL
    return False


def outputs_complete(root: Path) -> bool:
    return all((root / "output" / name).exists() for name in REQUIRED)


def unclusterable_entity_graph(root: Path) -> bool:
    path = root / "output" / "create_base_extracted_entities.parquet"
    if not path.exists():
        return False
    frame = pd.read_parquet(path)
    graphs = [nx.parse_graphml(str(value)) for value in frame["entity_graph"]]
    return bool(graphs) and all(len(graph.nodes) == 0 or len(graph.edges) == 0 for graph in graphs)


async def build_one(root: Path, documents: list[dict], config_path: Path, chunk_size: int) -> dict:
    digest = _input_digest(documents)
    if terminal(root, digest):
        return {"status": "reused", "documents": len(documents)}
    root.mkdir(parents=True, exist_ok=True)
    write_inputs(root, documents)
    config = graph_config(root, config_path, chunk_size)
    results = await build_index(config)
    errors = [str(error) for result in results for error in (result.errors or [])]
    if errors and unclusterable_entity_graph(root):
        marker = {
            "status": "empty", "documents": len(documents), "reason": "official GraphRAG graph has zero nodes or zero edges",
            "protocol": INDEX_PROTOCOL, "input_sha256": digest,
        }
        (root / "empty.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        return marker
    if errors or not outputs_complete(root):
        raise RuntimeError(f"GraphRAG indexing failed: {errors[:3]}")
    marker = {
        "status": "complete", "documents": len(documents), "workflows": len(results),
        "protocol": INDEX_PROTOCOL, "input_sha256": digest,
    }
    (root / "complete.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


async def run(args: argparse.Namespace) -> None:
    global _active_dataset
    _active_dataset = args.dataset
    documents = read_jsonl(args.data_dir / f"{args.dataset}_documents.jsonl")
    if args.limit_documents:
        documents = documents[: args.limit_documents]
    groups: dict[str, list[dict]] = defaultdict(list)
    doc_to_key = {}
    for row in documents:
        key = str(row.get("source_article_id", row["doc_id"])) if args.dataset == "quality" else "unified"
        doc_to_key[row["doc_id"]] = key
        if not groups[key]:
            groups[key].append(row)
        elif args.dataset != "quality":
            groups[key].append(row)
    (args.output / args.dataset).mkdir(parents=True, exist_ok=True)
    mapping = args.output / args.dataset / "doc_to_index.json"
    mapping.write_text(json.dumps(doc_to_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    keys = sorted(groups)
    if args.index_key:
        if args.index_key not in groups:
            raise ValueError(f"Unknown index key: {args.index_key}")
        keys = [args.index_key]
    if args.limit_indexes:
        keys = keys[:args.limit_indexes]
    semaphore = asyncio.Semaphore(args.index_workers)
    finished = 0

    async def build_key(key: str):
        async with semaphore:
            root = args.output / args.dataset / hashlib.sha256(key.encode()).hexdigest()[:20]
            return await build_one(root, groups[key], args.model_config, 512 if args.dataset == "quality" else 1024)

    tasks = [asyncio.create_task(build_key(key)) for key in keys]
    for future in asyncio.as_completed(tasks):
        status = await future
        finished += 1
        print(f"[M4] GraphRAG {args.dataset}: indexes {finished}/{len(keys)} ({status['status']})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["quality", "multihop_rag"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--limit-indexes", type=int, default=0)
    parser.add_argument("--limit-documents", type=int, default=0)
    parser.add_argument("--index-key")
    parser.add_argument("--index-workers", type=int, default=5)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
