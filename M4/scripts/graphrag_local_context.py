from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tiktoken

from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.vector_stores import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)


COMMUNITY_LEVEL = 2
MAX_CONTEXT_TOKENS = 8000

logging.getLogger("graphrag.query.context_builder.community_context").setLevel(logging.ERROR)


class QueryEmbedder(BaseTextEmbedding):
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, text: str, **_: Any) -> list[float]:
        return self.vectors[text]

    async def aembed(self, text: str, **_: Any) -> list[float]:
        return self.embed(text)


class EntityVectorStore(BaseVectorStore):
    def __init__(self, entities: list[Any]) -> None:
        super().__init__(collection_name="entity_description_embeddings")
        self.entities = entities
        self.include_ids: set[str] | None = None
        matrix = np.asarray([entity.description_embedding for entity in entities], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.maximum(norms, 1e-12)

    def connect(self, **_: Any) -> None:
        return None

    def load_documents(self, documents: list[VectorStoreDocument], overwrite: bool = True) -> None:
        del documents, overwrite

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **_: Any
    ) -> list[VectorStoreSearchResult]:
        query = np.asarray(query_embedding, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = self.matrix @ query
        order = np.argsort(-scores)
        results = []
        for position in order:
            entity = self.entities[int(position)]
            if self.include_ids is not None and entity.id not in self.include_ids:
                continue
            results.append(
                VectorStoreSearchResult(
                    document=VectorStoreDocument(id=entity.id, text=entity.description, vector=None),
                    score=float(scores[int(position)]),
                )
            )
            if len(results) == k:
                break
        return results

    def similarity_search_by_text(
        self, text: str, text_embedder: Any, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        return self.similarity_search_by_vector(text_embedder(text), k=k, **kwargs)

    def filter_by_id(self, include_ids: list[str] | list[int]) -> None:
        self.include_ids = {str(value) for value in include_ids}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_builder(index_dir: Path, embedder: QueryEmbedder) -> LocalSearchMixedContext | None:
    if (index_dir / "empty.json").exists():
        return None
    output = index_dir / "output"
    nodes = pd.read_parquet(output / "create_final_nodes.parquet")
    entities = read_indexer_entities(
        nodes, pd.read_parquet(output / "create_final_entities.parquet"), COMMUNITY_LEVEL
    )
    if not entities:
        return None
    reports = read_indexer_reports(
        pd.read_parquet(output / "create_final_community_reports.parquet"),
        nodes,
        COMMUNITY_LEVEL,
    )
    return LocalSearchMixedContext(
        entities=entities,
        entity_text_embeddings=EntityVectorStore(entities),
        text_embedder=embedder,
        text_units=read_indexer_text_units(
            pd.read_parquet(output / "create_final_text_units.parquet")
        ),
        community_reports=reports,
        relationships=read_indexer_relationships(
            pd.read_parquet(output / "create_final_relationships.parquet")
        ),
        token_encoder=tiktoken.get_encoding("o200k_base"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    queries = _read_jsonl(args.queries)
    vectors = np.load(args.vectors)
    if len(queries) != len(vectors):
        raise ValueError("Query manifest and vector count differ")
    vector_by_question = {
        row["question"]: np.asarray(vector, dtype=np.float32).tolist()
        for row, vector in zip(queries, vectors)
    }
    embedder = QueryEmbedder(vector_by_question)
    base = args.artifact_dir / args.dataset
    mapping = json.loads((base / "doc_to_index.json").read_text(encoding="utf-8"))
    builders: dict[str, LocalSearchMixedContext | None] = {}
    rows = []
    for query in queries:
        key = mapping[query["doc_id"]] if args.dataset == "quality" else "unified"
        if key not in builders:
            stem = hashlib.sha256(key.encode()).hexdigest()[:20]
            builders[key] = _load_builder(base / stem, embedder)
        builder = builders[key]
        retrieval = []
        if builder is not None:
            context, _ = builder.build_context(
                query=query["question"],
                max_tokens=MAX_CONTEXT_TOKENS,
                text_unit_prop=0.5,
                community_prop=0.1,
                top_k_mapped_entities=10,
                top_k_relationships=10,
                include_entity_rank=True,
                include_relationship_weight=True,
                include_community_rank=False,
                return_candidate_context=False,
            )
            if str(context).strip():
                retrieval.append(
                    {
                        "chunk_id": f"graphrag-local:{hashlib.sha256(key.encode()).hexdigest()[:20]}",
                        "doc_id": query.get("doc_id", ""),
                        "score": 1.0,
                        "policy": "official_graphrag_0.3.5_local_search",
                        "text": str(context),
                    }
                )
        rows.append({"query_id": query["query_id"], "retrieval": retrieval})
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
