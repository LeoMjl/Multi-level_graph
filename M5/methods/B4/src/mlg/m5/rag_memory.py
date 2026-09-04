from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mlg.m5.backend import GenerationResult, TextBackend
from mlg.m5.dataset import ChapterPrompt
from mlg.m5.memory import MemoryContext, MemoryStrategy, count_tokens
from mlg.m5.rag_codec import (
    directory_sha256, hashed_bigrams, pack_vectors, unpack_vectors,
)

_pack_vectors, _unpack_vectors = pack_vectors, unpack_vectors


EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
CHUNK_TOKENS = 900
CHUNK_OVERLAP_TOKENS = 120
VECTOR_TOP_K = 8

@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    call: GenerationResult

class EmbeddingBackend(Protocol):
    model: str
    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingBatch: ...

class LocalSentenceTransformerEmbeddingBackend:
    """Pinned, offline embedding backend for API-free formal runs."""

    def __init__(self, *, device: str = "cpu") -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        artifact = snapshot_download(
            EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
            local_files_only=True,
        )
        self.model = (
            f"{EMBEDDING_MODEL}@{EMBEDDING_REVISION}"
            f"#{directory_sha256(Path(artifact))}"
        )
        self._encoder = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device=device,
            trust_remote_code=False,
            local_files_only=True,
        )

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingBatch:
        started = time.perf_counter()
        vectors = self._encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        rows = [row.astype("float32", copy=False).tolist() for row in vectors]
        _validate_vectors(rows, len(texts))
        tokens = sum(count_tokens(text) for text in texts)
        return EmbeddingBatch(rows, GenerationResult(
            text="", model=self.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=tokens, total_tokens=tokens, purpose=purpose,
        ))


class FakeEmbeddingBackend:
    model = "fake-hashed-bigram-embedding-v1"
    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingBatch:
        started = time.perf_counter()
        rows = [hashed_bigrams(text) for text in texts]
        return EmbeddingBatch(rows, GenerationResult(
            text="", model=self.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=sum(count_tokens(text) for text in texts), purpose=purpose,
        ))


class FlatVectorRAGMemory(MemoryStrategy):
    name = "flat_vector_rag"
    def __init__(
        self, embeddings: EmbeddingBackend, *, token_budget: int = 12000,
        top_k: int = VECTOR_TOP_K,
    ) -> None:
        super().__init__(token_budget=token_budget)
        self.embeddings = embeddings
        self.top_k = top_k
        self.items: list[dict[str, Any]] = []
        self.vectors: list[list[float]] = []

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        config = self.protocol_config()
        if not self.items:
            return MemoryContext(metadata={**config, "api_calls": [],
                                           "ranked_candidates": []})
        query = _query_text(prompt)
        batch = self.embeddings.embed([query], purpose="memory_query")
        if self.vectors and len(batch.vectors[0]) != len(self.vectors[0]):
            raise RuntimeError("Query embedding dimension changed during the run")
        scored = [(_cosine(batch.vectors[0], vector), index)
                  for index, vector in enumerate(self.vectors)]
        scored.sort(key=lambda row: (-row[0], row[1]))
        blocks: list[str] = []
        selected: list[int] = []
        used = 0
        selected_indices: set[int] = set()
        for score, index in scored[:self.top_k]:
            item = self.items[index]
            block = f"[历史第{item['chapter_id']}章片段{item['chunk_id']}]\n{item['text']}"
            tokens = count_tokens(block)
            if used + tokens > self.token_budget:
                continue
            blocks.append(block)
            used += tokens
            selected.append(int(item["chapter_id"]))
            selected_indices.add(index)
        candidates = [{
            "chapter_id": self.items[index]["chapter_id"],
            "chunk_id": self.items[index]["chunk_id"],
            "score": score,
            "selected": index in selected_indices,
        } for score, index in scored]
        return MemoryContext("\n\n".join(blocks), sorted(set(selected)), {
            **config, "memory_tokens": used,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "ranked_candidates": candidates,
            "api_calls": [batch.call.to_dict()],
        })

    def observe(
        self, prompt: ChapterPrompt, text: str, backend: TextBackend,
    ) -> list[GenerationResult]:
        chunks = _chunks(text, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
        batch = self.embeddings.embed(chunks, purpose="memory_build")
        _validate_vectors(batch.vectors, len(chunks))
        self.items.extend({"chapter_id": prompt.chapter_id, "chunk_id": index,
                           "text": chunk} for index, chunk in enumerate(chunks))
        self.vectors.extend(batch.vectors)
        return [batch.call]

    def protocol_config(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embeddings.model,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "top_k": self.top_k,
            "similarity": "cosine",
            "tie_break": "ascending_chunk_index",
        }

    def to_state(self) -> dict[str, Any]:
        return {**super().to_state(), "config": self.protocol_config(),
                "items": self.items, "vectors": pack_vectors(self.vectors)}

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        if state.get("config") != self.protocol_config():
            raise ValueError("Flat-vector RAG configuration changed during resume")
        self.items = list(state.get("items", []))
        self.vectors = unpack_vectors(state.get("vectors", {}))
        _validate_vectors(self.vectors, len(self.items))

def _query_text(prompt: ChapterPrompt) -> str:
    return "\n".join([prompt.title, prompt.chapter_goal,
                      *prompt.must_include, *prompt.must_avoid])


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    import tiktoken
    tokens = tiktoken.get_encoding("o200k_base").encode(text)
    step = size - overlap
    return [tiktoken.get_encoding("o200k_base").decode(tokens[start:start + size])
            for start in range(0, len(tokens), step)] or [text]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    denom = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / denom if denom else 0.0


def _validate_vectors(vectors: list[list[float]], expected: int) -> None:
    if len(vectors) != expected or (vectors and not vectors[0]):
        raise RuntimeError("Embedding backend returned an invalid vector count")
    dims = len(vectors[0]) if vectors else 0
    if any(len(row) != dims or any(not math.isfinite(value) for value in row)
           for row in vectors):
        raise RuntimeError("Embedding backend returned invalid vector values")
