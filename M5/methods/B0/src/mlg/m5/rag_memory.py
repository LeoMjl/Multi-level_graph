from __future__ import annotations

import base64
import math
import os
import time
import zlib
from array import array
from dataclasses import dataclass
from typing import Any, Protocol

from mlg.m5.backend import GenerationResult, TextBackend
from mlg.m5.dataset import ChapterPrompt
from mlg.m5.memory import MemoryContext, MemoryStrategy, count_tokens


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    call: GenerationResult


class EmbeddingBackend(Protocol):
    model: str

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingBatch: ...


class OpenAIEmbeddingBackend:
    def __init__(self, *, model: str = "text-embedding-3-small", api_key_env: str = "OPENAI_API_KEY") -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"M5 vector baseline requires {api_key_env}")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, max_retries=5, timeout=300.0)
        self.model = model

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingBatch:
        started = time.perf_counter()
        response = self.client.embeddings.create(model=self.model, input=texts)
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        call = GenerationResult(
            text="", model=self.model, elapsed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=int(tokens) if tokens is not None else None,
            total_tokens=int(tokens) if tokens is not None else None, purpose=purpose,
        )
        return EmbeddingBatch([list(item.embedding) for item in response.data], call)


class LocalSentenceTransformerEmbeddingBackend:
    """Offline embedding backend for collaboration runs; never calls an API."""

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model
        self._encoder = SentenceTransformer(
            model,
            device=device,
            trust_remote_code=True,
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
        return EmbeddingBatch(
            [row.astype("float32", copy=False).tolist() for row in vectors],
            GenerationResult(
                text="", model=self.model,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                input_tokens=sum(len(text) for text in texts), purpose=purpose,
            ),
        )


class FakeEmbeddingBackend:
    model = "fake-hashed-bigram-embedding"

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingBatch:
        started = time.perf_counter()
        vectors = [_hashed_bigrams(text) for text in texts]
        return EmbeddingBatch(vectors, GenerationResult(
            text="", model=self.model, elapsed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=sum(len(text) for text in texts), purpose=purpose,
        ))


class FlatVectorRAGMemory(MemoryStrategy):
    name = "flat_vector_rag"

    def __init__(self, embeddings: EmbeddingBackend, *, token_budget: int = 12000, top_k: int = 8) -> None:
        super().__init__(token_budget=token_budget)
        self.embeddings = embeddings
        self.top_k = top_k
        self.items: list[dict[str, Any]] = []
        self.vectors: list[list[float]] = []

    def context_for(self, prompt: ChapterPrompt) -> MemoryContext:
        if not self.items:
            return MemoryContext(metadata={"top_k": self.top_k, "api_calls": []})
        query = _query_text(prompt)
        batch = self.embeddings.embed([query], purpose="memory_query")
        scored = [(_cosine(batch.vectors[0], vector), index) for index, vector in enumerate(self.vectors)]
        scored.sort(reverse=True)
        blocks: list[str] = []
        selected: list[int] = []
        used = 0
        rows: list[dict[str, Any]] = []
        for score, index in scored[:self.top_k]:
            item = self.items[index]
            block = f"[历史第{item['chapter_id']}章片段{item['chunk_id']}]\n{item['text']}"
            tokens = count_tokens(block)
            if used + tokens > self.token_budget:
                continue
            blocks.append(block)
            used += tokens
            selected.append(int(item["chapter_id"]))
            rows.append({"chapter_id": item["chapter_id"], "chunk_id": item["chunk_id"], "score": score})
        return MemoryContext("\n\n".join(blocks), sorted(set(selected)), {
            "top_k": self.top_k, "retrieval": rows, "api_calls": [batch.call.to_dict()],
        })

    def observe(self, prompt: ChapterPrompt, text: str, backend: TextBackend) -> list[GenerationResult]:
        chunks = _chunks(text, 900, 120)
        batch = self.embeddings.embed(chunks, purpose="memory_build")
        start = len(self.items)
        self.items.extend({"chapter_id": prompt.chapter_id, "chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks))
        self.vectors.extend(batch.vectors)
        if len(self.items) != start + len(batch.vectors):
            raise RuntimeError("Embedding backend returned the wrong vector count")
        return [batch.call]

    def to_state(self) -> dict[str, Any]:
        return {
            **super().to_state(), "top_k": self.top_k, "items": self.items,
            "vectors": _pack_vectors(self.vectors), "embedding_model": self.embeddings.model,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self.top_k = int(state.get("top_k", self.top_k))
        self.items = list(state.get("items", []))
        self.vectors = _unpack_vectors(state.get("vectors", {}))


def _query_text(prompt: ChapterPrompt) -> str:
    return "\n".join([prompt.title, prompt.chapter_goal, *prompt.must_include, *prompt.must_avoid])


def _chunks(text: str, size: int, overlap: int) -> list[str]:
    import tiktoken
    encoder = tiktoken.get_encoding("o200k_base")
    tokens = encoder.encode(text)
    return [encoder.decode(tokens[start:start + size]) for start in range(0, len(tokens), size - overlap)] or [text]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    denom = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / denom if denom else 0.0


def _hashed_bigrams(text: str, dims: int = 128) -> list[float]:
    values = [0.0] * dims
    for index in range(max(0, len(text) - 1)):
        token = text[index:index + 2].encode("utf-8")
        values[zlib.crc32(token) % dims] += 1.0
    return values


def _pack_vectors(vectors: list[list[float]]) -> dict[str, Any]:
    if not vectors:
        return {"rows": 0, "cols": 0, "data": ""}
    flat = array("f", (value for row in vectors for value in row))
    data = base64.b64encode(zlib.compress(flat.tobytes(), 6)).decode("ascii")
    return {"rows": len(vectors), "cols": len(vectors[0]), "data": data}


def _unpack_vectors(payload: dict[str, Any]) -> list[list[float]]:
    rows, cols = int(payload.get("rows", 0)), int(payload.get("cols", 0))
    if not rows or not cols:
        return []
    flat = array("f")
    flat.frombytes(zlib.decompress(base64.b64decode(payload["data"])))
    return [list(flat[start:start + cols]) for start in range(0, rows * cols, cols)]
