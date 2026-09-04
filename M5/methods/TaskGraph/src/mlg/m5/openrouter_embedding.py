from __future__ import annotations

import math
import os
import time
from typing import Any, Sequence

from mlg.m5.backend import GenerationResult
from mlg.m5.rag_memory import EmbeddingBatch


class OpenRouterEmbeddingBackend:
    """OpenRouter embeddings-only backend with no provider/model fallback."""

    provider = "openrouter_embeddings_api"

    def __init__(
        self,
        *,
        model: str = "nvidia/nemotron-3-embed-1b:free",
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        max_input_chars: int = 1600,
    ) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"OpenRouter embeddings require {api_key_env}")
        from openai import OpenAI

        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.dimensions = None
        self.max_input_chars = max_input_chars
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            # Formal runs must not hide provider retries behind one audit row.
            max_retries=0,
            timeout=300.0,
        )

    def embed(self, texts: list[str], *, purpose: str) -> EmbeddingBatch:
        if any(len(text) > self.max_input_chars for text in texts):
            raise RuntimeError(
                "Embedding input exceeds the node-text boundary; full chapters are forbidden"
            )
        started = time.perf_counter()
        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        data = list(getattr(response, "data", None) or [])
        vectors = _ordered_finite_vectors(data, expected_count=len(texts))
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "total_tokens", None)
        return EmbeddingBatch(vectors, GenerationResult(
            text="",
            model=self.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            total_tokens=int(input_tokens) if input_tokens is not None else None,
            response_id=str(getattr(response, "id", "") or ""),
            purpose=purpose,
        ))


def _ordered_finite_vectors(
    data: Sequence[Any], *, expected_count: int, provider: str = "OpenRouter",
) -> list[list[float]]:
    """Validate an embeddings response and restore request order by item.index."""
    if len(data) != expected_count:
        raise RuntimeError(
            f"{provider} returned an incomplete embedding index set: "
            f"expected {expected_count}, received {len(data)}"
        )

    by_index: dict[int, list[float]] = {}
    dimension: int | None = None
    for item in data:
        index = getattr(item, "index", None)
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError(
                f"{provider} embedding item has a missing or invalid index"
            )
        if not 0 <= index < expected_count:
            raise RuntimeError(f"{provider} embedding index is out of range: {index}")
        if index in by_index:
            raise RuntimeError(f"{provider} returned duplicate embedding index: {index}")

        raw_vector = getattr(item, "embedding", None)
        try:
            vector = [float(value) for value in raw_vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"{provider} embedding vector at index {index} is invalid"
            ) from exc
        if not vector:
            raise RuntimeError(f"{provider} embedding vector at index {index} is empty")
        if any(not math.isfinite(value) for value in vector):
            raise RuntimeError(
                f"{provider} embedding vector at index {index} contains NaN or Inf"
            )
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise RuntimeError(f"{provider} returned inconsistent embedding dimensions")
        by_index[index] = vector

    missing = [index for index in range(expected_count) if index not in by_index]
    if missing:
        raise RuntimeError(f"{provider} omitted embedding indices: {missing}")
    return [by_index[index] for index in range(expected_count)]
