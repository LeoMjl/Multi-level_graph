from __future__ import annotations

import os
import time

from mlg.m5.backend import GenerationResult
from mlg.m5.openrouter_embedding import _ordered_finite_vectors
from mlg.m5.rag_memory import EmbeddingBatch


class DashScopeEmbeddingBackend:
    """Alibaba Cloud Model Studio text embeddings with a fixed dense dimension."""

    provider = "dashscope_openai_compatible_api"

    def __init__(
        self,
        *,
        model: str = "qwen3.7-text-embedding",
        api_key_env: str = "DASHSCOPE_API_KEY",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions: int = 2048,
        max_input_chars: int = 1600,
    ) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"DashScope embeddings require {api_key_env}")
        if dimensions <= 0:
            raise ValueError("DashScope embedding dimensions must be positive")
        from openai import OpenAI

        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.max_input_chars = max_input_chars
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            # A formal audit row must correspond to exactly one provider request.
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
            dimensions=self.dimensions,
            encoding_format="float",
        )
        data = list(getattr(response, "data", None) or [])
        vectors = _ordered_finite_vectors(
            data, expected_count=len(texts), provider="DashScope",
        )
        if any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(
                "DashScope returned a vector dimension different from the configured value"
            )
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
