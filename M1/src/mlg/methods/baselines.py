from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from mlg.methods.base import (
    EMBEDDING_INPUT_MAX_CHARS,
    Method,
    formatted_history_items,
    format_history,
    keyword_answer,
    pack_text_items,
    parse_prediction_json,
    trim_text,
)
from mlg.methods.longmem_reader import longmem_reader_schema
from mlg.methods.memory_units import MemoryUnit, atomic_units_from_history, raw_chunks_from_history
from mlg.methods.vector_cache import (
    build_vector_cache_spec,
    fake_vector_cache_metadata,
    load_vector_cache,
    rank_vectors,
    vector_count,
)


RETRIEVAL_CONTEXT_MAX_TOKENS = 32_000


def _normalized_excerpt(text: str) -> str:
    return " ".join(str(text).lower().split()).strip()


def enforce_longmem_evidence_grounding(
    payload: dict,
    answer: str,
    evidence: list[str],
    source_texts: list[str],
) -> tuple[str, list[str], dict]:
    """Drop evidence excerpts that cannot be found in retrieved source memory.

    This check is answer- and question-type agnostic: it prevents invented
    citations without changing answers through example-specific rules.
    """
    if "longmemeval-v2" not in str(payload.get("dataset", "")).lower():
        return answer, evidence, {}

    normalized_sources = [_normalized_excerpt(text) for text in source_texts if str(text).strip()]
    grounded = []
    rejected = []
    for item in evidence:
        normalized = _normalized_excerpt(item)
        if normalized and any(normalized in source for source in normalized_sources):
            grounded.append(str(item))
        else:
            rejected.append(str(item))

    metadata: dict = {
        "evidence_grounding": {
            "reported_count": len(evidence),
            "grounded_count": len(grounded),
            "rejected_count": len(rejected),
        }
    }
    return answer, grounded, metadata


def response_contract(payload: dict) -> str:
    dataset = str(payload.get("dataset", "")).lower()
    if "longmemeval-v2" in dataset:
        return (
            longmem_reader_schema(payload) + " "
            "First decompose the question into entities, properties, relations, scope, quantifiers, and "
            "any temporal or list-order constraint. Verify every requested predicate against evidence "
            "from the same entity and scope; a name match alone does not prove another property. For "
            "before/after/between/next questions, follow recorded state and action order. For ordered-list "
            "questions, preserve source order. A question may contain a false premise: reject it only when "
            "the retrieved evidence covers the requested scope; do not treat failure to retrieve an item "
            "as proof that it does not exist. Do not guess when evidence is incomplete. "
            "Do not put analysis or another JSON object inside answer."
        )
    if "toolbench" in dataset:
        return (
            "Return JSON only with this schema: "
            "{\"answer\":\"tool call summary\", \"tool_name\":\"...\", \"api_name\":\"...\", "
            "\"arguments\":{}, \"evidence\":[\"tool/request snippets\"], \"stage\":\"tool_execution\", "
            "\"dependencies\":[{\"source\":\"request/tool argument used\", \"target\":\"tool_call\"}]}."
        )
    return (
        "Return JSON only: {\"answer\":\"...\", \"evidence\":[\"...\"], "
        "\"stage\":\"...\", \"dependencies\":[{\"source\":\"supporting fact used\", \"target\":\"answer\"}]}."
    )


class TruncatedContextMethod(Method):
    """Head-and-tail context baseline with an explicit 48K character budget."""

    name = "truncated_context"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._context_cache: dict[str, str] = {}
        self._context_build_count = 0

    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        cache_key = self.shared_haystack_key(payload)
        cache_hit = cache_key in self._context_cache
        if not cache_hit:
            with self.measure_phase("memory_build_time_ms"):
                self._context_cache[cache_key] = format_history(payload, max_chars=48000)
                self._context_build_count += 1
        context = self._context_cache[cache_key]
        texts = [context]
        cache_meta = {
            "haystack_cache_hit": cache_hit,
            "haystack_key": cache_key,
            "haystack_build_count": self._context_build_count,
            "haystack_build_ms": 0.0 if cache_hit else self._phase_timings["memory_build_time_ms"],
        }
        if self.can_call_llm():
            system = (
                "You are the Truncated Context baseline for a long-context benchmark. "
                "Use only the provided 48K-character head-and-tail history view. Do not use graph memory, "
                "retrieval indexes, or hidden labels. Return JSON with answer, evidence, stage, dependencies."
            )
            user = (
                f"Dataset: {payload['dataset']}\n"
                f"Question/task:\n{payload['query']}\n\n"
                f"Truncated chronological history:\n{context}\n\n"
                + response_contract(payload)
            )
            with self.measure_phase("reader_generation_time_ms"):
                raw = self.chat_json(system, user)
                answer, evidence, stage, dependencies, metadata = parse_prediction_json(raw)
            answer, evidence, grounding_meta = enforce_longmem_evidence_grounding(
                payload, answer, evidence, [context]
            )
            metadata.update(grounding_meta)
            metadata.update(cache_meta)
            metadata["context_policy"] = "head_tail_48000_llm"
            if not answer:
                if self.fallback_allowed():
                    fallback_answer, fallback_evidence = keyword_answer(payload["query"], texts)
                    metadata["fallback"] = "keyword_after_llm_error_or_empty"
                    return fallback_answer, fallback_evidence, infer_stage(payload["query"], texts), [], metadata
                metadata.update({
                    "status": "failed",
                    "failure_stage": "reader_generation",
                    "failure_reason": str(raw.get("_error", "reader returned an empty answer")),
                })
                return "", [], "", [], metadata
            return answer, evidence, stage, dependencies, metadata
        if not self.fake_llm:
            return "", [], "", [], {
                **cache_meta,
                "context_policy": "head_tail_48000_llm",
                "status": "failed",
                "failure_stage": "reader_generation",
                "failure_reason": f"LLM API is not configured ({self.runtime.api_key_env})",
            }
        with self.measure_phase("reader_generation_time_ms"):
            answer, evidence = keyword_answer(payload["query"], texts)
        return answer, evidence, infer_stage(payload["query"], texts), [], {
            "context_policy": "head_tail_48000",
            **cache_meta,
        }


class BoundedContextMethod(Method):
    """Query-independent prefix window bounded by the configured reader context."""

    name = "bounded_context"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._context_cache: dict[str, tuple[str, dict]] = {}
        self._context_build_count = 0
        self._last_long_context_request_started = 0.0

    def _chat_with_rate_limit(self, system: str, user: str) -> tuple[dict, dict]:
        total_wait_seconds = 0.0
        attempts = 0
        max_attempts = max(1, int(self.runtime.long_context_rate_limit_retries) + 1)
        while attempts < max_attempts:
            interval = max(0.0, float(self.runtime.long_context_min_start_interval_seconds))
            elapsed = time.monotonic() - self._last_long_context_request_started
            wait_seconds = max(0.0, interval - elapsed) if self._last_long_context_request_started else 0.0
            if wait_seconds:
                time.sleep(wait_seconds)
                total_wait_seconds += wait_seconds
            self._last_long_context_request_started = time.monotonic()
            attempts += 1
            raw = self.chat_json(system, user)
            error = str(raw.get("_error", "")).lower()
            retryable = "429" in error and any(
                marker in error for marker in ("token-limit", "insufficient_quota", "rate limit")
            )
            if not retryable or attempts >= max_attempts:
                return raw, {
                    "minimum_start_interval_seconds": interval,
                    "wait_time_seconds": total_wait_seconds,
                    "attempt_count": attempts,
                    "rate_limit_retry_count": attempts - 1,
                }
        raise AssertionError("unreachable long-context retry loop")

    def _history_budget(self) -> int:
        # Reserve 25% for tokenizer mismatch, chat-template tokens, system/query
        # text, and completion. Qwen does not expose its server tokenizer through
        # the compatible API; a 10% reserve was rejected by the real 1M endpoint
        # for accessibility-tree text even when o200k estimated 900K tokens.
        # The half-window cap keeps tiny test configurations meaningful as well.
        context_window = max(int(self.runtime.context_window_tokens), 1)
        reserve = max(32_000, context_window // 4, int(self.runtime.max_tokens))
        reserve = min(reserve, max(1, context_window // 2))
        return max(1, context_window - reserve)

    def _shared_context(self, payload: dict) -> tuple[str, dict]:
        cache_key = self.shared_haystack_key(payload)
        cache_hit = cache_key in self._context_cache
        if not cache_hit:
            with self.measure_phase("memory_build_time_ms"):
                budget = self._history_budget()
                context, packing = pack_text_items(
                    formatted_history_items(payload),
                    max_tokens=budget,
                )
                packing.update({
                    "configured_context_window_tokens": int(self.runtime.context_window_tokens),
                    "history_token_budget": budget,
                    "estimator_safety_reserve_fraction": 0.25,
                    "selection_policy": "chronological_prefix_complete_items_then_partial_final_item",
                })
                self._context_cache[cache_key] = (context, packing)
                self._context_build_count += 1
        context, packing = self._context_cache[cache_key]
        return context, {
            "haystack_cache_hit": cache_hit,
            "haystack_key": cache_key,
            "haystack_build_count": self._context_build_count,
            "haystack_build_ms": 0.0 if cache_hit else self._phase_timings["memory_build_time_ms"],
            "context_window": dict(packing),
            "phase_applicability": {"memory_build": True, "memory_query": False, "reader_generation": True},
        }

    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        context, cache_meta = self._shared_context(payload)
        if self.can_call_llm():
            system = (
                "You are the Bounded Context baseline for LongMemEval-V2. Use only the fixed, "
                "query-independent chronological prefix that fits the configured reader window. "
                "Do not use retrieval, graph memory, or hidden labels."
            )
            user = (
                f"Dataset: {payload['dataset']}\n"
                f"Question/task:\n{payload['query']}\n\n"
                f"Fixed bounded history window:\n{context}\n\n"
                + response_contract(payload)
            )
            with self.measure_phase("reader_generation_time_ms"):
                raw, rate_limit_meta = self._chat_with_rate_limit(system, user)
                answer, evidence, stage, dependencies, metadata = parse_prediction_json(raw)
            metadata["long_context_rate_limit"] = rate_limit_meta
            answer, evidence, grounding_meta = enforce_longmem_evidence_grounding(
                payload, answer, evidence, [context]
            )
            metadata.update(grounding_meta)
            metadata.update(cache_meta)
            metadata["context_policy"] = "bounded_model_window_prefix_llm"
            if answer:
                return answer, evidence, stage, dependencies, metadata
            if self.fallback_allowed():
                fallback_answer, fallback_evidence = keyword_answer(payload["query"], [context])
                metadata["fallback"] = "keyword_after_llm_error_or_empty"
                return fallback_answer, fallback_evidence, infer_stage(payload["query"], [context]), [], metadata
            metadata.update({
                "status": "failed",
                "failure_stage": "reader_generation",
                "failure_reason": str(raw.get("_error", "reader returned an empty answer")),
            })
            return "", [], "", [], metadata
        if not self.fake_llm:
            return "", [], "", [], {
                **cache_meta,
                "context_policy": "bounded_model_window_prefix_llm",
                "status": "failed",
                "failure_stage": "reader_generation",
                "failure_reason": f"LLM API is not configured ({self.runtime.api_key_env})",
            }
        with self.measure_phase("reader_generation_time_ms"):
            answer, evidence = keyword_answer(payload["query"], [context])
        return answer, evidence, infer_stage(payload["query"], [context]), [], {
            **cache_meta,
            "context_policy": "bounded_model_window_prefix",
        }


@dataclass
class SharedRagHaystack:
    units: list[MemoryUnit]
    search_views: list[str]
    embeddings: object | None = None
    build_error: str = ""
    vector_cache_metadata: dict = field(default_factory=dict)


class EmbeddingRagMethod(Method):
    """Reusable dense-index RAG base with fail-closed retrieval."""

    retrieval_name = "embedding_rag"
    context_policy = "embedding_rag_llm"
    reader_label = "Embedding RAG"
    retrieval_k = 8
    vector_index_kind = "embedding_rag_v1"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._haystack_cache: dict[str, SharedRagHaystack] = {}
        self._haystack_build_count = 0

    def make_units(self, payload: dict) -> list[MemoryUnit]:
        raise NotImplementedError

    def embedding_query_text(self, query: str) -> str:
        return query

    def _shared_haystack(self, payload: dict) -> tuple[SharedRagHaystack, dict]:
        cache_key = self.shared_haystack_key(payload)
        cache_hit = cache_key in self._haystack_cache
        if not cache_hit:
            with self.measure_phase("memory_build_time_ms"):
                units = self.make_units(payload)
                bundle = SharedRagHaystack(
                    units=units,
                    search_views=[unit.embedding_text.lower() for unit in units],
                )
                texts = [unit.embedding_text for unit in units]
                if self.fake_llm:
                    bundle.vector_cache_metadata = fake_vector_cache_metadata()
                else:
                    cache_spec = build_vector_cache_spec(
                        self.runtime,
                        index_kind=self.vector_index_kind,
                        haystack_key=cache_key,
                        unit_ids=[unit.unit_id for unit in units],
                        texts=texts,
                    )
                    cached_vectors, cache_metadata = load_vector_cache(cache_spec)
                    bundle.embeddings = cached_vectors
                    bundle.vector_cache_metadata = cache_metadata
                if bundle.embeddings is None and self.can_call_embedding() and units:
                    bundle.embeddings, bundle.vector_cache_metadata = self.embed_texts_resumable(
                        texts,
                        cache_spec=cache_spec,
                        prior_metadata=bundle.vector_cache_metadata,
                        phase="memory_build",
                        progress_label=f"{self.name}:{cache_key}",
                    )
                    if bundle.embeddings is None:
                        bundle.build_error = self._last_embedding_error or "embedding index build returned incomplete vectors"
                elif bundle.embeddings is None and not self.fake_llm:
                    bundle.build_error = (
                        f"Embedding API is not configured ({self.runtime.embedding_api_key_env})"
                    )
                self._haystack_cache[cache_key] = bundle
                self._haystack_build_count += 1
        return self._haystack_cache[cache_key], {
            "haystack_cache_hit": cache_hit,
            "haystack_key": cache_key,
            "haystack_build_count": self._haystack_build_count,
            "haystack_build_ms": 0.0 if cache_hit else self._phase_timings["memory_build_time_ms"],
            **self._haystack_cache[cache_key].vector_cache_metadata,
        }

    def _retrieve(self, payload: dict) -> tuple[list[str], dict]:
        haystack, cache_meta = self._shared_haystack(payload)
        query = str(payload.get("query", ""))
        retrieval_meta = {
            **cache_meta,
            "candidate_count": len(haystack.units),
            "indexed_embedding_count": vector_count(haystack.embeddings),
            "unit_type_counts": {
                kind: sum(unit.kind == kind for unit in haystack.units)
                for kind in sorted({unit.kind for unit in haystack.units})
            },
            "phase_applicability": {"memory_build": True, "memory_query": True, "reader_generation": True},
        }
        if haystack.build_error:
            retrieval_meta.update({
                "retrieval": "failed",
                "failure_stage": "memory_build",
                "error": haystack.build_error,
            })
            return [], retrieval_meta
        with self.measure_phase("memory_query_time_ms"):
            if vector_count(haystack.embeddings):
                vectors = self.embed_texts(
                    [self.embedding_query_text(query)],
                    phase="memory_query",
                )
                if len(vectors) == 1:
                    qvec = vectors[0]
                    selected = rank_vectors(qvec, haystack.embeddings, k=self.retrieval_k)
                    retrieval_meta.update({
                        "retrieval": self.retrieval_name,
                        "scores": [score for score, _ in selected],
                        "retrieved_unit_ids": [haystack.units[idx].unit_id for _, idx in selected],
                    })
                    return [haystack.units[idx].text for _, idx in selected], retrieval_meta
                retrieval_meta.update({
                    "retrieval": "failed",
                    "failure_stage": "memory_query",
                    "error": self._last_embedding_error or "query embedding returned no vector",
                })
                return [], retrieval_meta
            if self.fake_llm:
                query_terms = {
                    term.strip(".,:;!?()[]{}\"'").lower()
                    for term in query.split()
                    if len(term) > 2
                }
                ranked = sorted(
                    (
                        (float(sum(term in view for term in query_terms)), idx)
                        for idx, view in enumerate(haystack.search_views)
                    ),
                    reverse=True,
                )
                selected = [idx for score, idx in ranked if score > 0][: self.retrieval_k]
                if not selected:
                    selected = list(range(max(0, len(haystack.units) - self.retrieval_k), len(haystack.units)))
                retrieval_meta.update({
                    "retrieval": f"fake_lexical_{self.retrieval_name}",
                    "retrieved_unit_ids": [haystack.units[idx].unit_id for idx in selected],
                })
                return [haystack.units[idx].text for idx in selected], retrieval_meta
        retrieval_meta.update({
            "retrieval": "failed",
            "failure_stage": "memory_build",
            "error": f"{self.reader_label} vector index is empty",
        })
        return [], retrieval_meta

    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        retrieved, retrieval_meta = self._retrieve(payload)
        if retrieval_meta.get("retrieval") == "failed":
            return "", [], "", [], {
                **retrieval_meta,
                "context_policy": self.context_policy,
                "status": "failed",
                "failure_stage": retrieval_meta.get("failure_stage", "memory_query"),
                "failure_reason": retrieval_meta.get("error", f"{self.reader_label} retrieval failed"),
            }
        with self.measure_phase("memory_query_time_ms"):
            context, packing = pack_text_items(
                retrieved,
                max_tokens=RETRIEVAL_CONTEXT_MAX_TOKENS,
            )
        retrieval_meta["context_packing"] = {
            **packing,
            "context_token_budget": RETRIEVAL_CONTEXT_MAX_TOKENS,
        }
        if self.can_call_llm():
            system = (
                f"You are the {self.reader_label} baseline for a long-context benchmark. "
                "Answer using only the retrieved memory units. Do not use hidden labels. "
                "Return JSON with answer, evidence, stage, dependencies."
            )
            user = (
                f"Dataset: {payload['dataset']}\n"
                f"Question/task:\n{payload['query']}\n\n"
                f"Retrieved memory units:\n{context}\n\n"
                + response_contract(payload)
            )
            with self.measure_phase("reader_generation_time_ms"):
                raw = self.chat_json(system, user)
                answer, evidence, stage, dependencies, metadata = parse_prediction_json(raw)
            answer, evidence, grounding_meta = enforce_longmem_evidence_grounding(
                payload, answer, evidence, retrieved
            )
            metadata.update(grounding_meta)
            metadata.update(retrieval_meta)
            metadata["context_policy"] = self.context_policy
            if not answer:
                if self.fallback_allowed():
                    fallback_answer, fallback_evidence = keyword_answer(payload["query"], retrieved)
                    metadata["fallback"] = "keyword_after_llm_error_or_empty"
                    return fallback_answer, fallback_evidence, infer_stage(payload["query"], fallback_evidence), [], metadata
                metadata.update({
                    "status": "failed",
                    "failure_stage": "reader_generation",
                    "failure_reason": str(raw.get("_error", "reader returned an empty answer")),
                })
                return "", [], "", [], metadata
            returned_evidence = evidence
            if "longmemeval-v2" not in str(payload.get("dataset", "")).lower():
                returned_evidence = evidence or retrieved[:3]
            return answer, returned_evidence, stage, dependencies, metadata
        if not self.fake_llm:
            return "", [], "", [], {
                **retrieval_meta,
                "context_policy": self.context_policy,
                "status": "failed",
                "failure_stage": "reader_generation",
                "failure_reason": f"LLM API is not configured ({self.runtime.api_key_env})",
            }
        with self.measure_phase("reader_generation_time_ms"):
            answer, evidence = keyword_answer(payload["query"], retrieved)
        return answer, evidence, infer_stage(payload["query"], evidence), [], {
            "context_policy": f"fake_{self.context_policy}",
            **retrieval_meta,
        }


class RawRagMethod(EmbeddingRagMethod):
    """Conventional fixed-width chunks over the serialized raw haystack."""

    name = "raw_rag"
    retrieval_name = "raw_chunk_embedding_rag"
    context_policy = "raw_chunk_embedding_rag_llm"
    reader_label = "Raw RAG"
    vector_index_kind = "raw_chunks_7000_overlap500_v1"

    def make_units(self, payload: dict) -> list[MemoryUnit]:
        return raw_chunks_from_history(payload.get("history", []))


class AtomicRagMethod(EmbeddingRagMethod):
    """Flat retrieval over lossless trajectory/state fragments."""

    name = "atomic_rag"
    retrieval_name = "atomic_state_embedding_rag"
    context_policy = "atomic_state_embedding_rag_llm"
    reader_label = "Atomic RAG"
    retrieval_k = 10
    vector_index_kind = "longmemeval_atomic_fragments_v2"

    def make_units(self, payload: dict) -> list[MemoryUnit]:
        return atomic_units_from_history(payload.get("history", []))


class PlanAndSolveMethod(Method):
    name = "plan_and_solve"

    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        if self.can_call_llm():
            plan_system = (
                "You are the planning step of the Plan-and-Solve baseline. "
                "Create a concise solution plan from the user task and available history summary. "
                "Do not answer yet. Return JSON with a plan array and useful_focus_terms array."
            )
            history_preview = format_history(payload, max_chars=16000)
            plan_user = (
                f"Question/task:\n{payload['query']}\n\nHistory preview:\n{history_preview}\n\n"
                "Return JSON only: {\"plan\":[\"...\"], \"useful_focus_terms\":[\"...\"]}."
            )
            with self.measure_phase("reader_generation_time_ms"):
                plan_raw = self.chat_json(plan_system, plan_user, max_tokens=1024)
            plan_items = plan_raw.get("plan", [])
            if not isinstance(plan_items, list):
                plan_items = [str(plan_items)]
            plan_items = [str(item) for item in plan_items if str(item).strip()]
            if not plan_items and not self.fallback_allowed():
                return "", [], "", [], {
                    "context_policy": "plan_and_solve_llm",
                    "plan_raw": plan_raw,
                    "status": "failed",
                    "failure_stage": "reader_generation",
                    "failure_reason": str(plan_raw.get("_error", "planning step returned an empty plan")),
                }
            solve_system = (
                "You are the solve step of the Plan-and-Solve baseline. "
                "Follow the provided plan and use the chronological history. "
                "Do not use graph memory or hidden labels. Return JSON with answer, evidence, stage, dependencies."
            )
            solve_user = (
                f"Plan:\n{chr(10).join('- ' + str(item) for item in plan_items)}\n\n"
                f"Question/task:\n{payload['query']}\n\n"
                f"Chronological history:\n{format_history(payload, max_chars=42000)}\n\n"
                + response_contract(payload)
            )
            with self.measure_phase("reader_generation_time_ms"):
                raw = self.chat_json(solve_system, solve_user)
                answer, evidence, stage, dependencies, metadata = parse_prediction_json(raw)
            metadata.update({"context_policy": "plan_and_solve_llm", "plan": plan_items, "plan_raw": plan_raw})
            if not answer:
                if self.fallback_allowed():
                    texts = [f"Plan: {'; '.join(str(item) for item in plan_items)}\n{msg['content']}" for msg in payload["history"]]
                    fallback_answer, fallback_evidence = keyword_answer(payload["query"], texts)
                    metadata["fallback"] = "keyword_after_llm_error_or_empty"
                    return fallback_answer, fallback_evidence, infer_stage(payload["query"], fallback_evidence), [], metadata
                metadata.update({
                    "status": "failed",
                    "failure_stage": "reader_generation",
                    "failure_reason": str(raw.get("_error", "reader returned an empty answer")),
                })
                return "", [], "", [], metadata
            return answer, evidence, stage, dependencies, metadata
        if not self.fake_llm:
            return "", [], "", [], {
                "context_policy": "plan_and_solve_llm",
                "status": "failed",
                "failure_stage": "reader_generation",
                "failure_reason": f"LLM API is not configured ({self.runtime.api_key_env})",
            }
        texts = [f"Plan: understand task, retrieve relevant history, answer.\n{msg['content']}" for msg in payload["history"]]
        with self.measure_phase("reader_generation_time_ms"):
            answer, evidence = keyword_answer(payload["query"], texts)
        return answer, evidence, infer_stage(payload["query"], evidence), [], {
            "context_policy": "plan_then_truncated_context",
            "plan": ["identify relevant history", "answer from evidence"],
        }


def infer_stage(query: str, texts: list[str]) -> str:
    haystack = (query + "\n" + "\n".join(texts)).lower()
    for stage in ("requirements", "data_collection", "training", "deployment"):
        if stage in haystack or stage.replace("_", " ") in haystack:
            return stage
    if "tool" in haystack or "api" in haystack:
        return "tool_execution"
    return ""
