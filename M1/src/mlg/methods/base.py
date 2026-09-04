from __future__ import annotations

import time
import hashlib
import json
import math
import re
from functools import lru_cache
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from mlg.config import RuntimeConfig
from mlg.data.validation import assert_no_oracle_payload
from mlg.schemas import Episode, Prediction


class Method(ABC):
    name = "base"
    uses_graph_sidecar = False
    graph_ablation = ""

    def __init__(
        self,
        *,
        fake_llm: bool = False,
        model: str = "",
        runtime: RuntimeConfig | None = None,
        model_profile: str = "default",
        seed: int = 0,
        allow_fallback: bool | None = None,
    ) -> None:
        self.fake_llm = fake_llm
        self.runtime = runtime or RuntimeConfig()
        self.model = model or self.runtime.model
        self.model_profile = model_profile or self.runtime.profile_name
        self.seed = seed
        self.allow_fallback = fake_llm if allow_fallback is None else allow_fallback
        self._client = None
        self._embedding_client = None
        self._last_embedding_error = ""
        self._api_calls: list[dict[str, Any]] = []
        self._phase_timings: dict[str, float] = {}

    @staticmethod
    def shared_haystack_key(payload: dict) -> str:
        """Return a stable, non-oracle identity for one reusable history."""
        history_ref = payload.get("history_ref", {})
        signature = str(history_ref.get("signature", "")).strip()
        if signature:
            return f"{payload.get('dataset', '')}:{signature}"
        digest = hashlib.sha256()
        for message in payload.get("history", []):
            digest.update(str(message.get("role", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(message.get("content", "")).encode("utf-8"))
            digest.update(b"\0")
        return f"{payload.get('dataset', '')}:{digest.hexdigest()}"

    def predict(self, episode: Episode) -> Prediction:
        payload = episode.method_input(
            include_visible_graph=self.uses_graph_sidecar,
            graph_ablation=self.graph_ablation,
        )
        assert_no_oracle_payload(payload)
        self._api_calls = []
        self._phase_timings = {
            "memory_build_time_ms": 0.0,
            "memory_query_time_ms": 0.0,
            "reader_generation_time_ms": 0.0,
        }
        start = time.perf_counter()
        answer, evidence, stage, dependencies, metadata = self._predict_from_payload(payload)
        end_to_end_time_ms = (time.perf_counter() - start) * 1000
        status = str(metadata.get("status", "ok"))
        failure_stage = str(metadata.get("failure_stage", ""))
        failure_reason = str(metadata.get("failure_reason", ""))
        api_usage = summarize_api_usage(self._api_calls, fake_llm=self.fake_llm)
        metadata["api_calls"] = list(self._api_calls)
        metadata["api_usage"] = api_usage
        metadata["usage_missing"] = bool(api_usage.get("usage_missing"))
        metadata["fallback_policy"] = "allow_smoke_fallback" if self.allow_fallback else "fail_closed"
        metadata["timing"] = {
            **self._phase_timings,
            "end_to_end_time_ms": end_to_end_time_ms,
        }
        token_usage = None if api_usage.get("usage_missing") else int(api_usage["total"]["total_tokens"])
        return Prediction(
            episode_id=episode.episode_id,
            method=self.name,
            model_profile=self.model_profile,
            answer=answer,
            evidence=evidence,
            stage=stage,
            dependencies=dependencies,
            token_usage=token_usage,
            api_usage=api_usage,
            memory_build_time_ms=self._phase_timings["memory_build_time_ms"],
            memory_query_time_ms=self._phase_timings["memory_query_time_ms"],
            reader_generation_time_ms=self._phase_timings["reader_generation_time_ms"],
            end_to_end_time_ms=end_to_end_time_ms,
            status=status,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            seed=self.seed,
            metadata=metadata,
        )

    @abstractmethod
    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        raise NotImplementedError

    def can_call_llm(self) -> bool:
        return bool(self.runtime.api_key) and not self.fake_llm

    def fallback_allowed(self) -> bool:
        return bool(self.allow_fallback)

    @contextmanager
    def measure_phase(self, field: str):
        if field not in self._phase_timings:
            raise ValueError(f"Unknown timing phase: {field}")
        started = time.perf_counter()
        try:
            yield
        finally:
            self._phase_timings[field] += (time.perf_counter() - started) * 1000

    def client(self):
        if self._client is None:
            import httpx
            from openai import OpenAI

            http_client = httpx.Client(
                verify=self.runtime.verify_ssl,
                trust_env=not self.runtime.ignore_proxy,
                timeout=self.runtime.request_timeout_seconds,
            )
            self._client = OpenAI(api_key=self.runtime.api_key, base_url=self.runtime.base_url, http_client=http_client)
        return self._client

    def can_call_embedding(self) -> bool:
        return bool(self.runtime.embedding_api_key) and not self.fake_llm

    def embedding_client(self):
        if self._embedding_client is None:
            import httpx
            from openai import OpenAI

            http_client = httpx.Client(
                verify=self.runtime.verify_ssl,
                trust_env=not self.runtime.ignore_proxy,
                timeout=self.runtime.request_timeout_seconds,
            )
            self._embedding_client = OpenAI(
                api_key=self.runtime.embedding_api_key,
                base_url=self.runtime.embedding_base_url,
                http_client=http_client,
                max_retries=5,
            )
        return self._embedding_client

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        phase: str = "reader_generation",
    ) -> dict[str, Any]:
        if not self.can_call_llm():
            return {"_error": f"LLM API is not configured ({self.runtime.api_key_env})"}
        started = time.perf_counter()
        try:
            extra_body = (
                {"thinking": {"type": self.runtime.thinking_mode}}
                if self.runtime.thinking_mode in {"enabled", "disabled"}
                else None
            )
            response = self.client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=self.runtime.temperature,
                max_tokens=max_tokens or self.runtime.max_tokens,
                extra_body=extra_body,
            )
            self._api_calls.append(
                api_call_record(
                    phase=phase,
                    kind="chat.completions",
                    model=self.model,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    usage=getattr(response, "usage", None),
                )
            )
            content = response.choices[0].message.content or "{}"
            parsed = parse_json_response(content)
            finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "")
            if finish_reason:
                parsed["_finish_reason"] = finish_reason
            return parsed
        except Exception as exc:
            self._api_calls.append(
                api_call_record(
                    phase=phase,
                    kind="chat.completions",
                    model=self.model,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    usage=None,
                    error=str(exc),
                )
            )
            return {"_error": str(exc)}

    def embed_texts(self, texts: list[str], *, phase: str = "memory_query") -> list[list[float]]:
        if not self.can_call_embedding() or not texts:
            return []
        self._last_embedding_error = ""
        vectors: list[list[float]] = []
        offset = 0
        try:
            while offset < len(texts):
                end = self._embedding_batch_end(texts, offset)
                while True:
                    batch = texts[offset:end]
                    estimated_tokens = sum(token_count(text) for text in batch)
                    started = time.perf_counter()
                    try:
                        response = self.embedding_client().embeddings.create(
                            model=self.runtime.embedding_model,
                            input=batch,
                            encoding_format="float",
                        )
                        response_data = getattr(response, "data", None)
                        if not response_data:
                            model_extra = getattr(response, "model_extra", None) or {}
                            provider_error = (
                                model_extra.get("error", {})
                                if isinstance(model_extra, dict)
                                else {}
                            )
                            provider_message = (
                                provider_error.get("message", "")
                                if isinstance(provider_error, dict)
                                else str(provider_error)
                            )
                            detail = f": {provider_message}" if provider_message else ""
                            raise RuntimeError(
                                f"embedding provider returned no vectors{detail}"
                            )
                        batch_vectors = [
                            getattr(item, "embedding", None)
                            for item in response_data
                        ]
                        if len(batch_vectors) != len(batch) or any(
                            not vector for vector in batch_vectors
                        ):
                            raise RuntimeError(
                                "embedding provider returned incomplete vectors"
                            )
                    except Exception as exc:
                        error = str(exc)
                        retry_with_smaller_batch = (
                            len(batch) > 1
                            and self._embedding_input_too_long(error)
                        )
                        call = api_call_record(
                            phase=phase,
                            kind="embeddings",
                            model=self.runtime.embedding_model,
                            elapsed_ms=(time.perf_counter() - started) * 1000,
                            usage=None,
                            error=error,
                        )
                        call.update({
                            "input_count": len(batch),
                            "estimated_input_tokens": estimated_tokens,
                            "adaptive_batch_retry": retry_with_smaller_batch,
                        })
                        if self._embedding_input_too_long(error):
                            call.update({
                                "usage_not_applicable": True,
                                "usage_not_applicable_reason": (
                                    "provider_rejected_input_before_embedding"
                                ),
                            })
                        self._api_calls.append(call)
                        if retry_with_smaller_batch:
                            end = offset + max(1, len(batch) // 2)
                            continue
                        raise
                    call = api_call_record(
                        phase=phase,
                        kind="embeddings",
                        model=self.runtime.embedding_model,
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        usage=getattr(response, "usage", None),
                    )
                    call.update({
                        "input_count": len(batch),
                        "estimated_input_tokens": estimated_tokens,
                        "adaptive_batch_retry": False,
                    })
                    self._api_calls.append(call)
                    vectors.extend(batch_vectors)
                    break
                offset = end
            return vectors
        except Exception as exc:
            self._last_embedding_error = str(exc)
            return []

    def _embedding_batch_end(self, texts: list[str], start: int) -> int:
        max_items = max(1, int(self.runtime.embedding_batch_max_items))
        return min(len(texts), start + max_items)

    @staticmethod
    def _embedding_input_too_long(error: str) -> bool:
        normalized = error.lower()
        return any(
            marker in normalized
            for marker in (
                "exceeds maximum allowed token size",
                "input length",
                "maximum context length",
                "too many tokens",
                "input is too long",
            )
        )

    def embed_texts_resumable(
        self,
        texts: list[str],
        *,
        cache_spec: Any,
        prior_metadata: dict[str, Any] | None = None,
        phase: str = "memory_build",
        progress_label: str = "vector-index",
    ) -> tuple[Any | None, dict[str, Any]]:
        """Embed a large index with per-batch disk checkpoints and visible progress."""
        from mlg.methods.vector_cache import (
            finalize_partial_vector_cache,
            load_partial_vector_cache,
            persist_vector_cache,
            persist_vector_cache_batch,
        )

        metadata = dict(prior_metadata or {})
        if not texts:
            self._last_embedding_error = "embedding index contains no texts"
            metadata.update({"vector_cache_status": "build_failed", "vector_cache_error": self._last_embedding_error})
            return None, metadata
        if not cache_spec.enabled:
            vectors = self.embed_texts(texts, phase=phase)
            if len(vectors) != len(texts):
                return None, metadata
            return persist_vector_cache(cache_spec, vectors, metadata)

        try:
            completed, _, _ = load_partial_vector_cache(cache_spec)
        except Exception as exc:
            self._last_embedding_error = str(exc)
            metadata.update({
                "vector_cache_status": "invalid_partial_rebuild_required",
                "vector_cache_error": self._last_embedding_error,
            })
            return None, metadata

        initial_completed = completed
        total = len(texts)
        started = time.perf_counter()
        metadata["embedding_batch_policy"] = {
            "max_items": int(self.runtime.embedding_batch_max_items),
            "overflow_policy": "halve_and_retry_on_provider_length_error",
            "token_estimator": "o200k_base_audit_only",
        }
        if completed:
            print(
                f"[vector-cache] {progress_label} resume {completed}/{total} "
                f"({completed / total:.1%})",
                flush=True,
            )
        while completed < total:
            end = self._embedding_batch_end(texts, completed)
            batch_vectors = self.embed_texts(texts[completed:end], phase=phase)
            if len(batch_vectors) != end - completed:
                metadata.update({
                    "vector_cache_status": "partial_build_failed",
                    "vector_cache_partial_count": completed,
                    "vector_cache_error": self._last_embedding_error or "embedding batch returned incomplete vectors",
                })
                return None, metadata
            metadata = persist_vector_cache_batch(
                cache_spec,
                batch_vectors,
                start=completed,
                prior_metadata=metadata,
            )
            completed = end
            completed_batches = int(
                metadata.get("vector_cache_partial_batches", 0)
            )
            if completed == total or completed_batches % 10 == 0:
                elapsed = time.perf_counter() - started
                new_items = completed - initial_completed
                rate = new_items / elapsed if elapsed > 0 else 0.0
                remaining_seconds = (total - completed) / rate if rate > 0 else 0.0
                print(
                    f"[vector-cache] {progress_label} {completed}/{total} "
                    f"({completed / total:.1%}) checkpoint={completed_batches} "
                    f"eta={remaining_seconds:.0f}s",
                    flush=True,
                )
        vectors, metadata = finalize_partial_vector_cache(cache_spec, metadata)
        metadata["vector_cache_resumed_from_count"] = initial_completed
        return vectors, metadata


def api_call_record(
    *,
    phase: str,
    kind: str,
    model: str,
    elapsed_ms: float,
    usage: Any,
    error: str = "",
) -> dict[str, Any]:
    parsed_usage = extract_api_usage(usage)
    return {
        "phase": phase,
        "kind": kind,
        "model": model,
        "elapsed_ms": elapsed_ms,
        "usage": parsed_usage,
        "usage_missing": parsed_usage is None,
        "error": error,
    }


def extract_api_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)),
            "completion_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None)),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total = usage.get("total_tokens")
    if prompt is None and total is None:
        return None
    prompt_value = int(prompt or 0)
    completion_value = int(completion or 0)
    total_value = int(total if total is not None else prompt_value + completion_value)
    return {
        "prompt_tokens": prompt_value,
        "completion_tokens": completion_value,
        "total_tokens": total_value,
    }


def summarize_api_usage(calls: list[dict[str, Any]], *, fake_llm: bool) -> dict[str, Any]:
    phases = ("memory_build", "memory_query", "reader_generation")
    output: dict[str, Any] = {}
    for phase in phases:
        phase_calls = [call for call in calls if call.get("phase") == phase]
        applicable_calls = [
            call for call in phase_calls
            if not call.get("usage_not_applicable")
        ]
        known = [call["usage"] for call in phase_calls if call.get("usage") is not None]
        output[phase] = {
            "prompt_tokens": sum(item["prompt_tokens"] for item in known),
            "completion_tokens": sum(item["completion_tokens"] for item in known),
            "total_tokens": sum(item["total_tokens"] for item in known),
            "call_count": len(phase_calls),
            "usage_missing_call_count": sum(
                bool(call.get("usage_missing"))
                for call in applicable_calls
            ),
            "usage_not_applicable_call_count": (
                len(phase_calls) - len(applicable_calls)
            ),
        }
    output["total"] = {
        field: sum(output[phase][field] for phase in phases)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    output["call_count"] = len(calls)
    output["usage_missing"] = False if fake_llm else (
        not calls or any(
            call.get("usage_missing") and not call.get("usage_not_applicable")
            for call in calls
        )
    )
    output["source"] = "not_applicable_fake" if fake_llm else "api_response_usage"
    return output


def trim_text(text: str, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    marker = "\n...[trimmed]...\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    content_budget = max_chars - len(marker)
    head = content_budget // 2
    tail = content_budget - head
    return text[:head] + marker + text[-tail:]


@lru_cache(maxsize=1)
def token_encoder():
    """Return the deterministic tokenizer used for local context budgeting.

    Qwen does not expose its server-side tokenizer through the OpenAI-compatible
    endpoint. We use o200k_base with a separate safety reserve and always record
    the estimator name plus the API's actual prompt usage.
    """
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def token_count(text: str) -> int:
    return len(token_encoder().encode(text))


def truncate_text_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    encoded = token_encoder().encode(text)
    if len(encoded) <= max_tokens:
        return text
    return token_encoder().decode(encoded[:max_tokens])


def pack_text_items(items: list[str], *, max_tokens: int, separator: str = "\n\n") -> tuple[str, dict[str, Any]]:
    """Pack complete items in order and token-truncate only the final item."""
    if max_tokens <= 0:
        return "", {"packed_items": 0, "packed_tokens_estimate": 0, "truncated": bool(items)}
    encoder = token_encoder()
    separator_tokens = encoder.encode(separator)
    output_tokens: list[int] = []
    packed_items = 0
    truncated = False
    for item in items:
        encoded = encoder.encode(item)
        prefix = separator_tokens if output_tokens else []
        remaining = max_tokens - len(output_tokens) - len(prefix)
        if remaining <= 0:
            truncated = True
            break
        output_tokens.extend(prefix)
        if len(encoded) <= remaining:
            output_tokens.extend(encoded)
            packed_items += 1
            continue
        output_tokens.extend(encoded[:remaining])
        packed_items += 1
        truncated = True
        break
    if packed_items < len(items):
        truncated = True
    return encoder.decode(output_tokens), {
        "packed_items": packed_items,
        "total_items": len(items),
        "packed_tokens_estimate": len(output_tokens),
        "truncated": truncated,
        "tokenizer": "tiktoken:o200k_base_estimate",
    }


def formatted_history_items(payload: dict) -> list[str]:
    return [
        f"[turn={msg.get('turn_index', '')} role={msg.get('role', 'user')}]\n{msg.get('content', '')}"
        for msg in payload.get("history", [])
    ]


def extract_partial_json_answer(content: str) -> str | None:
    """Recover a leading answer string from a truncated JSON model response."""
    for field in ("answer", "final_answer"):
        match = re.search(
            rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)"',
            content,
            flags=re.DOTALL,
        )
        if not match:
            continue
        try:
            return str(json.loads(f'"{match.group(1)}"'))
        except json.JSONDecodeError:
            return match.group(1)
    return None


def parse_json_response(content: str) -> dict[str, Any]:
    """Parse provider JSON, repairing malformed or truncated containers if needed."""
    candidates = [content]
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end >= start:
        candidates.append(content[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    try:
        from json_repair import loads as repair_json_loads

        for candidate in candidates:
            parsed = repair_json_loads(candidate)
            if isinstance(parsed, dict):
                parsed.setdefault("_response_parse", "json_repair")
                return parsed
    except (ImportError, ValueError, TypeError):
        pass
    partial_answer = extract_partial_json_answer(content)
    if partial_answer is not None:
        return {"answer": partial_answer, "_response_parse": "partial_json_answer"}
    return {"answer": content, "_response_parse": "unparsed_content"}


def format_history(payload: dict, *, max_chars: int = 24000) -> str:
    parts = []
    for msg in payload.get("history", []):
        role = msg.get("role", "user")
        turn = msg.get("turn_index", "")
        content = str(msg.get("content", ""))
        parts.append(f"[turn={turn} role={role}]\n{content}")
    return trim_text("\n\n".join(parts), max_chars=max_chars)


def parse_prediction_json(raw: dict[str, Any]) -> tuple[str, list[str], str, list[dict], dict]:
    answer = normalize_model_answer(str(raw.get("answer", raw.get("final_answer", ""))))
    if not answer:
        answer = normalize_model_answer(
            _task_action_answer_from_raw(raw)
            or _action_answer_from_raw(raw)
            or _tool_answer_from_raw(raw)
        )
    evidence_raw = raw.get("evidence", raw.get("evidence_quotes", []))
    if isinstance(evidence_raw, str):
        evidence = [evidence_raw]
    elif isinstance(evidence_raw, list):
        evidence = [str(item) for item in evidence_raw]
    else:
        evidence = []
    stage = str(raw.get("stage", raw.get("current_stage", "")))
    deps_raw = raw.get("dependencies", raw.get("dependency_links", raw.get("used_dependencies", [])))
    dependencies = normalize_dependencies(deps_raw)
    return answer, evidence, stage, dependencies, {"llm_raw": raw}


def normalize_model_answer(answer: str) -> str:
    """Remove answer-format wrappers, including JSON-decoded LaTeX escapes."""
    clean = str(answer).strip()
    # Models sometimes emit ``\boxed`` or ``\text`` with only one JSON
    # backslash. A conforming decoder then turns ``\b`` into backspace and
    # ``\t`` into tab before this function sees the value.
    clean = clean.replace("\x08oxed", "\\boxed")
    clean = clean.replace("\t" + "ext", "\\text")
    for _ in range(3):
        match = re.fullmatch(
            r"\s*\\?(?:boxed|text)\s*\{(.*)\}\s*",
            clean,
            flags=re.DOTALL,
        )
        if not match:
            break
        clean = match.group(1).strip()
    return clean


def _action_answer_from_raw(raw: dict[str, Any]) -> str:
    action_type = raw.get("action_type") or raw.get("action") or raw.get("operation")
    target = raw.get("target_element") or raw.get("element") or raw.get("element_id")
    value = raw.get("value") or raw.get("input_value")
    if not action_type and not target and not value:
        return ""
    parts = [str(action_type).upper() if action_type else "ACTION"]
    if target:
        parts.append(f"element={target}")
    if value:
        parts.append(f"value={value}")
    return " ".join(parts)


def _task_action_answer_from_raw(raw: dict[str, Any]) -> str:
    action_id = raw.get("action_id") or raw.get("next_action_id") or raw.get("selected_action")
    return f"action_id={action_id}" if action_id else ""


def _tool_answer_from_raw(raw: dict[str, Any]) -> str:
    tool_name = raw.get("tool_name")
    api_name = raw.get("api_name")
    arguments = raw.get("arguments") or raw.get("tool_input")
    if not tool_name and not api_name:
        return ""
    parts = []
    if tool_name:
        parts.append(f"tool_name={tool_name}")
    if api_name:
        parts.append(f"api_name={api_name}")
    if arguments is not None:
        parts.append(f"arguments={json.dumps(arguments, ensure_ascii=False, sort_keys=True)}")
    return " ".join(parts)


def normalize_dependencies(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str) and item.strip():
            normalized.append({"source": item.strip(), "target": ""})
    return normalized


def lexical_scores(query: str, texts: list[str]) -> list[tuple[float, int, str]]:
    query_terms = {term.strip(".,:;!?()[]{}\"'").lower() for term in query.split() if len(term) > 2}
    scored = []
    for idx, text in enumerate(texts):
        lower = text.lower()
        score = float(sum(1 for term in query_terms if term in lower))
        if score:
            scored.append((score, idx, text))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_top_k(query: str, texts: list[str], *, k: int = 6, method: Method | None = None) -> tuple[list[str], dict[str, Any]]:
    if not texts:
        return [], {"retrieval": "empty"}
    original_count = len(texts)
    if len(texts) > 80:
        lexical_prefilter = lexical_scores(query, texts)[:80]
        if lexical_prefilter:
            texts = [item[2] for item in lexical_prefilter]
        else:
            texts = texts[-80:]
    if method is not None and method.can_call_embedding():
        try:
            # DashScope embeddings reject an individual input longer than 33k.
            # Rank using bounded head+tail views while returning the original text.
            embedding_inputs = [
                trim_text(item, max_chars=EMBEDDING_INPUT_MAX_CHARS)
                for item in [query] + texts
            ]
            vectors = method.embed_texts(embedding_inputs)
            if len(vectors) == len(texts) + 1:
                qvec = vectors[0]
                scored = [(cosine(qvec, vec), idx, texts[idx]) for idx, vec in enumerate(vectors[1:])]
                scored.sort(key=lambda item: item[0], reverse=True)
                return [item[2] for item in scored[:k]], {
                    "retrieval": "embedding",
                    "candidate_count": original_count,
                    "embedded_count": len(texts),
                    "scores": [item[0] for item in scored[:k]],
                }
            if getattr(method, "_last_embedding_error", ""):
                if not method.fallback_allowed():
                    return [], {
                        "retrieval": "failed",
                        "candidate_count": original_count,
                        "embedded_count": 0,
                        "error": method._last_embedding_error,
                    }
                lexical = lexical_scores(query, texts)[:k]
                return [item[2] for item in lexical] or texts[-k:], {
                    "retrieval": "lexical_fallback_after_embedding_error",
                    "candidate_count": original_count,
                    "embedded_count": 0,
                    "error": method._last_embedding_error,
                }
        except Exception as exc:
            if not method.fallback_allowed():
                return [], {
                    "retrieval": "failed",
                    "candidate_count": original_count,
                    "embedded_count": 0,
                    "error": str(exc),
                }
            lexical = lexical_scores(query, texts)[:k]
            return [item[2] for item in lexical] or texts[-k:], {
                "retrieval": "lexical_fallback_after_embedding_error",
                "candidate_count": original_count,
                "embedded_count": 0,
                "error": str(exc),
            }
    if method is not None and not method.fallback_allowed():
        return [], {
            "retrieval": "failed",
            "candidate_count": original_count,
            "embedded_count": 0,
            "error": (
                "embedding retrieval returned incomplete vectors"
                if method.can_call_embedding()
                else f"Embedding API is not configured ({method.runtime.embedding_api_key_env})"
            ),
        }
    lexical = lexical_scores(query, texts)[:k]
    return [item[2] for item in lexical] or texts[-k:], {"retrieval": "lexical", "candidate_count": original_count}


# The default NVIDIA OpenRouter model rejects an individual input above 8192
# tokens. Accessibility-tree text can tokenize close to one token per character,
# so retain a small safety margin. This bound applies only to embedding views;
# retrieval still returns the original untruncated text.
EMBEDDING_INPUT_MAX_CHARS = 7500


def keyword_answer(query: str, texts: list[str]) -> tuple[str, list[str]]:
    query_terms = {term.strip(".,:;!?()[]{}\"'").lower() for term in query.split() if len(term) > 2}
    ranked = []
    for text in texts:
        lower = text.lower()
        score = sum(1 for term in query_terms if term in lower)
        if score:
            ranked.append((score, text))
    ranked.sort(key=lambda item: item[0], reverse=True)
    evidence = [item[1] for item in ranked[:3]]
    if not evidence and texts:
        evidence = [texts[-1]]
    answer = extract_direct_value(query, evidence) or (evidence[0][:240] if evidence else "")
    return answer, evidence


def extract_direct_value(query: str, evidence: list[str]) -> str:
    markers = [" is ", " equals ", " answer is ", " code is ", "->", ":"]
    for text in evidence:
        lower = text.lower()
        if "hidden needle code" in lower and "needle_value_" in lower:
            start = lower.find("needle_value_")
            return text[start : start + 17].strip(" .")
        for marker in markers:
            if marker in text:
                candidate = text.split(marker, 1)[-1].strip()
                if 0 < len(candidate) <= 120:
                    return candidate.strip(" .\"'")
    return ""
