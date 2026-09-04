from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlg.config import RuntimeConfig, load_model_profiles
from mlg.methods.base import Method
from mlg.methods.vector_cache import build_vector_cache_spec, load_vector_cache
from mlg.m4_taskgraph_summary import summarize_taskgraph_batch


class M4Model(Method):
    name = "m4_model_service"

    def _predict_from_payload(self, payload: dict):
        raise NotImplementedError("M4 uses batched document-QA calls")

    @classmethod
    def from_config(cls, path: Path, profile: str = "deepseek_v4_flash") -> "M4Model":
        model_profile = load_model_profiles(path)[profile]
        runtime = RuntimeConfig.from_profile(model_profile)
        return cls(model=runtime.model, runtime=runtime, model_profile=profile, allow_fallback=False)

    def cached_embeddings(
        self, texts: list[str], ids: list[str], *, dataset: str, kind: str,
    ) -> tuple[Any, dict[str, Any]]:
        spec = build_vector_cache_spec(
            self.runtime,
            index_kind=f"m4_{dataset}_{kind}_v2",
            haystack_key=f"m4-target-v2:{dataset}:{kind}",
            unit_ids=ids,
            texts=texts,
        )
        vectors, metadata = load_vector_cache(spec)
        if vectors is None:
            vectors, metadata = self.embed_texts_resumable(
                texts,
                cache_spec=spec,
                prior_metadata=metadata,
                phase="memory_build" if kind == "chunks" else "memory_query",
                progress_label=f"M4 {dataset} {kind}",
            )
        if vectors is None:
            raise RuntimeError(self._last_embedding_error or f"failed to embed M4 {dataset} {kind}")
        return vectors, metadata

    def answer_batch(
        self, dataset: str, items: list[dict[str, Any]], *, shared_context: str = "",
    ) -> tuple[dict[str, str], dict[str, Any]]:
        system = (
            "Write a high-quality short answer for each given question using only the provided search results "
            "(some of which might be irrelevant). Do not use outside knowledge. Return JSON as "
            '{"answers":[{"query_id":"...","answer":"..."}]} with no explanation.'
        )
        if items and all(item.get("method") == "TaskGraph" for item in items):
            system += (
                " TaskGraph results include a DOCUMENT MAP followed by evidence blocks. The map exposes each "
                "document's source and date, while evidence headers expose origin, edge, relation, seed, and "
                "route_terms provenance. Every routed path is composed only of relations stored in the "
                "query-independent corpus graph. Use these fields to locate the question's documents, but treat retrieval "
                "provenance only as a navigation hint and never as a factual claim; factual support must come from "
                "the evidence text."
            )
        questions = [
            {"query_id": item["query_id"], "question": item["question"], **(
                {} if shared_context else {"search_results": item["context"]}
            )}
            for item in items
        ]
        payload = {"search_results": shared_context, "questions": questions} if shared_context else {"questions": questions}
        max_tokens = max(512, 128 * len(items) + 128)
        raw = self.chat_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens)
        if raw.get("_error"):
            raise RuntimeError(str(raw["_error"]))
        answer_rows = raw.get("answers", [])
        if isinstance(answer_rows, dict):
            answer_rows = [{"query_id": key, "answer": value} for key, value in answer_rows.items()]
        answers = {
            str(row.get("query_id", "")): str(row.get("answer", "")).strip()
            for row in answer_rows
            if isinstance(row, dict) and str(row.get("answer", "")).strip()
        }
        expected = {item["query_id"] for item in items}
        repaired_query_id = False
        if len(expected) == 1 and not expected.intersection(answers) and len(answer_rows) == 1:
            only_row = answer_rows[0]
            if isinstance(only_row, dict) and str(only_row.get("answer", "")).strip():
                answers[next(iter(expected))] = str(only_row["answer"]).strip()
                repaired_query_id = True
        missing = sorted(expected - set(answers))
        if missing:
            raise RuntimeError(f"reader omitted {len(missing)} answers: {missing[:3]}")
        call = dict(self._api_calls[-1]) if self._api_calls else {}
        if repaired_query_id:
            call["single_query_id_repaired"] = True
        return {key: answers[key] for key in expected}, call

    def memory_keys_batch(
        self, items: list[dict[str, str]],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        system = (
            "Create one concise semantic retrieval key for each memory value. Summarize its entities, events, "
            "relations, dates, numbers, and distinctive details; preserve explicit names and values. The key is "
            "used only to retrieve the unchanged original value. Process every item independently and return JSON "
            'only as {"keys":[{"chunk_id":"...","key":"..."}]}.'
        )
        payload = {"items": [
            {"chunk_id": item["chunk_id"], "memory_value": item["text"]}
            for item in items
        ]}
        raw = self.chat_json(
            system, json.dumps(payload, ensure_ascii=False),
            max_tokens=max(512, 160 * len(items)), phase="memory_build",
        )
        if raw.get("_error"):
            raise RuntimeError(str(raw["_error"]))
        key_rows = raw.get("keys", [])
        if isinstance(key_rows, dict):
            key_rows = [{"chunk_id": key, "key": value} for key, value in key_rows.items()]
        keys = {
            str(row.get("chunk_id", "")): str(row.get("key", "")).strip()
            for row in key_rows
            if isinstance(row, dict) and str(row.get("key", "")).strip()
        }
        expected = {item["chunk_id"] for item in items}
        repaired_chunk_id = False
        if len(expected) == 1 and not expected.intersection(keys) and len(key_rows) == 1:
            only_row = key_rows[0]
            if isinstance(only_row, dict) and str(only_row.get("key", "")).strip():
                keys[next(iter(expected))] = str(only_row["key"]).strip()
                repaired_chunk_id = True
        missing = sorted(expected - set(keys))
        if missing:
            raise RuntimeError(f"memory key builder omitted {len(missing)} keys: {missing[:3]}")
        call = dict(self._api_calls[-1]) if self._api_calls else {}
        if repaired_chunk_id:
            call["single_chunk_id_repaired"] = True
        return {key: keys[key] for key in expected}, call

    def taskgraph_summaries_batch(
        self, items: list[dict[str, str]],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        return summarize_taskgraph_batch(self, items)

    def judge_batch(self, items: list[dict[str, str]]) -> tuple[dict[str, bool], dict[str, Any]]:
        system = (
            "Check whether each predicted answer appropriately answers its query in a similar way to the ground-truth "
            "answer. Assign 1 when it does and 0 when it does not. Judge semantic correctness, not exact wording. "
            'Return JSON only as {"judgments":[{"query_id":"...","correct":0}]}.'
        )
        payload = {"items": [
            {
                "query_id": item["query_id"], "query": item["question"],
                "ground_truth_answer": item["gold"], "predicted_answer": item["prediction"],
            }
            for item in items
        ]}
        raw = self.chat_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max(256, 32 * len(items)), phase="evaluation")
        if raw.get("_error"):
            raise RuntimeError(str(raw["_error"]))
        rows = raw.get("judgments", [])
        values = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            query_id = str(row.get("query_id", ""))
            value = str(row.get("correct", "")).strip().lower()
            if value not in {"0", "1", "false", "true"}:
                raise RuntimeError(f"judge returned invalid value for {query_id}: {value!r}")
            values[query_id] = value in {"1", "true"}
        expected = {item["query_id"] for item in items}
        missing = sorted(expected - set(values))
        if missing:
            raise RuntimeError(f"judge omitted {len(missing)} judgments: {missing[:3]}")
        call = dict(self._api_calls[-1]) if self._api_calls else {}
        return {key: values[key] for key in expected}, call
