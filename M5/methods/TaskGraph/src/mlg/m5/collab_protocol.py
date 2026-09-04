from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any

from mlg.m5.collab_state import (
    CollaborationProtocolError,
    CollaborationRunState,
    locked_action,
    now_utc,
    sha_file,
    sha_payload,
    sha_persisted_payload,
)
from mlg.m5.continuity import load_local_continuity
from mlg.m5.dataset import han_char_count
from mlg.m5.length_policy import (
    MECHANICAL_ACCEPTED_HAN_CHARS,
    is_mechanically_accepted,
)
from mlg.m5.io import atomic_write_json, atomic_write_text, read_json
from mlg.m5.memory import count_tokens
from mlg.m5.taskgraph_extract import atomic_facts, memory_system_prompt, memory_user_prompt
from mlg.m5.taskgraph_prompt_projection import (
    PackedMemory,
    build_projected_writer_prompts,
    sanitized_writer_payload,
)
from mlg.m5.taskgraph_relation import (
    parse_dependency_decisions,
)


class CollaborationTaskGraphRun(CollaborationRunState):
    """Three-stage transaction: prewrite judgment, chapter, deterministic writeback."""

    @locked_action
    def prepare_prewrite_judgment(self, chapter_id: int) -> Path:
        self._require("idle", chapter_id, expect_next=True)
        prompt = self.dataset.release(chapter_id)
        frozen_prompt = prompt.to_dict()
        self.state["transaction"] = {
            "transaction_id": uuid.uuid4().hex,
            "chapter_prompt": frozen_prompt,
            "chapter_prompt_sha256": sha_payload(frozen_prompt),
            "actor_provenance": copy.deepcopy(self.state["actor_provenance"]),
            "artifacts": {},
        }
        target_id, audit, embedding_calls = self.memory.prepare_context_judgment(prompt)
        targets = [(target_id, audit)]
        request = self._request("pre_write", chapter_id, targets)
        path = self._judgment_path("prewrite", chapter_id, "request")
        atomic_write_json(path, request)
        self.state.update({
            "stage": "awaiting_prewrite_judgment",
            "active_chapter": chapter_id,
            "transaction": {
                **self.state["transaction"],
                "target_id": target_id,
                "audit": audit,
                "embedding_calls": embedding_calls,
                "prewrite_request": self._relative(path),
                "prewrite_request_sha256": sha_file(path),
                "candidate_frozen_at": now_utc(),
            },
        })
        self.state["transaction"]["artifacts"]["prewrite_request"] = self._artifact(path)
        self._save()
        return path

    @locked_action
    def apply_prewrite_judgment(self, chapter_id: int, response_path: Path) -> Path:
        self._require("awaiting_prewrite_judgment", chapter_id)
        self._capture_actor("prewrite_judge")
        prompt = self._frozen_prompt(chapter_id)
        tx = self.state["transaction"]
        targets = [(str(tx["target_id"]), dict(tx["audit"]))]
        supplied = read_json(response_path)
        raw, stored_payload = _bound_response(
            supplied,
            transaction_id=tx["transaction_id"],
            request_sha256=tx["prewrite_request_sha256"],
            formal=self.state["run_mode"] == "formal",
        )
        validation = _validate_dependency_response(targets, raw)
        stored_response = self._judgment_path("prewrite", chapter_id, "response")
        atomic_write_json(stored_response, stored_payload)
        decisions = parse_dependency_decisions(self.memory.graph, targets, raw)
        context = self.memory.apply_context_decisions(
            prompt, targets[0][1], decisions[targets[0][0]],
            api_calls=list(tx.get("embedding_calls", [])),
        )
        graph_path = self.run_dir / "graph" / "prewrite" / f"chapter_{chapter_id:03d}.json"
        atomic_write_json(graph_path, self.memory.graph.to_dict())
        writer_memory = PackedMemory(
            text=context.text,
            blocks=(),
            memory_tokens=int(context.metadata.get("memory_tokens", 0)),
            dropped_count=int(context.metadata.get("dropped_count", 0)),
        )
        local = load_local_continuity(self.run_dir, chapter_id)
        system, user = build_projected_writer_prompts(
            self.dataset, prompt, writer_memory, local_continuity=local,
        )
        system_tokens = count_tokens(system)
        user_tokens = count_tokens(user)
        packet = sanitized_writer_payload(prompt, system, user)
        audit_path = self.run_dir / "audits" / f"prewrite_chapter_{chapter_id:03d}.json"
        atomic_write_json(audit_path, {
            "schema": "m5-controller-prewrite-audit-v1",
            "chapter_id": chapter_id,
            "transaction_id": tx["transaction_id"],
            "target_id": tx["target_id"],
            "dependency_audit": targets[0][1],
            "memory": context.metadata,
            "local_continuity": local.audit(),
            "writer_prompt": {
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "total_tokens": system_tokens + user_tokens,
            },
            "prewrite_graph": self._relative(graph_path),
        })
        input_path = self.run_dir / "inputs" / f"chapter_{chapter_id:03d}.json"
        atomic_write_json(input_path, packet)
        tx.update({
            "prewrite_response": self._relative(stored_response),
            "prewrite_graph": self._relative(graph_path),
            "prewrite_graph_sha256": sha_file(graph_path),
            "writer_input": self._relative(input_path),
            "writer_input_sha256": sha_file(input_path),
            "selected_node_ids": context.metadata["selected_node_ids"],
            "memory_tokens": context.metadata["memory_tokens"],
            "memory_blocks": context.metadata["memory_blocks"],
            "memory_deduplicated_count": context.metadata["deduplicated_count"],
            "memory_dropped_count": context.metadata["dropped_count"],
            "local_continuity": local.audit(),
            "writer_system_tokens": system_tokens,
            "writer_user_tokens": user_tokens,
            "writer_prompt_tokens": system_tokens + user_tokens,
            "controller_audit": self._relative(audit_path),
            "prewrite_validation": validation,
            "writer_released_at": now_utc(),
        })
        tx["artifacts"].update({
            "prewrite_response": self._artifact(stored_response),
            "prewrite_graph": self._artifact(graph_path),
            "controller_audit": self._artifact(audit_path),
            "writer_input": self._artifact(input_path),
        })
        self.state["stage"] = "awaiting_chapter_text"
        self._save()
        return input_path

    @locked_action
    def submit_chapter(self, chapter_id: int, text_path: Path) -> Path:
        self._require("awaiting_chapter_text", chapter_id)
        self._capture_actor("writer")
        prompt = self._frozen_prompt(chapter_id)
        text = text_path.read_text(encoding="utf-8-sig").strip()
        count = han_char_count(text)
        if not is_mechanically_accepted(count):
            low, high = MECHANICAL_ACCEPTED_HAN_CHARS
            raise CollaborationProtocolError(
                f"Chapter {chapter_id} has {count} Han characters; "
                f"mechanically accepted {low}..{high}"
            )
        chapter_path = self.run_dir / "chapters" / f"chapter_{chapter_id:03d}.md"
        atomic_write_text(chapter_path, text + "\n")
        request_path = self.run_dir / "writeback" / f"chapter_{chapter_id:03d}_request.json"
        full_registry = self.memory.active_state_registry()
        selection = self.memory.writeback_registry(text)
        registry = list(selection.rows)
        full_registry_sha256 = sha_payload(full_registry)
        prompt_registry_sha256 = sha_payload(registry)
        atomic_write_json(request_path, {
            "schema": "m5-l4-writeback-request-v3",
            "chapter_id": chapter_id,
            "system": memory_system_prompt(registry_provided=True),
            "user": memory_user_prompt(prompt, text, registry),
            "full_registry_count": len(full_registry),
            "full_registry_sha256": full_registry_sha256,
            "prompt_registry_sha256": prompt_registry_sha256,
            "registry_audit": selection.audit(),
        })
        self.state["transaction"].update({
            "chapter_text": self._relative(chapter_path),
            "chapter_text_sha256": sha_file(chapter_path),
            "han_chars": count,
            "writeback_registry": {
                **selection.audit(),
                "full_registry_sha256": full_registry_sha256,
                "prompt_registry_sha256": prompt_registry_sha256,
                "controller_source_graph": self.state["transaction"]["prewrite_graph"],
                "controller_source_graph_sha256": self.state["transaction"][
                    "prewrite_graph_sha256"
                ],
            },
            "writeback_request": self._relative(request_path),
            "writeback_request_sha256": sha_file(request_path),
            "chapter_submitted_at": now_utc(),
        })
        self.state["transaction"]["artifacts"].update({
            "chapter_text": self._artifact(chapter_path),
            "writeback_request": self._artifact(request_path),
        })
        self.state["stage"] = "awaiting_writeback"
        self._save()
        return request_path

    @locked_action
    def submit_writeback(self, chapter_id: int, writeback_path: Path) -> Path:
        self._require("awaiting_writeback", chapter_id)
        self._capture_actor("writeback_extractor")
        self._require_formal_generation_actor_provenance()
        prompt = self._frozen_prompt(chapter_id)
        supplied = read_json(writeback_path)
        if not isinstance(supplied, dict):
            raise CollaborationProtocolError("Writeback must be a JSON object")
        tx = self.state["transaction"]
        writeback, stored_payload = _bound_response(
            supplied,
            transaction_id=tx["transaction_id"],
            request_sha256=tx["writeback_request_sha256"],
            formal=self.state["run_mode"] == "formal",
        )
        if not isinstance(writeback, dict):
            raise CollaborationProtocolError("Writeback response must be a JSON object")
        chapter_path = self.run_dir / self.state["transaction"]["chapter_text"]
        source_text = chapter_path.read_text(encoding="utf-8-sig")
        registry = self.memory.active_state_registry()
        try:
            checked_facts = atomic_facts(
                prompt, writeback, source_text=source_text,
                registered_keys={row["key"] for row in registry},
                strict=self.state["run_mode"] == "formal",
            )
        except (ValueError, RuntimeError) as exc:
            raise CollaborationProtocolError(str(exc)) from exc
        invalid_quotes = [fact["key"] for fact in checked_facts if not fact["source_quote_valid"]]
        nonexact_quotes = [
            fact["key"] for fact in checked_facts
            if fact.get("source_quote_exact") is False
        ]
        stored_writeback = self.run_dir / "writeback" / f"chapter_{chapter_id:03d}.json"
        atomic_write_json(stored_writeback, stored_payload)
        try:
            facts, targets, build_call = self.memory.prepare_observation_judgment(
                prompt, writeback, source_text=source_text,
                parsed_facts=checked_facts,
            )
        except (ValueError, RuntimeError) as exc:
            raise CollaborationProtocolError(str(exc)) from exc
        lifecycle_edges = self.memory.apply_observation_decisions(
            prompt, facts, targets, {},
        )
        graph_path = self.run_dir / "graph" / f"chapter_{chapter_id:03d}.json"
        atomic_write_json(graph_path, self.memory.graph.to_dict())
        counts = {
            operation: sum(fact["operation"] == operation for fact in checked_facts)
            for operation in ("create", "update")
        }
        self.state["transaction"].update({
            "writeback": self._relative(stored_writeback),
            "facts": facts,
            "writeback_embedding_call": build_call.to_dict(),
            "invalid_writeback_source_quotes": invalid_quotes,
            "nonexact_writeback_source_quotes": nonexact_quotes,
            "writeback_validation": {
                "strict": self.state["run_mode"] == "formal",
                "source_quote_policy": "model-grounded-nonempty-v1",
                "facts_accepted": len(checked_facts),
                "source_quotes_valid": len(checked_facts) - len(invalid_quotes),
                "source_quotes_exact": len(checked_facts) - len(nonexact_quotes),
                **counts,
            },
            "lifecycle_edges": lifecycle_edges,
            "final_graph": self._relative(graph_path),
            "writeback_submitted_at": now_utc(),
        })
        self.state["transaction"]["artifacts"].update({
            "writeback": self._artifact(stored_writeback),
            "final_graph": self._artifact(graph_path),
        })
        return self._commit_validated_writeback(chapter_id)

    def _require_formal_generation_actor_provenance(self) -> None:
        if self.state.get("run_mode") != "formal":
            return
        actors = self.state.get("transaction", {}).get("actor_provenance", {})
        missing = []
        for role in ("prewrite_judge", "writer", "writeback_extractor"):
            row = actors.get(role, {})
            for field in ("model", "reasoning_effort", "agent_id"):
                if not row.get(field):
                    missing.append(f"{role}.{field}")
        if missing:
            raise CollaborationProtocolError(
                "Formal actor provenance is incomplete: " + ", ".join(missing)
            )

    def _commit_validated_writeback(self, chapter_id: int) -> Path:
        """Persist the deterministic lifecycle result, then run the commit guard."""
        tx = self.state["transaction"]
        completed_at = now_utc()
        record = {
            "schema": "m5-collaboration-chapter-record-v5",
            "chapter_id": chapter_id,
            "condition": "taskgraph",
            "protocol": "prewrite_funnel_context_then_deterministic_l4_lifecycle",
            "run_mode": self.state["run_mode"],
            "transaction_id": tx["transaction_id"],
            "dataset_fingerprint": self.state["dataset_fingerprint"],
            "config_fingerprint": self.state["config_fingerprint"],
            **({
                "config_transition": copy.deepcopy(
                    self.state["config_transition"]
                ),
            } if self.state.get("config_transition") else {}),
            "chapter_prompt_sha256": tx["chapter_prompt_sha256"],
            "target_id": tx["target_id"],
            "han_chars": tx["han_chars"],
            "text_sha256": tx["chapter_text_sha256"],
            "selected_node_ids": tx["selected_node_ids"],
            "memory_tokens": tx["memory_tokens"],
            "memory_blocks": tx["memory_blocks"],
            "memory_deduplicated_count": tx["memory_deduplicated_count"],
            "memory_dropped_count": tx["memory_dropped_count"],
            "local_continuity": tx["local_continuity"],
            "writer_system_tokens": tx["writer_system_tokens"],
            "writer_user_tokens": tx["writer_user_tokens"],
            "writer_prompt_tokens": tx["writer_prompt_tokens"],
            "controller_audit": tx["controller_audit"],
            "prewrite_graph": tx["prewrite_graph"],
            "prewrite_graph_sha256": tx["prewrite_graph_sha256"],
            "writer_input": tx["writer_input"],
            "writer_input_sha256": tx["writer_input_sha256"],
            "prewrite_response": tx["prewrite_response"],
            "writeback_request_sha256": tx["writeback_request_sha256"],
            "writeback_registry": tx["writeback_registry"],
            "final_graph": tx["final_graph"],
            "candidate_frozen_at": tx["candidate_frozen_at"],
            "writer_released_at": tx["writer_released_at"],
            "chapter_submitted_at": tx["chapter_submitted_at"],
            "writeback_submitted_at": tx["writeback_submitted_at"],
            "external_api_calls": [
                *tx.get("embedding_calls", []), tx["writeback_embedding_call"],
            ],
            "actor_provenance": tx["actor_provenance"],
            "judgment_validation": {"prewrite": tx["prewrite_validation"]},
            "writeback_validation": tx["writeback_validation"],
            "lifecycle_edges": tx["lifecycle_edges"],
            "invalid_writeback_source_quotes": tx.get(
                "invalid_writeback_source_quotes", []
            ),
            "nonexact_writeback_source_quotes": tx.get(
                "nonexact_writeback_source_quotes", []
            ),
            "artifacts": copy.deepcopy(tx["artifacts"]),
            "completed_at": completed_at,
        }
        record_path = self.run_dir / "records" / f"chapter_{chapter_id:03d}.json"
        atomic_write_json(record_path, record)
        tx["artifacts"]["record"] = self._artifact(record_path)
        tx.update({
            "record": self._relative(record_path),
            "completed_at": completed_at,
        })
        self.state.update({"stage": "pending_commit", "active_chapter": chapter_id})
        self._save()
        return self._finalize_pending_commit(chapter_id)

    @locked_action
    def apply_postwrite_judgment(self, chapter_id: int, response_path: Path) -> Path:
        raise CollaborationProtocolError(
            "Postwrite model judgment was removed; submit_writeback commits "
            "deterministic stable-key lifecycle edges"
        )

    def _finalize_pending_commit(self, chapter_id: int) -> Path:
        if self.state.get("stage") != "pending_commit":
            raise CollaborationProtocolError("No pending commit to finalize")
        tx = self.state["transaction"]
        checkpoint = self.run_dir / "checkpoints" / f"chapter_{chapter_id:03d}.json"
        receipt = {
            "schema": "m5-collaboration-commit-receipt-v1",
            "chapter_id": chapter_id,
            "transaction_id": tx["transaction_id"],
            "run_mode": self.state["run_mode"],
            "dataset_fingerprint": self.state["dataset_fingerprint"],
            "config_fingerprint": self.state["config_fingerprint"],
            "chapter_prompt_sha256": tx["chapter_prompt_sha256"],
            "final_memory_sha256": sha_persisted_payload(self.memory.to_state()),
            "artifacts": {
                key: value for key, value in tx["artifacts"].items()
                if key != "checkpoint"
            },
            "completed_at": tx["completed_at"],
        }
        atomic_write_json(checkpoint, receipt)
        tx["artifacts"]["checkpoint"] = self._artifact(checkpoint)
        self._save()

        from mlg.m5.collab_guard import guard_collaboration_commit

        result = guard_collaboration_commit(self.run_dir, chapter_id)
        if not result["ok"]:
            raise CollaborationProtocolError(
                "Commit guard rejected chapter: " + ", ".join(result["errors"])
            )
        self.state.setdefault("commit_ledger", {})[str(chapter_id)] = {
            "transaction_id": tx["transaction_id"],
            "record": tx["artifacts"]["record"],
            "final_graph": tx["artifacts"]["final_graph"],
            "checkpoint": tx["artifacts"]["checkpoint"],
            "completed_at": tx["completed_at"],
        }
        self.state.update({
            "stage": "idle", "last_completed": chapter_id,
            "active_chapter": None, "transaction": {},
        })
        self._save()
        return checkpoint

    @locked_action
    def repair_pending_commit(self, chapter_id: int | None = None) -> Path:
        if self.state.get("stage") == "idle":
            completed = int(self.state.get("last_completed", 0))
            if completed == 0:
                raise CollaborationProtocolError(
                    "Cannot repair an unstarted collaboration run"
                )
            if chapter_id not in (None, completed):
                raise CollaborationProtocolError("Requested chapter is not last committed")
            return self.run_dir / "checkpoints" / f"chapter_{completed:03d}.json"
        if self.state.get("stage") != "pending_commit":
            raise CollaborationProtocolError(
                f"Cannot repair stage {self.state.get('stage')}"
            )
        active = int(self.state.get("active_chapter") or -1)
        if chapter_id not in (None, active):
            raise CollaborationProtocolError("Requested chapter is not pending")
        return self._finalize_pending_commit(active)


_DEPENDENCY_TYPES = {
    "state_continuity", "entity_continuity", "constraint", "causal_support",
    "long_range_support", "plot_support", "contextual_relevance",
}


def _bound_response(
    supplied: dict[str, Any],
    *,
    transaction_id: str,
    request_sha256: str,
    formal: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind formal judge output to the exact transaction and request bytes."""
    wrapped = isinstance(supplied.get("response"), dict)
    if not wrapped:
        if formal:
            raise CollaborationProtocolError(
                "Formal response must include transaction_id, "
                "request_sha256, and response"
            )
        return supplied, supplied
    if not formal and "transaction_id" not in supplied and "request_sha256" not in supplied:
        return supplied["response"], supplied
    if supplied.get("transaction_id") != transaction_id:
        raise CollaborationProtocolError("Dependency response transaction mismatch")
    if supplied.get("request_sha256") != request_sha256:
        raise CollaborationProtocolError("Dependency response request hash mismatch")
    return supplied["response"], supplied


def _validate_dependency_response(
    targets: list[tuple[str, dict[str, Any]]], raw: dict[str, Any],
) -> dict[str, int]:
    """Reject partial or silently lossy judge output before graph mutation."""
    returned = raw.get("targets")
    if not isinstance(returned, list):
        raise CollaborationProtocolError("Dependency response must contain targets array")
    expected = {target_id: set(audit["Cfinal"]) for target_id, audit in targets}
    seen_targets: set[str] = set()
    dependency_count = 0
    for target in returned:
        if not isinstance(target, dict):
            raise CollaborationProtocolError("Dependency target row must be an object")
        target_id = str(target.get("target_id", ""))
        if target_id not in expected:
            raise CollaborationProtocolError(f"Unknown dependency target: {target_id}")
        if target_id in seen_targets:
            raise CollaborationProtocolError(f"Duplicate dependency target: {target_id}")
        seen_targets.add(target_id)
        rows = target.get("dependencies")
        if not isinstance(rows, list):
            raise CollaborationProtocolError(
                f"Dependencies must be an array for {target_id}"
            )
        seen_sources: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise CollaborationProtocolError("Dependency row must be an object")
            source_id = str(row.get("source_id", ""))
            if source_id not in expected[target_id]:
                raise CollaborationProtocolError(
                    f"Dependency source outside frozen candidates: {source_id}"
                )
            if source_id in seen_sources:
                raise CollaborationProtocolError(
                    f"Duplicate dependency source for {target_id}: {source_id}"
                )
            seen_sources.add(source_id)
            relation = str(row.get("dependency_type", ""))
            if relation not in _DEPENDENCY_TYPES:
                raise CollaborationProtocolError(
                    f"Invalid dependency_type for {source_id}: {relation}"
                )
            confidence = row.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                raise CollaborationProtocolError(f"Invalid confidence for {source_id}")
            if not 0.0 <= float(confidence) <= 1.0:
                raise CollaborationProtocolError(f"Confidence out of range: {source_id}")
            priority = row.get("priority")
            if not isinstance(priority, int) or isinstance(priority, bool) or priority < 1:
                raise CollaborationProtocolError(f"Invalid priority for {source_id}")
            dependency_count += 1
    missing = sorted(set(expected) - seen_targets)
    if missing:
        raise CollaborationProtocolError(
            "Dependency response omitted targets: " + ", ".join(missing)
        )
    return {
        "targets_expected": len(expected),
        "targets_returned": len(returned),
        "dependencies_returned": dependency_count,
        "invalid_rows": 0,
    }
