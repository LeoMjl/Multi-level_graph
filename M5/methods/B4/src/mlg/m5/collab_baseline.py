from __future__ import annotations

from pathlib import Path
from typing import Any

from mlg.m5.collab_baseline_support import (
    baseline_protocol_fingerprint,
    hierarchical_summary_request,
    new_baseline_memory,
    require_protocol_unchanged,
    require_stage,
    save_state,
    summary_request,
    validate_state,
)
from mlg.m5.collab_state import CollaborationProtocolError, now_utc, sha_file
from mlg.m5.dataset import M5Dataset, han_char_count
from mlg.m5.io import atomic_write_json, atomic_write_text, read_json
from mlg.m5.memory import RunningSummaryMemory, parse_json_object, truncate_tokens
from mlg.m5.memory_extra import HierarchicalSummaryMemory
from mlg.m5.prompts import build_writer_prompts, packet_payload


BASELINE_CONDITIONS = (
    "current_only", "recency_window", "running_summary",
    "flat_vector_rag", "hierarchical_summary",
)
class CollaborationBaselineRun:
    """Sequential external-writer protocol for one API-free M5 baseline."""

    schema = "m5-collaboration-baseline-v1"

    def __init__(
        self,
        m5_root: Path,
        run_dir: Path,
        *,
        condition: str,
        replicate: int = 1,
        token_budget: int = 12000,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "medium",
        run_mode: str = "pilot",
        isolation_policy: str | None = None,
        isolation_evidence_sha256: str | None = None,
        fake_embeddings: bool = False,
        max_retries: int = 5,
    ) -> None:
        if condition not in BASELINE_CONDITIONS:
            raise ValueError(f"Unknown collaboration baseline: {condition}")
        if run_mode not in {"pilot", "formal"}:
            raise ValueError("run_mode must be pilot or formal")
        if run_mode == "formal" and (
            isolation_policy != "windows_elevated_acl_permission_profile_v1"
            or not isolation_evidence_sha256
        ):
            raise CollaborationProtocolError(
                "Formal baseline creation/open requires fresh physical-isolation evidence"
            )
        self.dataset = M5Dataset(m5_root)
        self.run_dir = run_dir.resolve()
        self.condition = condition
        self.replicate = replicate
        self.token_budget = token_budget
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.run_mode = run_mode
        self.isolation_policy = isolation_policy
        self.isolation_evidence_sha256 = isolation_evidence_sha256
        self.fake_embeddings = fake_embeddings
        self.max_retries = max_retries
        if run_mode == "formal" and fake_embeddings:
            raise CollaborationProtocolError("Formal B3 cannot use fake embeddings")
        self.state_path = self.run_dir / "state.json"
        self.memory = self._new_memory()
        if self.state_path.is_file():
            self.state = read_json(self.state_path)
            self._validate_state()
            self.memory.load_state(self.state["memory"])
        else:
            self.state: dict[str, Any] = {
                "schema": self.schema,
                "condition": condition,
                "replicate": replicate,
                "stage": "idle",
                "last_completed": 0,
                "active_chapter": None,
                "transaction": {},
                "dataset_fingerprint": self.dataset.public_fingerprint(),
                "protocol_fingerprint": baseline_protocol_fingerprint(),
                "run_mode": run_mode,
                "isolation_policy": isolation_policy,
                "isolation_evidence_sha256": isolation_evidence_sha256,
                "experiment_config": self._experiment_config(),
            }
            self._save()

    def prepare(self, chapter_id: int) -> Path:
        require_protocol_unchanged(self.state)
        self._require("idle", chapter_id, expect_next=True)
        prompt = self.dataset.release(chapter_id)
        context = self.memory.context_for(prompt)
        system, user = build_writer_prompts(self.dataset, prompt, context)
        packet = packet_payload(prompt, system, user)
        packet.update({
            "condition": self.condition,
            "replicate": self.replicate,
            "collaboration_final": True,
            "writer_backend": "codex_cli_ephemeral_per_call",
            "writer_model": self.model,
            "writer_reasoning_effort": self.reasoning_effort,
            "future_prompt_access": False,
            "hidden_gold_access": False,
            "memory_audit": {
                **context.metadata,
                "selected_chapters": list(context.selected_chapters),
            },
        })
        path = self.run_dir / "inputs" / f"chapter_{chapter_id:03d}.json"
        atomic_write_json(path, packet)
        self.state.update({
            "stage": "awaiting_chapter_text",
            "active_chapter": chapter_id,
            "transaction": {
                "writer_input": self._relative(path),
                "writer_input_sha256": sha_file(path),
                "memory_audit": packet["memory_audit"],
                "prepared_at": now_utc(),
            },
        })
        self._save()
        return path

    def submit_chapter(
        self, chapter_id: int, text_path: Path, call_records: list[dict[str, Any]],
    ) -> Path | None:
        self._require("awaiting_chapter_text", chapter_id)
        text = text_path.read_text(encoding="utf-8-sig").strip()
        count = han_char_count(text)
        if not 2000 <= count <= 3000:
            raise CollaborationProtocolError(
                f"Chapter {chapter_id} has {count} Han characters; expected 2000..3000"
            )
        chapter_path = self.run_dir / "chapters" / f"chapter_{chapter_id:03d}.md"
        atomic_write_text(chapter_path, text + "\n")
        self.state["transaction"].update({
            "chapter_text": self._relative(chapter_path),
            "chapter_text_sha256": sha_file(chapter_path),
            "han_chars": count,
            "chapter_calls": call_records,
            "chapter_submitted_at": now_utc(),
        })
        if self.condition not in {"running_summary", "hierarchical_summary"}:
            memory_results = self.memory.observe(
                self.dataset.release(chapter_id), text, None,  # type: ignore[arg-type]
            )
            self._complete(chapter_id, [item.to_dict() for item in memory_results])
            return None
        request_path = self._write_summary_request(chapter_id, text)
        self.state["stage"] = "awaiting_summary"
        self.state["transaction"]["summary_request"] = self._relative(request_path)
        self._save()
        return request_path

    def submit_summary(self, chapter_id: int, summary_path: Path,
                       call_records: list[dict[str, Any]]) -> None:
        self._require("awaiting_summary", chapter_id)
        raw = summary_path.read_text(encoding="utf-8-sig")
        payload = parse_json_object(raw)
        memory = self.memory
        stored = self.run_dir / "summaries" / f"chapter_{chapter_id:03d}.json"
        if isinstance(memory, RunningSummaryMemory):
            summary = str(payload.get("summary", "")).strip()
            if not summary:
                raise CollaborationProtocolError("Running summary response has no summary")
            memory.summary = truncate_tokens(summary, self.token_budget, keep_end=True)
            memory.covered_through = chapter_id
            stored_payload = {"summary": memory.summary}
        elif isinstance(memory, HierarchicalSummaryMemory):
            chapter_summary = str(payload.get("chapter_summary", "")).strip()
            volume_summary = str(payload.get("volume_summary", "")).strip()
            if not chapter_summary or not volume_summary:
                raise CollaborationProtocolError(
                    "Hierarchical summary response requires both summary fields"
                )
            memory.apply_update(
                self.dataset.release(chapter_id),
                chapter_summary=chapter_summary,
                volume_summary=volume_summary,
            )
            stored_payload = {
                "chapter_summary": memory.chapter_summaries[chapter_id],
                "volume_summary": memory.volume_summaries[
                    self.dataset.release(chapter_id).volume_id
                ],
            }
        else:
            raise CollaborationProtocolError("Summary submitted to a non-summary baseline")
        atomic_write_json(stored, stored_payload)
        self.state["transaction"].update({
            "summary_response": self._relative(stored),
            "summary_response_sha256": sha_file(stored),
        })
        self._complete(chapter_id, call_records)

    def _complete(self, chapter_id: int, memory_calls: list[dict[str, Any]]) -> None:
        tx = self.state["transaction"]
        atomic_write_json(self.run_dir / "records" / f"chapter_{chapter_id:03d}.json", {
            "schema": "m5-collaboration-baseline-record-v1",
            "condition": self.condition,
            "chapter_id": chapter_id,
            "han_chars": tx["han_chars"],
            "selected_chapters": tx["memory_audit"].get("selected_chapters", []),
            "memory_metadata": tx["memory_audit"],
            "api_calls": [
                *tx["memory_audit"].get("api_calls", []),
                *tx["chapter_calls"],
                *memory_calls,
            ],
            "writer_input": tx["writer_input"],
            "writer_input_sha256": tx["writer_input_sha256"],
            "chapter_text": tx["chapter_text"],
            "chapter_text_sha256": tx["chapter_text_sha256"],
            "completed_at": now_utc(),
        })
        self.state.update({
            "stage": "idle", "last_completed": chapter_id,
            "active_chapter": None, "transaction": {},
        })
        self._save()

    def _write_summary_request(self, chapter_id: int, text: str) -> Path:
        path = self.run_dir / "summary_requests" / f"chapter_{chapter_id:03d}.json"
        if isinstance(self.memory, RunningSummaryMemory):
            payload = summary_request(self.memory, chapter_id, text)
        elif isinstance(self.memory, HierarchicalSummaryMemory):
            payload = hierarchical_summary_request(
                self.memory, self.dataset.release(chapter_id), text,
            )
        else:
            raise CollaborationProtocolError("Summary request requires summary memory")
        atomic_write_json(path, payload)
        return path

    def _new_memory(self):
        return new_baseline_memory(
            self.condition, self.token_budget,
            fake_embeddings=self.fake_embeddings,
        )
    def _validate_state(self) -> None:
        validate_state(
            self.state, schema=self.schema, condition=self.condition,
            replicate=self.replicate, fingerprint=self.dataset.public_fingerprint(),
        )
        if self.state.get("run_mode") != self.run_mode:
            raise CollaborationProtocolError("Baseline run_mode changed during the run")
        if self.state.get("experiment_config") != self._experiment_config():
            raise CollaborationProtocolError("Baseline experiment configuration changed")
        if self.isolation_policy is not None and (
            self.state.get("isolation_policy") != self.isolation_policy
            or self.state.get("isolation_evidence_sha256")
            != self.isolation_evidence_sha256
        ):
            raise CollaborationProtocolError("Baseline isolation evidence changed during the run")

    def _save(self) -> None:
        save_state(
            self.run_dir, self.state, self.memory, model=self.model,
            reasoning_effort=self.reasoning_effort, token_budget=self.token_budget,
            run_mode=self.run_mode,
            isolation_policy=self.isolation_policy,
            isolation_evidence_sha256=self.isolation_evidence_sha256,
        )

    def _require(self, stage: str, chapter_id: int, *, expect_next: bool = False) -> None:
        require_stage(self.state, stage, chapter_id, expect_next=expect_next)
    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.run_dir).as_posix()

    def _experiment_config(self) -> dict[str, Any]:
        method_config = getattr(self.memory, "protocol_config", lambda: {})()
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "token_budget": self.token_budget,
            "length_gate_han_chars": [2000, 3000],
            "max_retries": self.max_retries,
            "method_config": method_config,
        }
