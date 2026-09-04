from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mlg.m5.codex_cli_writer import CodexCliInvocationError, CodexCliWriter
from mlg.m5.collab_protocol import CollaborationTaskGraphRun
from mlg.m5.collab_state import CollaborationProtocolError, sha_file, sha_payload
from mlg.m5.dataset import han_char_count
from mlg.m5.io import atomic_write_json, read_json
from mlg.m5.length_policy import (
    MECHANICAL_ACCEPTED_HAN_CHARS,
    WRITER_REQUESTED_HAN_CHARS,
    is_mechanically_accepted,
)
from mlg.m5.memory import count_tokens, parse_json_object
from mlg.m5.taskgraph_config import PaperDependencyConfig


@dataclass(frozen=True)
class InvocationArtifact:
    output_path: Path
    audit_path: Path
    text: str
    payload: dict[str, Any] | None
    call: dict[str, Any]


class AttemptLedger:
    """Persist every Codex attempt and reuse a validated output after a crash."""

    def __init__(self, run_dir: Path, *, max_attempts: int) -> None:
        self.run_dir = run_dir.resolve()
        self.max_attempts = max_attempts

    def acquire(
        self,
        *,
        chapter_id: int,
        stage: str,
        transaction_id: str,
        request_sha256: str,
        writer: CodexCliWriter,
        system: str,
        user: str,
        purpose: str,
        suffix: str,
        validator: Callable[[str, dict[str, Any] | None], None],
    ) -> InvocationArtifact:
        directory = self.run_dir / "controller" / "attempts" / (
            f"chapter_{chapter_id:03d}"
        ) / stage
        directory.mkdir(parents=True, exist_ok=True)
        self._recover_pending(directory, transaction_id, request_sha256)
        self._reject_orphan_outputs(directory)
        cached = self._cached(directory, transaction_id, request_sha256, validator)
        if cached is not None:
            return cached
        prior_error = self._last_error(directory)
        while True:
            attempt = self._next_attempt(directory)
            if attempt > self.max_attempts:
                raise CollaborationProtocolError(
                    f"{stage} exhausted {self.max_attempts} fixed attempts"
                )
            output_path = directory / f"attempt_{attempt:02d}{suffix}"
            audit_path = directory / f"audit_{attempt:02d}.json"
            call_user = user + self._retry_feedback(prior_error)
            invocation_id = uuid.uuid4().hex
            call: dict[str, Any] = self._pending_call(
                writer, system, call_user, purpose, attempt, invocation_id,
            )
            self._write_pending_audit(
                audit_path, chapter_id, stage, transaction_id, request_sha256,
                output_path, call,
            )
            try:
                text, call = writer.generate(
                    system=system, user=call_user, purpose=purpose,
                    output_path=output_path, attempt=attempt,
                    invocation_id=invocation_id,
                )
                if call.get("invocation_id") != invocation_id:
                    raise RuntimeError("Codex invocation_id changed inside one attempt")
                payload = parse_json_object(text) if suffix == ".json" else None
                validator(text, payload)
            except CodexCliInvocationError as exc:
                prior_error = str(exc)
                self._write_audit(
                    audit_path, chapter_id, stage, transaction_id, request_sha256,
                    output_path, exc.record, False, prior_error,
                )
                continue
            except Exception as exc:
                prior_error = f"{type(exc).__name__}: {exc}"
                if call.get("status") == "pending":
                    call.update({
                        "status": "failed",
                        "failure_kind": "controller_exception_before_response",
                    })
                self._write_audit(
                    audit_path, chapter_id, stage, transaction_id, request_sha256,
                    output_path, call, False, prior_error,
                )
                continue
            self._write_audit(
                audit_path, chapter_id, stage, transaction_id, request_sha256,
                output_path, call, True, None,
            )
            return InvocationArtifact(output_path, audit_path, text, payload, call)

    def invalidate(self, artifact: InvocationArtifact, error: Exception) -> None:
        audit = read_json(artifact.audit_path)
        audit.update({
            "accepted": False,
            "protocol_error": f"{type(error).__name__}: {error}",
        })
        atomic_write_json(artifact.audit_path, audit)

    def invocations(self, artifact: InvocationArtifact) -> list[dict[str, Any]]:
        anchor = read_json(artifact.audit_path)
        rows: list[dict[str, Any]] = []
        for path in sorted(artifact.audit_path.parent.glob("audit_*.json")):
            audit = read_json(path)
            if audit.get("transaction_id") != anchor.get("transaction_id"):
                continue
            if audit.get("request_sha256") != anchor.get("request_sha256"):
                continue
            call = audit.get("call")
            if isinstance(call, dict) and call.get("invocation_id"):
                rows.append(dict(call))
        return rows

    def artifact_rows(self, artifact: InvocationArtifact) -> list[dict[str, Any]]:
        anchor = read_json(artifact.audit_path)
        rows: list[dict[str, Any]] = []
        for path in sorted(artifact.audit_path.parent.glob("audit_*.json")):
            audit = read_json(path)
            if audit.get("transaction_id") != anchor.get("transaction_id"):
                continue
            if audit.get("request_sha256") != anchor.get("request_sha256"):
                continue
            output_path = self.run_dir / str(audit["output_path"])
            call = audit.get("call") if isinstance(audit.get("call"), dict) else {}
            rows.append({
                "attempt": int(call.get("attempt") or path.stem.rsplit("_", 1)[-1]),
                "accepted": bool(audit.get("accepted")),
                "invocation_id": call.get("invocation_id"),
                "audit": {
                    "path": path.resolve().relative_to(self.run_dir).as_posix(),
                    "sha256": sha_file(path),
                },
                "output": ({
                    "path": output_path.resolve().relative_to(self.run_dir).as_posix(),
                    "sha256": sha_file(output_path),
                } if output_path.is_file() else None),
            })
        return rows

    def _recover_pending(
        self, directory: Path, transaction_id: str, request_sha256: str,
    ) -> None:
        """Conservatively charge an interrupted invocation before any retry."""
        for path in sorted(directory.glob("audit_*.json")):
            audit = read_json(path)
            if audit.get("audit_status") != "pending":
                continue
            if (
                audit.get("transaction_id") != transaction_id
                or audit.get("request_sha256") != request_sha256
            ):
                raise CollaborationProtocolError(
                    f"Pending attempt belongs to another transaction: {path}"
                )
            output_path = self.run_dir / str(audit["output_path"])
            text = (
                output_path.read_text(encoding="utf-8-sig").strip()
                if output_path.is_file() else ""
            )
            call = dict(audit.get("call") or {})
            call.update({
                "status": "failed",
                "failure_kind": "controller_crash_recovered",
                "output_tokens": count_tokens(text),
                "elapsed_ms": float(call.get("elapsed_ms") or 0),
            })
            call["total_tokens"] = int(call.get("input_tokens") or 0) + int(
                call["output_tokens"]
            )
            audit.update({
                "audit_status": "finalized", "accepted": False,
                "error": "Recovered unresolved pending invocation conservatively",
                "output_sha256": sha_file(output_path) if output_path.is_file() else None,
                "call": call,
            })
            atomic_write_json(path, audit)

    def _reject_orphan_outputs(self, directory: Path) -> None:
        referenced = {
            str(read_json(path).get("output_path"))
            for path in directory.glob("audit_*.json")
        }
        for pattern in ("attempt_*.json", "attempt_*.md"):
            for path in directory.glob(pattern):
                relative = path.resolve().relative_to(self.run_dir).as_posix()
                if relative not in referenced:
                    raise CollaborationProtocolError(
                        f"Orphan Codex output requires explicit audit repair: {path}"
                    )

    @staticmethod
    def _pending_call(
        writer: CodexCliWriter, system: str, user: str, purpose: str,
        attempt: int, invocation_id: str,
    ) -> dict[str, Any]:
        prompt = CodexCliWriter._envelope(system, user, purpose)
        return {
            "invocation_id": invocation_id,
            "purpose": purpose,
            "attempt": attempt,
            "model": writer.model,
            "reasoning_effort": writer.reasoning_effort,
            "backend": "codex_cli_ephemeral",
            "codex_version": getattr(writer, "codex_version", None),
            "session_reuse": False,
            "isolation": getattr(
                writer, "isolation_label", "fresh_ephemeral_read_only"
            ),
            "isolation_evidence_sha256": (
                getattr(getattr(writer, "isolation_evidence", None), "sha256", None)
            ),
            "status": "pending",
            "input_tokens": count_tokens(prompt),
            "output_tokens": 0,
            "total_tokens": count_tokens(prompt),
            "elapsed_ms": 0.0,
            "token_count_source": "offline_tiktoken_o200k_base",
        }

    def _write_pending_audit(
        self, path: Path, chapter_id: int, stage: str, transaction_id: str,
        request_sha256: str, output_path: Path, call: dict[str, Any],
    ) -> None:
        atomic_write_json(path, {
            "schema": "m5-codex-cli-attempt-v1",
            "audit_status": "pending",
            "chapter_id": chapter_id,
            "stage": stage,
            "transaction_id": transaction_id,
            "request_sha256": request_sha256,
            "accepted": False,
            "output_path": output_path.resolve().relative_to(self.run_dir).as_posix(),
            "output_sha256": None,
            "call": call,
        })

    def _cached(
        self,
        directory: Path,
        transaction_id: str,
        request_sha256: str,
        validator: Callable[[str, dict[str, Any] | None], None],
    ) -> InvocationArtifact | None:
        for audit_path in sorted(directory.glob("audit_*.json"), reverse=True):
            audit = read_json(audit_path)
            if not audit.get("accepted"):
                continue
            if audit.get("transaction_id") != transaction_id:
                continue
            if audit.get("request_sha256") != request_sha256:
                continue
            output_path = self.run_dir / str(audit["output_path"])
            if not output_path.is_file() or sha_file(output_path) != audit["output_sha256"]:
                raise CollaborationProtocolError(
                    f"Accepted attempt output drifted: {audit_path}"
                )
            text = output_path.read_text(encoding="utf-8-sig").strip()
            payload = parse_json_object(text) if output_path.suffix == ".json" else None
            try:
                validator(text, payload)
            except Exception as exc:
                raise CollaborationProtocolError(
                    f"Accepted attempt no longer validates: {audit_path}"
                ) from exc
            return InvocationArtifact(
                output_path, audit_path, text, payload, dict(audit["call"]),
            )
        return None

    @staticmethod
    def _next_attempt(directory: Path) -> int:
        numbers = []
        for path in directory.glob("audit_*.json"):
            try:
                numbers.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                pass
        return max(numbers, default=0) + 1

    @staticmethod
    def _last_error(directory: Path) -> str | None:
        paths = sorted(directory.glob("audit_*.json"))
        if not paths:
            return None
        row = read_json(paths[-1])
        return row.get("protocol_error") or row.get("error")

    @staticmethod
    def _retry_feedback(error: str | None) -> str:
        if not error:
            return ""
        if "Han characters" in error and "mechanically accepted" in error:
            low, high = WRITER_REQUESTED_HAN_CHARS
            return (
                "\n\n机械校验反馈：上一章正文长度未通过。请保持情节要求不变并"
                f"完整重写，正文仍严格控制在{low}—{high}个汉字。"
            )
        if "update requires a registered key:" in error:
            remainder = error.split("update requires a registered key:", 1)[1]
            remainder = remainder.splitlines()[0].strip()[:2000]
            key = remainder.split(";", 1)[0].strip()[:500]
            diagnostic = ""
            if ";" in remainder:
                diagnostic = (
                    "机械层给出的精确登记近邻仅供核对，不代表自动判定："
                    + remainder.split(";", 1)[1].strip()
                    + "。"
                )
            return (
                "\n\n机械校验反馈：上一输出把未登记key "
                f"{json.dumps(key, ensure_ascii=False)} 错写成了update，必须完整重做。"
                + diagnostic
                + "operation=update时，key与prior_key必须逐字复制STATE_KEY_REGISTRY中"
                "同一个已有key；若正文事实确属全新状态槽，则使用operation=create且"
                "prior_key=null。不得再次用未登记key执行update，也不得把相似含义"
                "自行改写成另一个key。系统不会替你猜测或自动修复key。"
            )
        if "create cannot reuse registered key:" in error:
            key = error.split("create cannot reuse registered key:", 1)[1]
            key = key.splitlines()[0].strip()[:500]
            return (
                "\n\n机械校验反馈：上一输出把已登记key "
                f"{json.dumps(key, ensure_ascii=False)} 错写成了create，必须完整重做。"
                "若正文是在更新该既有状态槽，请使用operation=update，并让key与"
                "prior_key逐字等于STATE_KEY_REGISTRY中的同一key；只有全新状态槽"
                "才能使用operation=create且prior_key=null。系统不会自动修复key。"
            )
        lifecycle_markers = (
            "create prior_key must be null:",
            "create must not provide prior_key:",
            "update prior_key must exactly match registered key:",
        )
        if any(marker in error for marker in lifecycle_markers):
            return (
                "\n\n机械校验反馈：上一输出未通过状态生命周期校验，必须完整重做。"
                "create只用于全新状态槽且prior_key必须为null；update只用于"
                "STATE_KEY_REGISTRY中的既有状态槽，key与prior_key必须逐字等于"
                "登记的同一个key。不得猜测、缩写或改写key。原错误为："
                + error[:2000]
            )
        return (
            "\n\n机械校验反馈：上一输出未通过，必须完整重做。错误为："
            + error[:12000]
        )

    def _write_audit(
        self, path: Path, chapter_id: int, stage: str, transaction_id: str,
        request_sha256: str, output_path: Path, call: dict[str, Any],
        accepted: bool, error: str | None,
    ) -> None:
        payload = {
            "schema": "m5-codex-cli-attempt-v1", "chapter_id": chapter_id,
            "audit_status": "finalized",
            "stage": stage, "transaction_id": transaction_id,
            "request_sha256": request_sha256, "accepted": accepted,
            "output_path": output_path.resolve().relative_to(self.run_dir).as_posix(),
            "output_sha256": sha_file(output_path) if output_path.is_file() else None,
            "call": call,
        }
        if error:
            payload["error"] = error
        atomic_write_json(path, payload)


class TaskGraphCodexController:
    """Drive the persisted protocol one stage at a time with fresh CLI calls."""

    def __init__(
        self,
        m5_root: Path,
        run_dir: Path,
        embeddings,
        *,
        judge_writer: CodexCliWriter,
        chapter_writer: CodexCliWriter,
        extractor_writer: CodexCliWriter,
        run_mode: str = "formal",
        retries: int = 5,
        token_budget: int = 12000,
        dependency_config: PaperDependencyConfig | None = None,
        protocol_class=CollaborationTaskGraphRun,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be non-negative")
        if run_mode == "formal" and retries != 5:
            raise ValueError("Formal v14 runs require exactly 5 retries (6 attempts)")
        configured_dependency = dependency_config or PaperDependencyConfig()
        default_dependency = PaperDependencyConfig()
        if run_mode == "formal" and (
            configured_dependency != default_dependency or token_budget != 12000
        ):
            raise ValueError(
                "Formal v14 runs require the preregistered dependency and token policies"
            )
        self.m5_root = m5_root.resolve()
        self.run_dir = run_dir.resolve()
        self.embeddings = embeddings
        self.run_mode = run_mode
        self.token_budget = token_budget
        self.dependency_config = configured_dependency
        self.protocol_class = protocol_class
        self.writers = {
            "prewrite_judge": judge_writer,
            "writer": chapter_writer,
            "writeback_extractor": extractor_writer,
        }
        isolation_labels = {
            getattr(writer, "isolation_label", "fresh_ephemeral_read_only")
            for writer in self.writers.values()
        }
        if len(isolation_labels) != 1:
            raise ValueError("All controller actors must use one isolation policy")
        isolation = next(iter(isolation_labels))
        evidence = getattr(chapter_writer, "isolation_evidence", None)
        physical = isolation == "windows_elevated_acl_permission_profile_v1"
        if physical and any(
            getattr(writer, "isolation_evidence", None) is None
            or writer.isolation_evidence.sha256 != evidence.sha256
            for writer in self.writers.values()
        ):
            raise ValueError("Physical-isolation evidence differs across actors")
        self.controller_policy = {
            "schema": (
                "m5-taskgraph-controller-policy-v2" if physical
                else "m5-taskgraph-controller-policy-v1"
            ),
            "controller_revision": "m5-taskgraph-codex-cli-controller-v3",
            "backend": "codex_cli_ephemeral",
            "isolation": isolation,
            "max_attempts": retries + 1,
            "roles": {
                name: {
                    "model": writer.model,
                    "reasoning_effort": writer.reasoning_effort,
                    "codex_version": getattr(writer, "codex_version", None),
                    "timeout_seconds": getattr(writer, "timeout_seconds", None),
                    **({"isolation_evidence_sha256": evidence.sha256} if physical else {}),
                }
                for name, writer in self.writers.items()
            },
            **({
                "isolation_evidence": evidence.policy_dict(),
                "isolation_evidence_sha256": evidence.sha256,
            } if physical else {}),
        }
        self.ledger = AttemptLedger(self.run_dir, max_attempts=retries + 1)
        self._state_existed = (self.run_dir / "state.json").is_file()
        self._run_dir_was_nonempty = (
            self.run_dir.is_dir() and any(self.run_dir.iterdir())
        )

    def run_to(
        self,
        chapter_end: int,
        *,
        fresh: bool,
        force_unlock: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if fresh and self._run_dir_was_nonempty:
            raise CollaborationProtocolError(
                "run requires an empty --run-dir; use resume for existing state"
            )
        if not fresh and not self._state_existed:
            raise CollaborationProtocolError(
                "resume requires an existing state.json; use run for a fresh directory"
            )
        with self._controller_lock(force_unlock=force_unlock):
            run = self._protocol()
            self._ensure_controller_config(run.state)
            if not 1 <= chapter_end <= run.dataset.total_chapters:
                raise CollaborationProtocolError(
                    f"chapter_end must be within 1..{run.dataset.total_chapters}"
                )
            while int(run.state["last_completed"]) < chapter_end:
                before = self._state_signature(run.state)
                previous_completed = int(run.state["last_completed"])
                self._advance_one(run.state)
                run = self._protocol()
                if self._state_signature(run.state) == before:
                    raise CollaborationProtocolError(
                        "Controller made no durable state-machine progress"
                    )
                if progress and int(run.state["last_completed"]) > previous_completed:
                    progress(self.status(run.state))
            return run.state

    @staticmethod
    def status(state: dict[str, Any]) -> dict[str, Any]:
        tx = state.get("transaction", {})
        return {
            "run_mode": state.get("run_mode"),
            "stage": state.get("stage"),
            "last_completed": state.get("last_completed"),
            "active_chapter": state.get("active_chapter"),
            "transaction_id": tx.get("transaction_id"),
        }

    def _protocol(
        self,
        role: str | None = None,
        call: dict[str, Any] | None = None,
    ) -> CollaborationTaskGraphRun:
        actors = {
            name: {
                "model": writer.model,
                "reasoning_effort": writer.reasoning_effort,
            }
            for name, writer in self.writers.items()
        }
        if role and call:
            actors[role].update({
                "agent_id": call.get("invocation_id"),
                "input_tokens": call.get("input_tokens"),
                "output_tokens": call.get("output_tokens"),
                "elapsed_ms": call.get("elapsed_ms"),
                "invocations": call.get("invocations", []),
                "attempt_artifacts": call.get("attempt_artifacts", []),
            })
        return self.protocol_class(
            self.m5_root, self.run_dir, self.embeddings,
            token_budget=self.token_budget,
            dependency_config=self.dependency_config,
            run_mode=self.run_mode,
            actor_provenance=actors,
            controller_policy=self.controller_policy,
        )

    def _ensure_controller_config(self, state: dict[str, Any]) -> None:
        stored_policy = state.get("config", {}).get("controller_policy")
        if stored_policy != self.controller_policy:
            raise CollaborationProtocolError("Controller policy changed during run")
        payload = {
            "schema": "m5-taskgraph-controller-config-v1",
            "policy": self.controller_policy,
            "policy_sha256": sha_payload(self.controller_policy),
        }
        path = self.run_dir / "controller" / "config.json"
        if path.is_file():
            if read_json(path) != payload:
                raise CollaborationProtocolError("controller/config.json drifted")
        else:
            atomic_write_json(path, payload)

    @contextmanager
    def _controller_lock(self, *, force_unlock: bool):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / ".controller.lock"
        if force_unlock:
            path.unlink(missing_ok=True)
        payload = json.dumps({"pid": os.getpid(), "mode": "codex_cli_controller"})
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise CollaborationProtocolError(
                f"Another controller owns this run: {path}"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            yield
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _state_signature(state: dict[str, Any]) -> tuple[Any, ...]:
        tx = state.get("transaction", {})
        return (
            state.get("stage"), state.get("last_completed"),
            state.get("active_chapter"), tx.get("transaction_id"),
        )

    def _advance_one(self, state: dict[str, Any]) -> None:
        stage = str(state.get("stage"))
        chapter_id = int(
            state.get("active_chapter") or int(state.get("last_completed", 0)) + 1
        )
        if stage == "idle":
            self._protocol().prepare_prewrite_judgment(chapter_id)
        elif stage == "awaiting_prewrite_judgment":
            self._apply_prewrite(chapter_id, state)
        elif stage == "awaiting_chapter_text":
            self._submit_chapter(chapter_id, state)
        elif stage == "awaiting_writeback":
            self._submit_writeback(chapter_id, state)
        elif stage == "pending_commit":
            self._protocol().repair_pending_commit(chapter_id)
        elif stage == "awaiting_postwrite_judgment":
            raise CollaborationProtocolError(
                "Obsolete postwrite-judge stage cannot be auto-run; start a fresh "
                "three-role run or migrate this pilot evidence explicitly"
            )
        else:
            raise CollaborationProtocolError(f"Unknown controller stage: {stage}")

    def _apply_prewrite(self, chapter_id: int, state: dict[str, Any]) -> None:
        tx = state["transaction"]
        request_path = self.run_dir / tx["prewrite_request"]
        request = read_json(request_path)
        transaction_id = str(tx["transaction_id"])
        request_sha256 = str(tx.get("prewrite_request_sha256") or sha_file(request_path))
        while True:
            artifact = self.ledger.acquire(
                chapter_id=chapter_id, stage="prewrite_judge",
                transaction_id=transaction_id, request_sha256=request_sha256,
                writer=self.writers["prewrite_judge"],
                system=str(request["system"]),
                user=json.dumps(request["payload"], ensure_ascii=False, indent=2),
                purpose="prewrite_judge", suffix=".json",
                validator=self._validate_dependency_json,
            )
            response_path = self._bound_response(
                chapter_id, "prewrite", transaction_id, request_sha256,
                artifact.payload or {},
            )
            judge_run = self._protocol(
                "prewrite_judge", self._actor_call(artifact),
            )
            try:
                judge_run.apply_prewrite_judgment(chapter_id, response_path)
                return
            except CollaborationProtocolError as exc:
                if not self._same_stage("awaiting_prewrite_judgment", transaction_id):
                    raise
                self.ledger.invalidate(artifact, exc)

    def _submit_chapter(self, chapter_id: int, state: dict[str, Any]) -> None:
        tx = state["transaction"]
        packet_path = self.run_dir / tx["writer_input"]
        packet = read_json(packet_path)
        artifact = self.ledger.acquire(
            chapter_id=chapter_id, stage="writer",
            transaction_id=str(tx["transaction_id"]),
            request_sha256=str(tx.get("writer_input_sha256") or sha_file(packet_path)),
            writer=self.writers["writer"],
            system=str(packet["system"]), user=str(packet["user"]),
            purpose="chapter", suffix=".md", validator=self._validate_chapter,
        )
        self._protocol("writer", self._actor_call(artifact)).submit_chapter(
            chapter_id, artifact.output_path,
        )

    def _submit_writeback(self, chapter_id: int, state: dict[str, Any]) -> None:
        tx = state["transaction"]
        request_path = self.run_dir / tx["writeback_request"]
        request = read_json(request_path)
        transaction_id = str(tx["transaction_id"])
        request_sha256 = str(
            tx.get("writeback_request_sha256") or sha_file(request_path)
        )
        while True:
            artifact = self.ledger.acquire(
                chapter_id=chapter_id, stage="writeback_extractor",
                transaction_id=transaction_id, request_sha256=request_sha256,
                writer=self.writers["writeback_extractor"],
                system=str(request["system"]), user=str(request["user"]),
                purpose="writeback_extractor", suffix=".json",
                validator=self._validate_writeback_json,
            )
            normalized = self._bound_response(
                chapter_id, "writeback", transaction_id, request_sha256,
                artifact.payload or {},
            )
            extractor_run = self._protocol(
                "writeback_extractor", self._actor_call(artifact),
            )
            try:
                extractor_run.submit_writeback(chapter_id, normalized)
                return
            except CollaborationProtocolError as exc:
                current = read_json(self.run_dir / "state.json")
                if current.get("stage") == "pending_commit":
                    return
                if not self._same_stage("awaiting_writeback", transaction_id):
                    raise
                self.ledger.invalidate(artifact, exc)

    def _bound_response(
        self, chapter_id: int, label: str, transaction_id: str,
        request_sha256: str, payload: dict[str, Any],
    ) -> Path:
        path = self._bound_directory(chapter_id) / f"{label}.json"
        atomic_write_json(path, {
            "transaction_id": transaction_id,
            "request_sha256": request_sha256,
            "response": payload,
        })
        return path

    def _bound_directory(self, chapter_id: int) -> Path:
        path = self.run_dir / "controller" / "bound" / f"chapter_{chapter_id:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _same_stage(self, stage: str, transaction_id: str) -> bool:
        current = read_json(self.run_dir / "state.json")
        return (
            current.get("stage") == stage
            and current.get("transaction", {}).get("transaction_id") == transaction_id
        )

    def _actor_call(self, artifact: InvocationArtifact) -> dict[str, Any]:
        invocations = self.ledger.invocations(artifact)
        return {
            **artifact.call,
            "input_tokens": sum(int(row.get("input_tokens") or 0) for row in invocations),
            "output_tokens": sum(
                int(row.get("output_tokens") or 0) for row in invocations
            ),
            "elapsed_ms": sum(float(row.get("elapsed_ms") or 0) for row in invocations),
            "invocations": invocations,
            "attempt_artifacts": self.ledger.artifact_rows(artifact),
        }

    @staticmethod
    def _validate_dependency_json(
        text: str, payload: dict[str, Any] | None,
    ) -> None:
        if not payload or not isinstance(payload.get("targets"), list):
            raise ValueError("dependency response requires a targets array")

    @staticmethod
    def _validate_chapter(text: str, payload: dict[str, Any] | None) -> None:
        count = han_char_count(text)
        if not is_mechanically_accepted(count):
            low, high = MECHANICAL_ACCEPTED_HAN_CHARS
            raise ValueError(
                f"chapter has {count} Han characters; mechanically accepted {low}..{high}"
            )

    @staticmethod
    def _validate_writeback_json(
        text: str, payload: dict[str, Any] | None,
    ) -> None:
        if not payload or not str(payload.get("summary", "")).strip():
            raise ValueError("writeback requires a non-empty summary")
        facts = payload.get("facts")
        if not isinstance(facts, list):
            raise ValueError("writeback requires a facts array")
        if any(not isinstance(row, dict) for row in facts):
            raise ValueError("every writeback fact must be an object")
