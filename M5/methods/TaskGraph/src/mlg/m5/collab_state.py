from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from mlg.m5.dataset import M5Dataset
from mlg.m5.io import atomic_write_json, read_json
from mlg.m5.length_policy import (
    CURRENT_LENGTH_POLICY_REVISION,
    chapter_length_policy_payload,
)
from mlg.m5.rag_memory import EmbeddingBackend
from mlg.m5.taskgraph_config import PaperDependencyConfig
from mlg.m5.taskgraph_extract import (
    LIFECYCLE_KEY_SUGGESTION_LIMIT,
    LIFECYCLE_KEY_SUGGESTION_MIN_RATIO,
    MAX_EXTRACTED_L4_FACTS,
)
from mlg.m5.taskgraph_novel import TaskGraphNovelMemory
from mlg.m5.taskgraph_relation import (
    dependency_judge_system_prompt,
    dependency_request_payload,
)


class CollaborationProtocolError(RuntimeError):
    pass


class _ForbiddenRelationshipBackend:
    model = "external-codex-collaboration-only"

    def generate(self, system: str, user: str, *, purpose: str = "chapter"):
        raise CollaborationProtocolError(
            f"The collaboration protocol must externalize model call: {purpose}"
        )


class CollaborationRunState:
    schema = "m5-collaboration-taskgraph-v5"
    legacy_schemas = {
        "m5-collaboration-taskgraph-v3", "m5-collaboration-taskgraph-v4",
    }
    protocol_revision = CURRENT_LENGTH_POLICY_REVISION

    def __init__(
        self,
        m5_root: Path,
        run_dir: Path,
        embeddings: EmbeddingBackend,
        *,
        token_budget: int = 12000,
        dependency_config: PaperDependencyConfig | None = None,
        run_mode: str | None = None,
        actor_provenance: dict[str, dict[str, Any]] | None = None,
        controller_policy: dict[str, Any] | None = None,
    ) -> None:
        if run_mode not in (None, "pilot", "formal"):
            raise ValueError("run_mode must be pilot or formal")
        config = dependency_config or PaperDependencyConfig()
        if run_mode == "formal" and (
            config != PaperDependencyConfig() or token_budget != 12000
        ):
            raise ValueError(
                "Formal v14 runs require the preregistered dependency and token policies"
            )
        self.dataset = M5Dataset(m5_root)
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.memory = TaskGraphNovelMemory(
            embeddings, _ForbiddenRelationshipBackend(), token_budget=token_budget,
            dependency_config=config, volume_briefs=self.dataset.volume_briefs(),
        )
        self.state_path = self.run_dir / "state.json"
        self._state_existed_at_open = self.state_path.is_file()
        self.dataset_fingerprint = self.dataset.public_fingerprint()
        self.config_payload = {
            "protocol_revision": self.protocol_revision,
            "memory_schema": self.memory.state_schema,
            "embedding_model": embeddings.model,
            "embedding_backend": _embedding_backend_payload(embeddings),
            "token_budget": token_budget,
            "dependency_config": asdict(config),
            "chapter_length_policy": chapter_length_policy_payload(),
            "writeback_policy": {
                "schema": "m5-l4-writeback-policy-v3",
                "min_atomic_facts": 1,
                "max_atomic_facts": MAX_EXTRACTED_L4_FACTS,
                "chapter_summary_l4": 1,
                "source_quote": "model_grounded_nonempty",
                "exact_match": "audit_only",
                "retry_feedback": "typed_lifecycle_and_source_evidence-v2",
                "lifecycle_key_suggestions": {
                    "method": "difflib_sequence_ratio-v1",
                    "min_ratio": LIFECYCLE_KEY_SUGGESTION_MIN_RATIO,
                    "limit": LIFECYCLE_KEY_SUGGESTION_LIMIT,
                    "role": "diagnostic_only",
                },
                "auto_repair": False,
            },
            "controller_policy": controller_policy or {
                "schema": "m5-external-manual-policy-v1",
                "backend": "external_manual",
                "max_attempts": None,
                "isolation": None,
                "controller_revision": None,
            },
            "algorithm_sources_sha256": _algorithm_sources_fingerprint(),
        }
        self.config_fingerprint = sha_payload(self.config_payload)
        self.requested_actors = _normalized_actors(actor_provenance)
        with self.run_lock():
            if self.state_path.is_file():
                self.state = read_json(self.state_path)
                migrated = self._validate_or_migrate(run_mode)
                self._validate_idle_commit_closure()
                self.memory.load_state(self.state["memory"])
                actors_changed = self._merge_requested_actors()
                if migrated:
                    self._save()
                elif actors_changed:
                    self._save()
            else:
                self.state = {
                    "schema": self.schema,
                    "stage": "idle",
                    "last_completed": 0,
                    "active_chapter": None,
                    "transaction": {},
                    "commit_ledger": {},
                    "run_mode": run_mode or "pilot",
                    "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config_fingerprint,
                "config": self.config_payload,
                "actor_provenance": _run_actor_config(self.requested_actors),
                }
                self._save()

    def _validate_or_migrate(self, requested_mode: str | None) -> bool:
        migrated = False
        stored_schema = self.state.get("schema")
        if stored_schema in self.legacy_schemas:
            raise CollaborationProtocolError(
                "Legacy v3/v4 collaboration state is read-only pilot evidence; "
                "start a new run directory"
            )
        elif stored_schema != self.schema:
            raise CollaborationProtocolError("Incompatible collaboration state")
        stored_mode = self.state.get("run_mode")
        if requested_mode is not None and requested_mode != stored_mode:
            raise CollaborationProtocolError(
                f"Cannot change run_mode from {stored_mode} to {requested_mode}"
            )
        mismatches = []
        if sha_payload(self.state.get("config")) != self.state.get(
            "config_fingerprint"
        ):
            mismatches.append("stored_config_fingerprint")
        if self.state.get("dataset_fingerprint") != self.dataset_fingerprint:
            mismatches.append("dataset_fingerprint")
        if self.state.get("config_fingerprint") != self.config_fingerprint:
            mismatches.append("config_fingerprint")
        if mismatches:
            raise CollaborationProtocolError(
                "Run inputs/config changed: " + ", ".join(mismatches)
            )
        self._validate_config_history()
        return migrated

    def _validate_config_history(self) -> None:
        history = self.state.get("config_history")
        transition = self.state.get("config_transition")
        if history is None and transition is None:
            return
        if not isinstance(history, dict) or not isinstance(transition, dict):
            raise CollaborationProtocolError("Config history is incomplete")
        for fingerprint, row in history.items():
            config = row.get("config") if isinstance(row, dict) else None
            chapter_range = row.get("chapter_range") if isinstance(row, dict) else None
            if (
                not isinstance(config, dict)
                or sha_payload(config) != fingerprint
                or not isinstance(chapter_range, list)
                or len(chapter_range) != 2
                or not all(isinstance(value, int) for value in chapter_range)
                or chapter_range[0] < 1
                or chapter_range[0] > chapter_range[1]
            ):
                raise CollaborationProtocolError("Config history is invalid")
        if (
            transition.get("schema")
            != "m5-v13-qwen-embedding-continuation-v1"
            or transition.get("source_config_fingerprint") not in history
            or transition.get("target_config_fingerprint")
            != self.state.get("config_fingerprint")
            or transition.get("target_embedding_model")
            != self.state.get("config", {}).get("embedding_model")
        ):
            raise CollaborationProtocolError("Config transition is invalid")
        for field in ("migration_plan", "audit_report"):
            relative = transition.get(field)
            if not isinstance(relative, str):
                raise CollaborationProtocolError(f"Config transition lacks {field}")
            path = (self.run_dir / relative).resolve()
            try:
                path.relative_to(self.run_dir)
            except ValueError as exc:
                raise CollaborationProtocolError(
                    f"Config transition {field} escapes run directory"
                ) from exc
            if not path.is_file():
                raise CollaborationProtocolError(f"Config transition lacks {field}")
        if sha_file(self.run_dir / transition["migration_plan"]) != transition.get(
            "migration_plan_sha256"
        ):
            raise CollaborationProtocolError("Config transition plan hash mismatch")

    def _merge_requested_actors(self) -> bool:
        stored = self.state.setdefault("actor_provenance", _normalized_actors(None))
        changed = False
        for role, requested in self.requested_actors.items():
            target = stored.setdefault(role, {})
            for key in ("model", "reasoning_effort"):
                value = requested.get(key)
                if value is None:
                    continue
                current = target.get(key)
                if current not in (None, value):
                    raise CollaborationProtocolError(
                        f"Actor provenance changed for {role}.{key}"
                    )
                if current != value:
                    target[key] = value
                    changed = True
        return changed

    def _validate_idle_commit_closure(self) -> None:
        errors = committed_memory_closure_errors(self.run_dir, self.state)
        if errors:
            raise CollaborationProtocolError(
                "Committed memory checkpoint mismatch: " + ", ".join(errors)
            )

    def _capture_actor(self, role: str) -> None:
        """Capture per-chapter agent and usage while keeping model config fixed."""
        tx = self.state.get("transaction", {})
        actors = tx.setdefault("actor_provenance", {})
        fixed = self.state.get("actor_provenance", {}).get(role, {})
        requested = self.requested_actors.get(role, {})
        row = {**fixed, **{
            key: value for key, value in requested.items() if value is not None
        }}
        actors[role] = row

    def _require_formal_actor_provenance(self) -> None:
        if self.state.get("run_mode") != "formal":
            return
        actors = self.state.get("transaction", {}).get("actor_provenance", {})
        missing = []
        for role in (
            "prewrite_judge", "writer", "writeback_extractor",
        ):
            row = actors.get(role, {})
            for field in ("model", "reasoning_effort", "agent_id"):
                if not row.get(field):
                    missing.append(f"{role}.{field}")
        if missing:
            raise CollaborationProtocolError(
                "Formal actor provenance is incomplete: " + ", ".join(missing)
            )

    def _save(self) -> None:
        self.state["memory"] = self.memory.to_state()
        self.state["updated_at"] = now_utc()
        atomic_write_json(self.state_path, self.state)
        manifest_path = self.run_dir.parent.parent / "manifest.json"
        policy = self.config_payload.get("controller_policy", {})
        evidence = policy.get("isolation_evidence", {})
        physical_isolation = (
            policy.get("schema") == "m5-taskgraph-controller-policy-v2"
            and evidence.get("probe_passed") is True
        )
        manifest = read_json(manifest_path) if manifest_path.is_file() else {
            "schema": "m5-collaborative-taskgraph-run-v5",
            "condition": "taskgraph",
            "chapter_range": [1, self.dataset.total_chapters],
            "replicates": 1,
            "dependency_method": self.memory.dependency_builder.method,
            "mainline_edges": False,
            "embedding_backend": self.config_payload["embedding_backend"]["provider"],
            "embedding_model": self.memory.embeddings.model,
            "embedding_dimensions": self.config_payload["embedding_backend"][
                "dimensions"
            ],
            "external_api_scope": (
                "Released L2-L4 node representations and current target only; "
                "no chapter full text, story bible, future prompts, or hidden gold."
            ),
        }
        manifest.update({
            "status": self.state["stage"],
            "last_completed": self.state["last_completed"],
            "active_chapter": self.state.get("active_chapter"),
            "run_mode": self.state["run_mode"],
            "dataset_fingerprint": self.state["dataset_fingerprint"],
            "config_fingerprint": self.state["config_fingerprint"],
            "actor_provenance": self.state.get("actor_provenance", {}),
            "writer_packet_isolation": (
                "sanitized_payload_plus_verified_filesystem_isolation"
                if physical_isolation else "sanitized_payload_only"
            ),
            "filesystem_blindness": (
                "verified_windows_native_sandbox_read_isolation"
                if physical_isolation else "operational_not_cryptographic"
            ),
            "isolation_evidence_sha256": (
                policy.get("isolation_evidence_sha256") if physical_isolation else None
            ),
            "updated_at": self.state["updated_at"],
        })
        atomic_write_json(manifest_path, manifest)

    def _reload(self) -> None:
        if not self.state_path.is_file():
            raise CollaborationProtocolError("Collaboration state disappeared")
        self.state = read_json(self.state_path)
        current_dataset = M5Dataset(self.dataset.root).public_fingerprint()
        runtime_config = dict(self.config_payload)
        runtime_config["algorithm_sources_sha256"] = _algorithm_sources_fingerprint()
        if current_dataset != self.state.get("dataset_fingerprint"):
            raise CollaborationProtocolError("Dataset changed during resumable run")
        if sha_payload(runtime_config) != self.state.get("config_fingerprint"):
            raise CollaborationProtocolError("Algorithm/config changed during resumable run")
        self._validate_or_migrate(None)
        self._validate_idle_commit_closure()
        self.memory.load_state(self.state["memory"])

    @contextmanager
    def run_lock(self):
        """Serialize one short local mutation; stale locks require explicit repair."""
        lock_path = self.run_dir / ".collaboration.lock"
        payload = json.dumps({
            "pid": os.getpid(), "created_at": now_utc(), "nonce": uuid.uuid4().hex,
        })
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            age = max(0.0, time.time() - lock_path.stat().st_mtime)
            raise CollaborationProtocolError(
                f"Run is locked by another process ({lock_path}, age={age:.1f}s)"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _frozen_prompt(self, chapter_id: int):
        from mlg.m5.dataset import ChapterPrompt

        tx = self.state.get("transaction", {})
        raw = tx.get("chapter_prompt")
        if not isinstance(raw, dict) or int(raw.get("chapter_id", -1)) != chapter_id:
            raise CollaborationProtocolError("Frozen chapter prompt is missing")
        if tx.get("chapter_prompt_sha256") != sha_payload(raw):
            raise CollaborationProtocolError("Frozen chapter prompt hash mismatch")
        return ChapterPrompt.from_dict(raw)

    def _artifact(self, path: Path) -> dict[str, str]:
        return {"path": self._relative(path), "sha256": sha_file(path)}

    def _require(self, stage: str, chapter_id: int, *, expect_next: bool = False) -> None:
        if self.state["stage"] != stage:
            raise CollaborationProtocolError(
                f"Expected stage {stage}, found {self.state['stage']}"
            )
        expected = int(self.state["last_completed"]) + 1
        active = self.state.get("active_chapter")
        if (expect_next and chapter_id != expected) or (
            not expect_next and int(active or -1) != chapter_id
        ):
            raise CollaborationProtocolError(
                f"Chapter {chapter_id} violates sequential transaction state"
            )

    def _request(
        self, phase: str, chapter_id: int, targets,
        *, payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": "m5-dependency-judgment-request-v1",
            "phase": phase,
            "chapter_id": chapter_id,
            "transaction_id": self.state.get("transaction", {}).get("transaction_id"),
            "chapter_prompt_sha256": self.state.get("transaction", {}).get(
                "chapter_prompt_sha256"
            ),
            "created_at": now_utc(),
            "funnel_formula": "Cfinal=rank_cap(Vc_union_Vr_union_Cdep)",
            "system": dependency_judge_system_prompt(),
            "payload": payload or dependency_request_payload(self.memory.graph, targets),
        }

    def _judgment_path(self, phase: str, chapter_id: int, kind: str) -> Path:
        return self.run_dir / "judgments" / f"{phase}_chapter_{chapter_id:03d}_{kind}.json"

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.run_dir).as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha_persisted_payload(payload: Any) -> str:
    """Hash the JSON value observed after a write/read persistence boundary."""
    persisted = json.loads(json.dumps(payload, ensure_ascii=False))
    return sha_payload(persisted)


def committed_memory_closure_errors(
    run_dir: Path, state: dict[str, Any],
) -> list[str]:
    """Bind an idle run's memory to the latest committed checkpoint receipt."""
    if state.get("stage") != "idle":
        return []
    completed = int(state.get("last_completed", 0) or 0)
    if completed == 0:
        return []
    ledger = state.get("commit_ledger")
    if not isinstance(ledger, dict):
        return ["idle_commit_ledger_missing"]
    try:
        ledger_chapters = {int(key) for key in ledger}
    except (TypeError, ValueError):
        return ["idle_commit_ledger_keys"]
    if ledger_chapters != set(range(1, completed + 1)):
        return ["idle_commit_ledger_latest"]
    row = ledger.get(str(completed))
    descriptor = row.get("checkpoint") if isinstance(row, dict) else None
    if not isinstance(descriptor, dict):
        return ["idle_checkpoint_descriptor"]
    try:
        checkpoint = (run_dir / str(descriptor["path"])).resolve()
        checkpoint.relative_to(run_dir.resolve())
    except (KeyError, ValueError):
        return ["idle_checkpoint_path"]
    if not checkpoint.is_file():
        return ["idle_checkpoint_missing"]
    if descriptor.get("sha256") != sha_file(checkpoint):
        return ["idle_checkpoint_sha256"]
    try:
        receipt = read_json(checkpoint)
    except (OSError, ValueError):
        return ["idle_checkpoint_json"]
    if not isinstance(receipt, dict):
        return ["idle_checkpoint_json"]
    errors: list[str] = []
    if receipt.get("schema") != "m5-collaboration-commit-receipt-v1":
        errors.append("idle_checkpoint_schema")
    if receipt.get("chapter_id") != completed:
        errors.append("idle_checkpoint_chapter")
    if not isinstance(row, dict) or receipt.get("transaction_id") != row.get(
        "transaction_id"
    ):
        errors.append("idle_checkpoint_transaction")
    if receipt.get("final_memory_sha256") != sha_persisted_payload(
        state.get("memory")
    ):
        errors.append("idle_memory_checkpoint_mismatch")
    return errors


def locked_action(method):
    """Reload after acquiring the run lock so stale processes cannot overwrite state."""
    @wraps(method)
    def wrapper(self: CollaborationRunState, *args, **kwargs):
        with self.run_lock():
            self._reload()
            return method(self, *args, **kwargs)

    return wrapper


def _normalized_actors(
    supplied: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    roles = (
        "prewrite_judge", "writer", "writeback_extractor",
    )
    result: dict[str, dict[str, Any]] = {}
    supplied = supplied or {}
    for role in roles:
        raw = supplied.get(role, {})
        result[role] = {
            "model": raw.get("model"),
            "reasoning_effort": raw.get("reasoning_effort"),
            "agent_id": raw.get("agent_id"),
            "input_tokens": raw.get("input_tokens"),
            "output_tokens": raw.get("output_tokens"),
            "elapsed_ms": raw.get("elapsed_ms"),
            "invocations": list(raw.get("invocations") or []),
            "attempt_artifacts": list(raw.get("attempt_artifacts") or []),
        }
    return result


def _run_actor_config(
    actors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        role: {
            "model": row.get("model"),
            "reasoning_effort": row.get("reasoning_effort"),
            "agent_id": None,
            "input_tokens": None,
            "output_tokens": None,
            "elapsed_ms": None,
            "invocations": [],
            "attempt_artifacts": [],
        }
        for role, row in actors.items()
    }


def _embedding_backend_payload(embeddings: EmbeddingBackend) -> dict[str, Any]:
    dimensions = getattr(embeddings, "dimensions", None)
    base_url = getattr(embeddings, "base_url", None)
    return {
        "provider": str(
            getattr(embeddings, "provider", embeddings.__class__.__name__)
        ),
        "model": embeddings.model,
        "dimensions": int(dimensions) if dimensions is not None else None,
        "base_url": str(base_url).rstrip("/") if base_url else None,
    }


def _algorithm_sources_fingerprint() -> str:
    directory = Path(__file__).resolve().parent
    repository = directory.parents[2]
    names = (
        "codex_cli_writer.py", "codex_formal_isolation.py", "collab_guard.py",
        "collab_protocol.py", "continuity.py",
        "collab_state.py", "dataset.py", "length_policy.py",
        "dashscope_embedding.py", "openrouter_embedding.py", "prompts.py",
        "taskgraph_checkpoint.py", "taskgraph_cli_controller.py", "taskgraph_config.py",
        "taskgraph_dependency.py", "taskgraph_extract.py", "taskgraph_novel.py",
        "taskgraph_prompt_projection.py", "taskgraph_relation.py",
        "backend.py", "memory.py", "rag_memory.py", "io.py",
    )
    paths = [directory / name for name in names]
    paths.extend([
        repository / "src" / "mlg" / "graph" / "__init__.py",
        repository / "src" / "mlg" / "graph" / "task_graph.py",
        repository / "tools" / "run_m5_taskgraph_collab.py",
        repository / "tools" / "set_m5_codex_isolation_acl.ps1",
    ])
    digest = hashlib.sha256()
    for path in paths:
        label = path.relative_to(repository).as_posix()
        digest.update(label.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()
