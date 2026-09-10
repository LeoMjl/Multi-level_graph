from __future__ import annotations

import hashlib
import subprocess
import time
import uuid
from datetime import datetime, timezone
from shutil import which
from pathlib import Path
from typing import Any

from mlg.m5.codex_formal_isolation import (
    FormalIsolationEvidence,
    formal_exec_policy_args,
    formal_global_policy_args,
    sanitized_codex_env,
)
from mlg.m5.io import atomic_write_text
from mlg.m5.memory import count_tokens


class CodexCliInvocationError(RuntimeError):
    """A failed Codex process with a persistable per-invocation audit row."""

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


class CodexCliWriter:
    """Fresh ephemeral Codex process for each controlled writer call."""

    def __init__(
        self,
        work_dir: Path,
        *,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "medium",
        timeout_seconds: float = 900.0,
        isolation_evidence: FormalIsolationEvidence | None = None,
    ) -> None:
        self.work_dir = work_dir.resolve()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.isolation_evidence = isolation_evidence
        self.isolation_label = (
            "windows_elevated_acl_permission_profile_v1"
            if isolation_evidence else "fresh_ephemeral_read_only_unverified"
        )
        if isolation_evidence and self.work_dir != Path(
            isolation_evidence.actor_root
        ).resolve():
            raise ValueError("Formal writer work_dir must match verified actor_root")
        self.codex_executable = which("codex.cmd") or which("codex")
        if not self.codex_executable:
            raise RuntimeError("Codex CLI executable was not found")
        self.codex_version = subprocess.check_output(
            [self.codex_executable, "--version"], text=True, encoding="utf-8",
            env=sanitized_codex_env(),
        ).strip()

    def generate(
        self,
        *,
        system: str,
        user: str,
        purpose: str,
        output_path: Path,
        attempt: int,
        invocation_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        invocation_id = invocation_id or uuid.uuid4().hex
        actor_dir = self.work_dir / purpose.replace("/", "_") / invocation_id
        actor_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_path.parent / (
            f"{output_path.stem}.attempt_{attempt:02d}.{invocation_id}.log"
        )
        prompt = self._envelope(system, user, purpose)
        command = [
            self.codex_executable, *formal_global_policy_args(),
            "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "-m", self.model,
            "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
            "--disable", "plugins", "--disable", "apps", "--disable", "hooks",
        ]
        if self.isolation_evidence:
            command.extend([
                *formal_exec_policy_args(),
                "--disable", "browser_use",
                "--disable", "computer_use",
                "--disable", "in_app_browser",
                "-C", str(actor_dir),
            ])
        else:
            command.extend(["-s", "read-only", "-C", str(actor_dir)])
        command.extend(["-o", str(output_path), "-"])
        started = time.perf_counter()
        record: dict[str, Any] = {
            "invocation_id": invocation_id,
            "purpose": purpose,
            "attempt": attempt,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "backend": "codex_cli_ephemeral",
            "codex_version": self.codex_version,
            "session_reuse": False,
            "isolation": self.isolation_label,
            "isolation_evidence_sha256": (
                self.isolation_evidence.sha256 if self.isolation_evidence else None
            ),
            "actor_dir": str(actor_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input_tokens": count_tokens(prompt),
            "output_tokens": 0,
            "token_count_source": "offline_tiktoken_o200k_base",
            "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output": str(output_path),
            "log": str(log_path),
        }
        try:
            process = subprocess.Popen(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                env=sanitized_codex_env(),
                cwd=actor_dir,
            )
            output, _ = process.communicate(prompt, timeout=self.timeout_seconds)
            completed = subprocess.CompletedProcess(command, process.returncode, output)
        except subprocess.TimeoutExpired as exc:
            # A .cmd launcher has descendants; terminate our entire timed-out tree.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False, timeout=30,
            )
            try:
                partial, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            elapsed_ms = (time.perf_counter() - started) * 1000
            atomic_write_text(log_path, partial or "")
            text = self._read_output(output_path)
            record.update({
                "status": "timeout", "elapsed_ms": elapsed_ms, "exit_code": None,
                "output_tokens": count_tokens(text),
                "output_sha256": _text_sha256(text) if text else None,
            })
            self._finish_usage(record)
            raise CodexCliInvocationError(
                f"Codex writer timed out after {elapsed_ms / 1000:.1f}s; log={log_path}",
                record,
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        atomic_write_text(log_path, completed.stdout or "")
        if completed.returncode != 0:
            text = self._read_output(output_path)
            record.update({
                "status": "failed", "elapsed_ms": elapsed_ms,
                "exit_code": completed.returncode,
                "output_tokens": count_tokens(text),
                "output_sha256": _text_sha256(text) if text else None,
            })
            self._finish_usage(record)
            raise CodexCliInvocationError(
                f"Codex writer exited {completed.returncode}; log={log_path}", record,
            )
        if not output_path.is_file():
            record.update({
                "status": "failed", "elapsed_ms": elapsed_ms,
                "exit_code": completed.returncode, "error": "missing_output",
            })
            self._finish_usage(record)
            raise CodexCliInvocationError(
                f"Codex writer produced no output: {output_path}", record,
            )
        text = self._read_output(output_path)
        if not text:
            record.update({
                "status": "failed", "elapsed_ms": elapsed_ms,
                "exit_code": completed.returncode, "error": "empty_output",
            })
            self._finish_usage(record)
            raise CodexCliInvocationError("Codex writer produced empty output", record)
        record.update({
            "status": "succeeded", "elapsed_ms": elapsed_ms,
            "exit_code": completed.returncode, "output_tokens": count_tokens(text),
            "output_sha256": _text_sha256(text),
        })
        self._finish_usage(record)
        return text, record

    @staticmethod
    def _read_output(output_path: Path) -> str:
        if not output_path.is_file():
            return ""
        return output_path.read_text(encoding="utf-8-sig").strip()

    @staticmethod
    def _finish_usage(record: dict[str, Any]) -> None:
        record["total_tokens"] = int(record["input_tokens"]) + int(
            record.get("output_tokens") or 0
        )
        record["finished_at"] = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _envelope(system: str, user: str, purpose: str) -> str:
        json_purposes = {
            "memory", "summary", "prewrite_judge", "dependency_judge",
            "writeback_extractor",
        }
        output_rule = (
            "只输出一个JSON对象，不加代码围栏或解释。"
            if purpose in json_purposes else
            "只输出章节标题和连续正文，不加解释、计划、摘要或字数统计。"
        )
        return (
            "这是一次全新、独立的受限写作调用。不要使用工具，不要读取文件系统，"
            "不要寻找任何未提供的章节或外部资料；下列SYSTEM与USER块是本次调用唯一事实来源。"
            f"{output_rule}\n\n<SYSTEM>\n{system}\n</SYSTEM>\n\n"
            f"<USER>\n{user}\n</USER>"
        )


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
