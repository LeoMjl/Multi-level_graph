from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Any, Sequence


PROFILE_NAME = "m5_formal_actor"
MIN_CODEX_VERSION = (0, 153, 4)
_SCRUB_ENV = (
    "CODEX_PERMISSION_PROFILE",
    "CODEX_SESSION_ID",
    "CODEX_THREAD_ID",
    "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
)
_SECRET_ENV_MARKERS = (
    "API_KEY", "ACCESS_KEY", "AUTH_TOKEN", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN",
)
_HISTORY_DIRS = (
    "sessions", "archived_sessions", "memories", "attachments", "automations",
    "log_backups", "sqlite", "process_manager", "visualizations",
    "dictation-history", "ambient-suggestions", "browser/sessions",
)
_HISTORY_DATABASES = (
    "state_5.sqlite", "logs_2.sqlite", "thread_history_1.sqlite",
    "queue_1.sqlite", "memories_1.sqlite", "goals_1.sqlite",
)
_HISTORY_FILES = (
    "history.jsonl", "session_index.jsonl", "transcription-history.jsonl",
    ".codex-global-state.json", ".codex-global-state.json.bak",
    ".codex-global-state.json.bak.bak",
)
def codex_history_root() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).resolve()


def codex_history_profile_paths() -> tuple[Path, ...]:
    """Keep the profile stable without materializing transient SQLite sidecars."""
    root = codex_history_root()
    names = (*_HISTORY_DIRS, ".sandbox", *_HISTORY_FILES)
    # Missing deny targets are materialized as directories by the Windows sandbox
    # setup helper. WAL/SHM names remain covered by the root-level history guard.
    return (root, *(root / name for name in (*names, *_HISTORY_DATABASES)))


def codex_history_probe_files() -> tuple[Path, ...]:
    root = codex_history_root()
    return tuple(
        root / name for name in (*_HISTORY_DATABASES, *_HISTORY_FILES, ".sandbox/sandbox.log")
        if (root / name).is_file()
    )


class FormalIsolationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FormalIsolationEvidence:
    codex_version: str
    implementation: str
    actor_root: str
    repository_root: str
    profile_sha256: str
    probe_identity: str
    forbidden_probe_count: int
    acl_probe_count: int
    verified_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "m5-formal-isolation-evidence-v1",
            **self.__dict__,
            "probe_passed": True,
        }

    def policy_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("verified_at", None)
        return payload

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.policy_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def permission_profile_config_args() -> tuple[str, ...]:
    history_denies = ",".join(
        json.dumps(path.as_posix()) + '="deny"' for path in codex_history_profile_paths()
    )
    filesystem = (
        '{":root"="deny",":minimal"="read",'
        '":workspace_roots"={"."="read"},'
        '":tmpdir"="deny",":slash_tmp"="deny",' + history_denies + '}'
    )
    return (
        "-c", f"permissions.{PROFILE_NAME}.filesystem={filesystem}",
        "-c", f"permissions.{PROFILE_NAME}.network.enabled=false",
    )


def formal_global_policy_args() -> tuple[str, ...]:
    """Options accepted only before the ``exec`` subcommand."""
    return ("-a", "never")


def formal_exec_policy_args() -> tuple[str, ...]:
    return (
        "--disable", "shell_tool",
        "--disable", "unified_exec",
        "--disable", "multi_agent",
        "--strict-config",
        "-c", 'windows.sandbox="elevated"',
        "-c", f'default_permissions="{PROFILE_NAME}"',
        *permission_profile_config_args(),
    )


def sanitized_codex_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in _SCRUB_ENV:
        env.pop(name, None)
    for name in tuple(env):
        if any(marker in name.upper() for marker in _SECRET_ENV_MARKERS):
            env.pop(name, None)
    env["CODEX_CI"] = "1"
    env["NO_COLOR"] = "1"
    return env


def formal_isolation_preflight(
    codex_executable: str,
    *,
    repository_root: Path,
    actor_root: Path,
    forbidden_files: Sequence[Path],
) -> FormalIsolationEvidence:
    repository = repository_root.resolve(strict=True)
    actor = actor_root.resolve()
    if _contains(repository, actor) or _contains(actor, repository):
        raise FormalIsolationError("Actor root must be physically outside the repository")
    actor.mkdir(parents=True, exist_ok=True)
    _reject_reparse_points(actor)
    forbidden = [path.resolve(strict=True) for path in forbidden_files]
    history_probes = codex_history_probe_files()
    persistent_acl_probes = [*forbidden, *history_probes]
    forbidden.extend(history_probes)
    if not forbidden or any(_contains(actor, path) for path in forbidden):
        raise FormalIsolationError("Forbidden probes must exist outside the actor root")

    # App metadata is atomically replaced; verify its effective denial below
    # without requiring a transient per-file ACL to survive that replacement.
    acl_probe_count = _verify_acl_denies(persistent_acl_probes)
    env = sanitized_codex_env()
    raw_version = subprocess.check_output(
        [codex_executable, "--version"], text=True, encoding="utf-8", env=env,
    ).strip()
    version = _parse_version(raw_version)
    if version < MIN_CODEX_VERSION:
        raise FormalIsolationError(
            f"Codex {version} is too old for permission profiles; need {MIN_CODEX_VERSION}"
        )

    probe_dir = actor / f"preflight_{uuid.uuid4().hex}"
    probe_dir.mkdir()
    canary = probe_dir / "readable_canary.txt"
    canary.write_text("m5-isolation-canary", encoding="utf-8")
    script = _probe_script(canary, forbidden)
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    command = [
        codex_executable, "sandbox", "-P", PROFILE_NAME,
        *permission_profile_config_args(),
        "-C", str(probe_dir),
        "powershell.exe", "-NoProfile", "-NonInteractive",
        "-EncodedCommand", encoded,
    ]
    completed = subprocess.run(
        command, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        check=False, env=env, cwd=probe_dir,
    )
    payload = _last_json_object(completed.stdout or "")
    if completed.returncode != 0 or payload is None:
        raise FormalIsolationError("Codex sandbox probe failed or returned ambiguous output")
    identity = str(payload.get("identity", "")).casefold()
    denied = payload.get("forbidden_readable")
    if not identity.endswith("\\codexsandboxoffline"):
        raise FormalIsolationError(f"Expected elevated offline sandbox identity, found {identity}")
    if payload.get("inside_readable") is not True:
        raise FormalIsolationError("Sandbox cannot read its isolated actor root")
    if not isinstance(denied, list) or len(denied) != len(forbidden) or any(denied):
        raise FormalIsolationError("At least one evaluator-only repository probe is readable")

    profile = "\0".join((*formal_global_policy_args(), *formal_exec_policy_args()))
    return FormalIsolationEvidence(
        codex_version=raw_version,
        implementation="windows_elevated_acl_and_permission_profile",
        actor_root=str(actor),
        repository_root=str(repository),
        profile_sha256=hashlib.sha256(profile.encode("utf-8")).hexdigest(),
        probe_identity=identity,
        forbidden_probe_count=len(forbidden),
        acl_probe_count=acl_probe_count,
        verified_at=datetime.now(timezone.utc).isoformat(),
    )


def _probe_script(canary: Path, forbidden: Sequence[Path]) -> str:
    items = ",".join(_ps_quote(str(path)) for path in forbidden)
    return (
        "$ErrorActionPreference='SilentlyContinue';"
        "function CanRead([string]$p){try{$s=[IO.File]::OpenRead($p);$s.Dispose();return $true}catch{return $false}};"
        f"$forbidden=@({items});"
        f"$result=[ordered]@{{identity=(& whoami);inside_readable=(CanRead {_ps_quote(str(canary))});"
        "forbidden_readable=@($forbidden|ForEach-Object{CanRead $_})};"
        "$result|ConvertTo-Json -Compress"
    )


def _verify_acl_denies(forbidden: Sequence[Path]) -> int:
    paths = ",".join(_ps_quote(str(path)) for path in forbidden)
    script = (
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false);"
        "$ErrorActionPreference='Stop';"
        "$names=@('CodexSandboxOffline','CodexSandboxOnline');"
        "$sids=@($names|ForEach-Object{(Get-LocalUser -Name $_).SID.Value});"
        f"$paths=@({paths});"
        "$results=@($paths|ForEach-Object{"
        "$path=$_;$deny=@((Get-Acl -LiteralPath $path).Access|"
        "Where-Object{$_.AccessControlType -eq 'Deny'}|ForEach-Object{"
        "try{$_.IdentityReference.Translate("
        "[Security.Principal.SecurityIdentifier]).Value}catch{}});"
        "$missing=@($sids|Where-Object{$deny -notcontains $_});"
        "[ordered]@{path=$path;ok=($missing.Count -eq 0)}});"
        "[ordered]@{sids=$sids;results=$results}|ConvertTo-Json -Compress -Depth 4"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    shell = which("pwsh.exe") or which("powershell.exe")
    if not shell:
        raise FormalIsolationError("PowerShell is required to verify Windows ACLs")
    completed = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60, check=False,
    )
    payload = _last_json_object(completed.stdout or "")
    results = payload.get("results") if payload else None
    if (
        completed.returncode != 0
        or not isinstance(results, list)
        or len(results) != len(forbidden)
        or any(row.get("ok") is not True for row in results if isinstance(row, dict))
        or any(not isinstance(row, dict) for row in results)
    ):
        raise FormalIsolationError(
            "M5 sandbox deny ACLs are missing; run set_m5_codex_isolation_acl.ps1 Apply"
        )
    return len(results)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _last_json_object(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_version(text: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        raise FormalIsolationError(f"Cannot parse Codex version: {text}")
    return tuple(int(item) for item in match.groups())


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _reject_reparse_points(path: Path) -> None:
    reparse = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for candidate in (path, *path.parents):
        attribute = getattr(os.stat(candidate), "st_file_attributes", 0)
        if attribute & reparse:
            raise FormalIsolationError(
                f"Actor root cannot traverse a reparse point: {candidate}"
            )
