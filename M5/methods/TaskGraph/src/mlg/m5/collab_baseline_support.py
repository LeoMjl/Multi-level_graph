from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mlg.m5.collab_state import CollaborationProtocolError, now_utc
from mlg.m5.io import atomic_write_json
from mlg.m5.memory import CurrentOnlyMemory, RecencyWindowMemory, RunningSummaryMemory


def baseline_protocol_fingerprint() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = (
        Path(__file__),
        Path(__file__).with_name("collab_baseline.py"),
        Path(__file__).with_name("codex_cli_writer.py"),
        Path(__file__).with_name("codex_formal_isolation.py"),
        Path(__file__).with_name("continuity.py"),
        Path(__file__).with_name("dataset.py"),
        Path(__file__).with_name("length_policy.py"),
        Path(__file__).with_name("memory.py"),
        Path(__file__).with_name("prompts.py"),
        root / "tools" / "run_m5_baseline_collab.py",
        root / "tools" / "set_m5_codex_isolation_acl.ps1",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_protocol_unchanged(state: dict[str, Any]) -> None:
    if state.get("protocol_fingerprint") != baseline_protocol_fingerprint():
        raise CollaborationProtocolError(
            "M5 collaboration protocol source changed during the run"
        )


def new_baseline_memory(condition: str, token_budget: int):
    if condition == "current_only":
        return CurrentOnlyMemory(token_budget=token_budget)
    if condition == "recency_window":
        return RecencyWindowMemory(token_budget=token_budget)
    if condition == "running_summary":
        return RunningSummaryMemory(token_budget=token_budget)
    raise ValueError(f"Unknown collaboration baseline: {condition}")


def require_stage(
    state: dict[str, Any], stage: str, chapter_id: int, *, expect_next: bool = False,
) -> None:
    if state.get("stage") != stage:
        raise CollaborationProtocolError(
            f"Expected stage {stage}, found {state.get('stage')}"
        )
    expected = int(state.get("last_completed", 0)) + 1
    active = int(state.get("active_chapter") or -1)
    if (expect_next and chapter_id != expected) or (
        not expect_next and chapter_id != active
    ):
        raise CollaborationProtocolError(
            f"Chapter {chapter_id} violates sequential transaction state"
        )


def validate_state(
    state: dict[str, Any], *, schema: str, condition: str, replicate: int,
    fingerprint: str,
) -> None:
    expected = {
        "schema": schema,
        "condition": condition,
        "replicate": replicate,
        "dataset_fingerprint": fingerprint,
        "protocol_fingerprint": baseline_protocol_fingerprint(),
    }
    changed = [key for key, value in expected.items() if state.get(key) != value]
    if changed:
        raise CollaborationProtocolError(
            f"Incompatible collaboration baseline state: {', '.join(changed)}"
        )


def save_state(
    run_dir: Path,
    state: dict[str, Any],
    memory: Any,
    *,
    model: str,
    reasoning_effort: str,
    token_budget: int,
    run_mode: str,
    isolation_policy: str | None,
    isolation_evidence_sha256: str | None,
) -> None:
    state["memory"] = memory.to_state()
    state["updated_at"] = now_utc()
    atomic_write_json(run_dir / "state.json", state)
    atomic_write_json(run_dir / "manifest.json", {
        "schema": "m5-collaboration-baseline-manifest-v1",
        "condition": state["condition"],
        "replicate": state["replicate"],
        "model": model,
        "reasoning_effort": reasoning_effort,
        "context_token_budget": token_budget,
        "run_mode": run_mode,
        "isolation_policy": isolation_policy,
        "isolation_evidence_sha256": isolation_evidence_sha256,
        "writer_backend": "codex_cli_ephemeral_per_call",
        "codex_session_reuse": False,
        "future_prompt_access": False,
        "hidden_gold_access": False,
        "dataset_fingerprint": state["dataset_fingerprint"],
        "protocol_fingerprint": state["protocol_fingerprint"],
        "stage": state["stage"],
        "last_completed": state["last_completed"],
        "active_chapter": state.get("active_chapter"),
        "updated_at": state["updated_at"],
    })


def summary_request(memory: RunningSummaryMemory, chapter_id: int, text: str) -> dict[str, Any]:
    return {
        "schema": "m5-running-summary-request-v1",
        "chapter_id": chapter_id,
        "system": (
            "你是长篇小说连续性记忆器。只依据既有摘要和刚完成正文更新摘要。"
            "保留人物知识边界、时间地点、物品位置/数量/损耗、承诺、未决线索和因果；"
            "不得预测未来。只输出JSON，字段summary。"
        ),
        "user": f"既有摘要：\n{memory.summary or '无'}\n\n第{chapter_id}章正文：\n{text}",
    }
