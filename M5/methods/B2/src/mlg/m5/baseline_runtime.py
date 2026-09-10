"""Durable, bounded invocation retries shared by the five method snapshots."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

from mlg.m5.codex_cli_writer import CodexCliInvocationError
from mlg.m5.collab_state import CollaborationProtocolError
from mlg.m5.dataset import han_char_count
from mlg.m5.io import atomic_write_json, read_json
from mlg.m5.memory import parse_json_object


def valid_memory_payload(request: dict, payload: dict) -> bool:
    keys = (
        ("chapter_summary", "volume_summary")
        if request.get("schema") == "m5-hierarchical-summary-request-v1"
        else ("summary",)
    )
    return all(isinstance(payload.get(k), str) and payload[k].strip() for k in keys)


def validated_generation(run, writer, args, chapter_id: int, request: dict, kind: str):
    chapter = kind == "writer"
    purpose, suffix = ("chapter", ".md") if chapter else ("memory", ".json")
    calls = []
    user = str(request["user"])
    for attempt in range(1, args.retries + 2):
        output = run.run_dir / "attempts" / (
            f"chapter_{chapter_id:03d}_{kind}_attempt_{attempt}{suffix}"
        )
        receipt = output.with_suffix(output.suffix + ".call.json")
        if receipt.exists():
            entry = read_json(receipt)
            call = entry.get("call", {"attempt": attempt, "purpose": purpose})
            text = output.read_text(encoding="utf-8-sig").strip() if output.exists() else ""
            if entry.get("status") == "accepted":
                if hashlib.sha256(text.encode("utf-8")).hexdigest() != call["output_sha256"]:
                    raise CollaborationProtocolError(f"Attempt output changed: {output}")
                calls.append(call)
                return output, calls
            calls.append(call)
            if entry.get("feedback"):
                user = str(request["user"]) + entry["feedback"]
            continue
        atomic_write_json(receipt, {
            "status": "started", "call": {
                "attempt": attempt, "purpose": purpose, "status": "interrupted_or_pending",
            },
        })
        try:
            text, call = writer.generate(
                system=str(request["system"]), user=user, purpose=purpose,
                output_path=output, attempt=attempt,
            )
        except CodexCliInvocationError as exc:
            calls.append(exc.record)
            atomic_write_json(receipt, {
                "status": "failed", "call": exc.record, "error": str(exc),
            })
            print(json.dumps({"chapter_id": chapter_id, "purpose": purpose,
                              "attempt": attempt, "error": str(exc)}), flush=True)
            # Quota/auth/configuration errors need attention, not repeated requests.
            log = Path(exc.record.get("log", ""))
            detail = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            permanent = ("usage limit", "rate limit reached", "insufficient_quota",
                         "not logged in", "unexpected argument", "invalid configuration",
                         "unknown variant", "unknown field", "model is not supported")
            if any(marker in detail.lower() for marker in permanent):
                raise
            if attempt <= args.retries:
                time.sleep(min(5 * attempt, 30))
            continue
        calls.append(call)
        if chapter:
            call["han_chars"] = han_char_count(text)
            valid = 2000 <= call["han_chars"] <= 3000
            feedback = (
                f"\n\n机械校验反馈：上次正文为{call['han_chars']}个汉字，未通过"
                "2000—3000汉字门禁。必须保持情节要求不变并完整重写；本次请控制在"
                "约2400—2600个汉字，绝对不得少于2000或超过3000。"
            )
        else:
            valid = valid_memory_payload(request, parse_json_object(text))
            feedback = (
                "\n\n上次输出缺少有效的chapter_summary或volume_summary字段；"
                "请同时给出两个非空字符串字段，并且只输出JSON。"
                if request.get("schema") == "m5-hierarchical-summary-request-v1"
                else "\n\n上次输出无有效summary字段；请只输出JSON。"
            )
        atomic_write_json(receipt, {
            "status": "accepted" if valid else "rejected",
            "call": call, "feedback": "" if valid else feedback,
        })
        if valid:
            return output, calls
        user = str(request["user"]) + feedback
    raise CollaborationProtocolError(
        f"Chapter {chapter_id} {kind} failed after {args.retries + 1} attempts; "
        "attempt receipts are preserved"
    )


@contextmanager
def exclusive_run(run_dir: Path):
    """The OS releases this lock on exit, including an interrupted controller."""
    import msvcrt

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "controller.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise CollaborationProtocolError("Another controller already owns this run") from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
