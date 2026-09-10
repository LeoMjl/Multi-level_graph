from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mlg.m5.codex_cli_writer import CodexCliWriter
from mlg.m5.baseline_runtime import exclusive_run, validated_generation
from mlg.m5.codex_formal_isolation import formal_isolation_preflight
from mlg.m5.collab_baseline import BASELINE_CONDITIONS, CollaborationBaselineRun
from mlg.m5.collab_state import CollaborationProtocolError
from mlg.m5.dataset import han_char_count
from mlg.m5.io import atomic_write_json, read_json
from mlg.m5.memory import parse_json_object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one M5 baseline with fresh ephemeral Codex writer calls",
    )
    parser.add_argument("action", choices=("run", "status", "preflight"))
    parser.add_argument(
        "--condition",
        choices=("hierarchical_summary",),
        default="hierarchical_summary",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "run")
    parser.add_argument("--text-dir", type=Path)
    parser.add_argument("--run-mode", choices=("pilot", "formal"), default="formal")
    parser.add_argument("--formal-isolation-root", type=Path)
    parser.add_argument(
        "--m5-root", type=Path, default=ROOT.parents[1],
    )
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--chapter-end", type=int, default=320)
    parser.add_argument("--token-budget", type=int, default=12000)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


def make_run(args, isolation_evidence=None) -> CollaborationBaselineRun:
    return CollaborationBaselineRun(
        args.m5_root, args.run_dir, condition=args.condition,
        replicate=args.replicate, token_budget=args.token_budget,
        model=args.model, reasoning_effort=args.reasoning_effort,
        run_mode=args.run_mode,
        text_dir=args.text_dir,
        isolation_policy=(
            "windows_elevated_acl_permission_profile_v1"
            if isolation_evidence else None
        ),
        isolation_evidence_sha256=(
            isolation_evidence.sha256 if isolation_evidence else None
        ),
        max_retries=args.retries,
    )


def writer_for(args, isolation_evidence=None) -> CodexCliWriter:
    work_dir = (
        Path(isolation_evidence.actor_root)
        if isolation_evidence else args.run_dir / "isolated_writer_workdir"
    )
    return CodexCliWriter(
        work_dir,
        model=args.model, reasoning_effort=args.reasoning_effort,
        timeout_seconds=args.timeout_seconds,
        isolation_evidence=isolation_evidence,
    )


def run_chapters(args) -> dict:
    isolation_evidence = (
        _formal_preflight(args) if args.run_mode == "formal" else None
    )
    run = make_run(args, isolation_evidence)
    if isolation_evidence:
        atomic_write_json(
            run.run_dir / "isolation_preflight.json",
            isolation_evidence.to_dict(),
        )
    if args.retries < 0 or not 1 <= args.chapter_end <= run.dataset.total_chapters:
        raise ValueError("Invalid retry count or chapter end")
    writer = writer_for(args, isolation_evidence)
    while int(run.state["last_completed"]) < args.chapter_end:
        chapter_id = int(run.state["last_completed"]) + 1
        if run.state["stage"] == "idle":
            packet_path = run.prepare(chapter_id)
        elif run.state["stage"] == "awaiting_chapter_text":
            packet_path = run.run_dir / run.state["transaction"]["writer_input"]
        elif run.state["stage"] == "awaiting_summary":
            _finish_summary(run, writer, args, chapter_id)
            _print_progress(run)
            continue
        else:
            raise CollaborationProtocolError(f"Unknown stage: {run.state['stage']}")
        packet = read_json(packet_path)
        from mlg.m5.collab_state import sha_file
        if sha_file(packet_path) != run.state["transaction"]["writer_input_sha256"]:
            raise CollaborationProtocolError("Prepared writer input changed")
        accepted, calls = validated_generation(
            run, writer, args, chapter_id, packet, "writer",
        )
        summary_request = run.submit_chapter(chapter_id, accepted, calls)
        if summary_request is not None:
            _finish_summary(run, writer, args, chapter_id)
        _print_progress(run)
    return run.state


def _finish_summary(run, writer, args, chapter_id: int) -> None:
    request_path = run.run_dir / run.state["transaction"]["summary_request"]
    request = read_json(request_path)
    from mlg.m5.collab_state import sha_file
    if sha_file(request_path) != run.state["transaction"]["summary_request_sha256"]:
        raise CollaborationProtocolError("Prepared summary input changed")
    accepted, calls = validated_generation(
        run, writer, args, chapter_id, request, "summary",
    )
    run.submit_summary(chapter_id, accepted, calls)


def _print_progress(run) -> None:
    print(json.dumps({
        "condition": run.condition,
        "replicate": run.replicate,
        "last_completed": run.state["last_completed"],
        "stage": run.state["stage"],
    }, ensure_ascii=True), flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "preflight":
        if args.run_mode != "formal":
            raise SystemExit("preflight is only defined for --run-mode formal")
        result = _formal_preflight(args).to_dict()
    elif args.action == "status":
        state_path = args.run_dir.resolve() / "state.json"
        if not state_path.is_file():
            raise SystemExit(f"No collaboration baseline state: {state_path}")
        result = read_json(state_path)
    else:
        with exclusive_run(args.run_dir.resolve()):
            result = run_chapters(args)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def _formal_preflight(args):
    codex = which("codex.cmd") or which("codex")
    if not codex:
        raise SystemExit("Codex CLI executable was not found")
    base = args.formal_isolation_root or (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "M5CodexActors"
    )
    key_source = f"baseline:{args.condition}:{args.run_dir.resolve()}"
    run_key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:20]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    from mlg.m5.io import atomic_write_text
    atomic_write_text(args.run_dir / "controller_canary.txt", "controller-only")
    return formal_isolation_preflight(
        codex,
        repository_root=args.m5_root.resolve().parent,
        actor_root=base.resolve() / run_key,
        forbidden_files=[
            args.run_dir / "controller_canary.txt",
            args.m5_root / "hooks_gold.jsonl",
            args.m5_root / "hook_schedule.jsonl",
            args.m5_root / "prompts" / "chapters_281_320.jsonl",
            ROOT / "src" / "mlg" / "m5" / "dataset.py",
        ],
    )


if __name__ == "__main__":
    main()
