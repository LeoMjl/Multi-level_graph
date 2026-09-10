"""Start TaskGraph once B3 finishes; never retry a failed controller automatically."""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from launch_baselines import runtime_environment

SEQUENCE_PATH: Path | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def main() -> None:
    global SEQUENCE_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--chapter-end", type=int, default=320)
    parser.add_argument("--start-now", action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run-id must be one directory name")

    m5 = Path(__file__).resolve().parent
    artifact = args.artifact_root.resolve()
    text_root = args.text_root.resolve()
    control = artifact / "controllers" / args.run_id
    sequence_path = control / "sequence.json"
    SEQUENCE_PATH = sequence_path
    b3_run = artifact / "runs" / args.run_id / "B3"
    b3_text = text_root / "B3"
    atomic_json(sequence_path, {
        "schema": "m5-sequential-regeneration-v1",
        "status": "parallel_start_authorized" if args.start_now else "waiting_for_B3",
        "pid": __import__("os").getpid(), "run_id": args.run_id,
        "started_at": now(), "chapter_end": args.chapter_end,
    })

    while not args.start_now:
        process_path = b3_run / "process_status.json"
        state_path = b3_run / "state.json"
        if not process_path.is_file() or not state_path.is_file():
            raise RuntimeError("B3 checkpoint or process status is missing")
        process = read_json(process_path)
        state = read_json(state_path)
        if process.get("status") == "failed":
            raise RuntimeError(f"B3 failed; TaskGraph was not started: {process.get('error')}")
        if process.get("status") == "completed":
            chapter_count = sum(1 for _ in b3_text.glob("chapter_*.md"))
            if int(state.get("last_completed", 0)) != args.chapter_end or chapter_count != args.chapter_end:
                raise RuntimeError("B3 reports completion without a complete committed text set")
            break
        if not process_alive(process.get("pid")):
            raise RuntimeError("B3 controller disappeared; TaskGraph was not started")
        time.sleep(30)

    taskgraph_run = artifact / "runs" / args.run_id / "TaskGraph"
    taskgraph_text = text_root / "TaskGraph"
    taskgraph_control = control / "TaskGraph"
    worker_status = taskgraph_control / "process_status.json"
    if worker_status.is_file():
        prior = read_json(worker_status)
        if prior.get("status") == "running" and process_alive(prior.get("pid")):
            raise RuntimeError("A TaskGraph worker is already running")
        raise RuntimeError("TaskGraph already has a prior worker record; refusing automatic retry")
    if taskgraph_run.exists() and any(taskgraph_run.iterdir()):
        raise RuntimeError("TaskGraph run directory is not empty; refusing to overwrite")

    command = [
        sys.executable, "-B", str(m5 / "taskgraph_worker.py"),
        "--control-dir", str(taskgraph_control), "--text-dir", str(taskgraph_text),
        str(m5 / "methods" / "TaskGraph" / "run.py"), "run",
        "--run-dir", str(taskgraph_run), "--formal-isolation-root", str(artifact / "actors"),
        "--m5-root", str(m5), "--run-mode", "formal",
        "--chapter-end", str(args.chapter_end), "--token-budget", "12000",
        "--retries", "5", "--timeout-seconds", "900",
        "--embedding-provider", "dashscope", "--embedding-model", "qwen3.7-text-embedding",
        "--embedding-api-key-env", "DASHSCOPE_API_KEY", "--embedding-dimensions", "2048",
        "--writer-model", "gpt-5.6-luna", "--writer-effort", "medium",
        "--judge-model", "gpt-5.6-luna", "--judge-effort", "medium",
        "--extractor-model", "gpt-5.6-luna", "--extractor-effort", "medium",
    ]
    taskgraph_control.mkdir(parents=True, exist_ok=False)
    log_path = taskgraph_control / "controller.log"
    with log_path.open("ab", buffering=0) as log:
        worker = subprocess.Popen(
            command, cwd=m5, env=runtime_environment(artifact), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    atomic_json(taskgraph_control / "launch.json", {
        "schema": "m5-taskgraph-launch-v1", "pid": worker.pid,
        "command": command, "log": str(log_path), "started_at": now(),
    })
    sequence = {
        "schema": "m5-sequential-regeneration-v1", "status": "TaskGraph_launched",
        "pid": __import__("os").getpid(), "taskgraph_pid": worker.pid,
        "run_id": args.run_id, "started_at": now(), "chapter_end": args.chapter_end,
        "launch_mode": "parallel" if args.start_now else "after_B3",
    }
    if args.start_now:
        sequence["authorization"] = "同时并行运行TaskGraph"
    atomic_json(sequence_path, sequence)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        target = SEQUENCE_PATH
        if isinstance(target, Path):
            atomic_json(target, {"schema": "m5-sequential-regeneration-v1",
                                 "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                 "finished_at": now()})
        raise
