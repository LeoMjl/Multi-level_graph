"""Run TaskGraph in one process while mirroring committed chapters to text storage."""
from __future__ import annotations

import json
import runpy
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from cli_preflight_compat import configure_preflight_timeout


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def mirror_chapters(run_dir: Path, text_dir: Path) -> int:
    source = run_dir / "chapters"
    text_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        for chapter in source.glob("chapter_*.md"):
            target = text_dir / chapter.name
            if not target.is_file() or target.stat().st_size != chapter.stat().st_size:
                shutil.copy2(chapter, target)
    return sum(1 for _ in text_dir.glob("chapter_*.md"))


def main() -> None:
    if len(sys.argv) < 7 or sys.argv[1] != "--control-dir" or sys.argv[3] != "--text-dir":
        raise SystemExit(
            "usage: taskgraph_worker.py --control-dir DIR --text-dir DIR SCRIPT ACTION ..."
        )
    control_dir = Path(sys.argv[2]).resolve()
    text_dir = Path(sys.argv[4]).resolve()
    script = Path(sys.argv[5]).resolve()
    arguments = sys.argv[6:]
    sys.path.insert(0, str(script.parent / "src"))
    try:
        run_index = arguments.index("--run-dir")
        run_dir = Path(arguments[run_index + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("TaskGraph arguments require --run-dir") from exc
    try:
        end_index = arguments.index("--chapter-end")
        chapter_end = int(arguments[end_index + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("TaskGraph arguments require --chapter-end") from exc

    status_path = control_dir / "process_status.json"
    status = {
        "schema": "m5-taskgraph-worker-status-v1",
        "status": "running",
        "pid": __import__("os").getpid(),
        "script": str(script),
        "arguments": arguments,
        "run_dir": str(run_dir),
        "text_dir": str(text_dir),
        "started_at": now(),
        "preflight_timeout_seconds": configure_preflight_timeout(),
    }
    atomic_json(status_path, status)
    stop = threading.Event()

    def synchronize() -> None:
        while not stop.wait(2):
            try:
                mirror_chapters(run_dir, text_dir)
            except OSError:
                pass

    thread = threading.Thread(target=synchronize, name="m5-taskgraph-text-mirror", daemon=True)
    thread.start()
    original_argv = sys.argv
    try:
        sys.argv = [str(script), *arguments]
        runpy.run_path(str(script), run_name="__main__")
        stop.set()
        thread.join(timeout=5)
        count = mirror_chapters(run_dir, text_dir)
        if count != chapter_end:
            raise RuntimeError(f"TaskGraph completed but mirrored {count}/{chapter_end} chapters")
    except BaseException as exc:
        stop.set()
        status.update(
            status="failed", finished_at=now(),
            error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
        )
        atomic_json(status_path, status)
        raise
    finally:
        sys.argv = original_argv
    status.update(status="completed", finished_at=now(), chapter_count=chapter_end)
    atomic_json(status_path, status)


if __name__ == "__main__":
    main()
