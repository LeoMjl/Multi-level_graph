from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def _replace_with_retry(source: str, destination: Path) -> None:
    """Tolerate brief Windows readers/antivirus locks on atomic state files."""
    delays = (0.05, 0.1, 0.2, 0.4, 0.8)
    for attempt, delay in enumerate(delays, start=1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(delay)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))
