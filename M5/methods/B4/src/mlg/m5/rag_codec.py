from __future__ import annotations

import base64
import hashlib
import zlib
from array import array
from pathlib import Path
from typing import Any


def pack_vectors(vectors: list[list[float]]) -> dict[str, Any]:
    if not vectors:
        return {"rows": 0, "cols": 0, "data": ""}
    flat = array("f", (value for row in vectors for value in row))
    data = base64.b64encode(zlib.compress(flat.tobytes(), 6)).decode("ascii")
    return {"rows": len(vectors), "cols": len(vectors[0]), "data": data}


def unpack_vectors(payload: dict[str, Any]) -> list[list[float]]:
    rows, cols = int(payload.get("rows", 0)), int(payload.get("cols", 0))
    if not rows or not cols:
        return []
    flat = array("f")
    flat.frombytes(zlib.decompress(base64.b64decode(payload["data"])))
    if len(flat) != rows * cols:
        raise RuntimeError("Packed embedding state has an invalid shape")
    return [list(flat[start:start + cols])
            for start in range(0, rows * cols, cols)]


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"Embedding artifact is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hashed_bigrams(text: str, dims: int = 128) -> list[float]:
    values = [0.0] * dims
    for index in range(max(0, len(text) - 1)):
        token = text[index:index + 2].encode("utf-8")
        values[zlib.crc32(token) % dims] += 1.0
    return values
