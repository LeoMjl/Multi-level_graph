from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mlg.m4_model import M4Model
from mlg.m4_retrieval import read_jsonl


KEY_PROTOCOL = "target-paper-llm-generated-key-v1"
KEY_BATCH_SIZE = 8


def build_memorystream(
    dataset: str,
    chunks: list[dict[str, Any]],
    model: M4Model,
    index_dir: Path,
    *,
    batch_size: int = KEY_BATCH_SIZE,
) -> tuple[Any, dict[str, Any]]:
    """Build the target paper's flat LLM-keyed memory table before queries are read."""
    index_dir.mkdir(parents=True, exist_ok=True)
    key_path = index_dir / f"{dataset}_keys.jsonl"
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    valid: dict[str, dict[str, Any]] = {}
    if key_path.exists():
        for row in read_jsonl(key_path):
            chunk = chunks_by_id.get(row.get("chunk_id"))
            if (
                chunk is not None
                and row.get("dataset") == dataset
                and row.get("protocol") == KEY_PROTOCOL
                and row.get("model") == model.model
                and bool(str(row.get("key", "")).strip())
                and row.get("value_sha256") == _sha(chunk["text"])
            ):
                valid[row["chunk_id"]] = row
        _write_rows(key_path, [valid[key] for key in sorted(valid)])

    pending = [row for row in chunks if row["chunk_id"] not in valid]
    api_calls = 0
    repaired_ids = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        try:
            keys, call = model.memory_keys_batch(batch)
            api_calls += 1
            repaired_ids += int(bool(call.get("single_chunk_id_repaired")))
            generated = _key_rows(dataset, batch, keys, model.model, call)
        except Exception as batch_error:
            generated = []
            for item in batch:
                try:
                    keys, call = model.memory_keys_batch([item])
                    api_calls += 1
                    repaired_ids += int(bool(call.get("single_chunk_id_repaired")))
                    generated.extend(_key_rows(dataset, [item], keys, model.model, call))
                except Exception as single_error:
                    raise RuntimeError(
                        f"MemoryStream key build failed for {item['chunk_id']}: {single_error}; "
                        f"batch error: {batch_error}"
                    ) from single_error
        _append_rows(key_path, generated)
        valid.update({row["chunk_id"]: row for row in generated})
        completed = min(start + len(batch), len(pending))
        if completed == len(pending) or completed % (20 * batch_size) == 0:
            print(
                f"[M4] {dataset}/MemoryStream keys: {completed}/{len(pending)}, api_calls={api_calls}",
                flush=True,
            )

    if len(valid) != len(chunks):
        raise RuntimeError(f"MemoryStream key table incomplete: {len(valid)}/{len(chunks)}")
    ordered = [valid[row["chunk_id"]] for row in chunks]
    _write_rows(key_path, ordered)
    vectors, vector_meta = model.cached_embeddings(
        [row["key"] for row in ordered],
        [row["chunk_id"] for row in ordered],
        dataset=dataset,
        kind=f"memorystream_keys_{KEY_PROTOCOL}",
    )
    metadata = {
        "protocol": KEY_PROTOCOL,
        "key_file": str(key_path),
        "entries": len(ordered),
        "logical_insertions": len(chunks),
        "new_api_calls": api_calls,
        "single_id_repairs": repaired_ids,
        "key_model": model.model,
        "embedding_cache": vector_meta,
    }
    return vectors, metadata


def _key_rows(
    dataset: str,
    batch: list[dict[str, Any]],
    keys: dict[str, str],
    model: str,
    call: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "chunk_id": item["chunk_id"],
            "key": keys[item["chunk_id"]],
            "value_sha256": _sha(item["text"]),
            "protocol": KEY_PROTOCOL,
            "model": model,
            "api_call": call,
        }
        for item in batch
    ]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
