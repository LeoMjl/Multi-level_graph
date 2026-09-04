from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlg.config import RuntimeConfig


VECTOR_CACHE_SCHEMA = "mlg-vector-index-v1"
VECTOR_CACHE_PARTIAL_SCHEMA = "mlg-vector-index-partial-v1"


@dataclass(frozen=True)
class VectorCacheSpec:
    enabled: bool
    cache_id: str
    directory: Path
    descriptor: dict[str, Any]


def build_vector_cache_spec(
    runtime: RuntimeConfig,
    *,
    index_kind: str,
    haystack_key: str,
    unit_ids: list[str],
    texts: list[str],
) -> VectorCacheSpec:
    if len(unit_ids) != len(texts):
        raise ValueError("vector cache unit_ids/texts length mismatch")
    digest = hashlib.sha256()
    for unit_id, text in zip(unit_ids, texts):
        _update_digest(digest, unit_id)
        _update_digest(digest, text)
    descriptor = {
        "schema": VECTOR_CACHE_SCHEMA,
        "index_kind": index_kind,
        "haystack_key": haystack_key,
        "embedding_base_url": runtime.embedding_base_url.rstrip("/"),
        "embedding_model": runtime.embedding_model,
        "unit_count": len(texts),
        "content_sha256": digest.hexdigest(),
    }
    encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    safe_kind = re.sub(r"[^a-zA-Z0-9_.-]+", "-", index_kind).strip("-.") or "vectors"
    directory = Path(runtime.vector_cache_dir).resolve() / f"{safe_kind}-{cache_id[:24]}"
    return VectorCacheSpec(
        enabled=bool(runtime.vector_cache_enabled),
        cache_id=cache_id,
        directory=directory,
        descriptor=descriptor,
    )


def load_vector_cache(spec: VectorCacheSpec) -> tuple[Any | None, dict[str, Any]]:
    metadata = _status_metadata(spec, status="disabled" if not spec.enabled else "miss", hit=False)
    if not spec.enabled:
        return None, metadata
    metadata_path = spec.directory / "metadata.json"
    vectors_path = spec.directory / "vectors.npy"
    if not metadata_path.is_file() or not vectors_path.is_file():
        try:
            completed, dimension, _ = load_partial_vector_cache(spec)
            if completed:
                metadata.update({
                    "vector_cache_status": "partial_resume_available",
                    "vector_cache_partial_count": completed,
                    "vector_cache_partial_dimension": dimension,
                })
        except Exception as exc:
            metadata.update({
                "vector_cache_status": "invalid_partial_rebuild_required",
                "vector_cache_error": str(exc),
            })
        return None, metadata
    try:
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored.get("descriptor") != spec.descriptor:
            raise ValueError("cache descriptor mismatch")
        expected_hash = str(stored.get("vectors_sha256", ""))
        if not expected_hash or _file_sha256(vectors_path) != expected_hash:
            raise ValueError("vectors.npy checksum mismatch")
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(spec.descriptor["unit_count"]), int(stored.get("dimension", 0)))
        if vectors.ndim != 2 or tuple(vectors.shape) != expected_shape:
            raise ValueError(f"vector shape mismatch: expected {expected_shape}, got {tuple(vectors.shape)}")
        if vectors.dtype != np.float32:
            raise ValueError(f"vector dtype mismatch: expected float32, got {vectors.dtype}")
        metadata.update({
            "vector_cache_hit": True,
            "vector_cache_status": "hit",
            "vector_cache_dimension": int(vectors.shape[1]),
            "vector_cache_bytes": int(vectors_path.stat().st_size),
        })
        return vectors, metadata
    except Exception as exc:
        metadata.update({
            "vector_cache_status": "invalid_rebuild_required",
            "vector_cache_error": str(exc),
        })
        return None, metadata


def persist_vector_cache(
    spec: VectorCacheSpec,
    vectors: Any,
    prior_metadata: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != int(spec.descriptor["unit_count"]):
        raise ValueError(
            "cannot persist vector cache with shape "
            f"{tuple(array.shape)} for {spec.descriptor['unit_count']} units"
        )
    metadata = dict(prior_metadata or _status_metadata(spec, status="miss", hit=False))
    if not spec.enabled:
        metadata.update({
            "vector_cache_hit": False,
            "vector_cache_status": "disabled",
            "vector_cache_dimension": int(array.shape[1]),
        })
        return array, metadata

    spec.directory.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    tmp_vectors = spec.directory / f".vectors-{token}.tmp"
    tmp_metadata = spec.directory / f".metadata-{token}.tmp"
    final_vectors = spec.directory / "vectors.npy"
    final_metadata = spec.directory / "metadata.json"
    try:
        with tmp_vectors.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        stored = {
            "schema": VECTOR_CACHE_SCHEMA,
            "cache_id": spec.cache_id,
            "descriptor": spec.descriptor,
            "dimension": int(array.shape[1]),
            "dtype": "float32",
            "vectors_sha256": _file_sha256(tmp_vectors),
            "vectors_bytes": int(tmp_vectors.stat().st_size),
        }
        with tmp_metadata.open("w", encoding="utf-8") as handle:
            json.dump(stored, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_vectors, final_vectors)
        os.replace(tmp_metadata, final_metadata)
        mmap = np.load(final_vectors, mmap_mode="r", allow_pickle=False)
        metadata.update({
            "vector_cache_hit": False,
            "vector_cache_status": "built_and_persisted",
            "vector_cache_dimension": int(array.shape[1]),
            "vector_cache_bytes": int(final_vectors.stat().st_size),
            "vector_cache_error": "",
        })
        return mmap, metadata
    finally:
        for temporary in (tmp_vectors, tmp_metadata):
            if temporary.exists():
                temporary.unlink()


def load_partial_vector_cache(spec: VectorCacheSpec) -> tuple[int, int, dict[str, Any]]:
    """Validate and return resumable progress for an incomplete vector index."""
    if not spec.enabled:
        return 0, 0, {}
    partial_vectors = spec.directory / "vectors.partial.npy"
    partial_metadata = spec.directory / "partial_metadata.json"
    if not partial_vectors.exists() and not partial_metadata.exists():
        return 0, 0, {}
    if not partial_vectors.is_file() or not partial_metadata.is_file():
        raise ValueError("partial vector cache is missing vectors or metadata")
    stored = json.loads(partial_metadata.read_text(encoding="utf-8"))
    if stored.get("schema") != VECTOR_CACHE_PARTIAL_SCHEMA:
        raise ValueError("partial vector cache schema mismatch")
    if stored.get("descriptor") != spec.descriptor:
        raise ValueError("partial vector cache descriptor mismatch")
    completed = int(stored.get("completed_count", 0))
    dimension = int(stored.get("dimension", 0))
    total = int(spec.descriptor["unit_count"])
    if completed <= 0 or completed > total or dimension <= 0:
        raise ValueError("partial vector cache progress is invalid")
    vectors = np.load(partial_vectors, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or tuple(vectors.shape) != (total, dimension):
        raise ValueError("partial vector cache shape mismatch")
    if vectors.dtype != np.float32:
        raise ValueError("partial vector cache dtype mismatch")
    batches = stored.get("completed_batches", [])
    if not isinstance(batches, list) or not batches:
        raise ValueError("partial vector cache has no completed batch records")
    expected_start = 0
    for batch in batches:
        start = int(batch.get("start", -1))
        end = int(batch.get("end", -1))
        expected_hash = str(batch.get("sha256", ""))
        if start != expected_start or end <= start or end > completed or not expected_hash:
            raise ValueError("partial vector cache batch sequence is invalid")
        actual_hash = hashlib.sha256(np.asarray(vectors[start:end]).tobytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"partial vector cache checksum mismatch at {start}:{end}")
        expected_start = end
    if expected_start != completed:
        raise ValueError("partial vector cache completed count mismatch")
    return completed, dimension, stored


def persist_vector_cache_batch(
    spec: VectorCacheSpec,
    batch_vectors: Any,
    *,
    start: int,
    prior_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Checkpoint one sequential embedding batch for crash-safe resume."""
    if not spec.enabled:
        raise ValueError("partial vector persistence requires an enabled cache")
    batch = np.asarray(batch_vectors, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[0] <= 0 or batch.shape[1] <= 0:
        raise ValueError(f"invalid embedding batch shape: {tuple(batch.shape)}")
    total = int(spec.descriptor["unit_count"])
    end = int(start) + int(batch.shape[0])
    if start < 0 or end > total:
        raise ValueError(f"embedding batch range {start}:{end} exceeds {total}")

    spec.directory.mkdir(parents=True, exist_ok=True)
    partial_vectors = spec.directory / "vectors.partial.npy"
    partial_metadata = spec.directory / "partial_metadata.json"
    if partial_vectors.is_file() and partial_metadata.is_file():
        stored = json.loads(partial_metadata.read_text(encoding="utf-8"))
        if stored.get("schema") != VECTOR_CACHE_PARTIAL_SCHEMA or stored.get("descriptor") != spec.descriptor:
            raise ValueError("partial vector cache metadata mismatch")
        completed = int(stored.get("completed_count", 0))
        dimension = int(stored.get("dimension", 0))
    elif not partial_vectors.exists() and not partial_metadata.exists():
        completed, dimension, stored = 0, 0, {}
    else:
        raise ValueError("partial vector cache is missing vectors or metadata")
    if completed != start:
        raise ValueError(f"embedding batch must start at resumable offset {completed}, got {start}")
    if completed and dimension != int(batch.shape[1]):
        raise ValueError("embedding batch dimension differs from partial cache")

    if not completed:
        vectors = np.lib.format.open_memmap(
            partial_vectors,
            mode="w+",
            dtype=np.float32,
            shape=(total, int(batch.shape[1])),
        )
        stored = {
            "schema": VECTOR_CACHE_PARTIAL_SCHEMA,
            "cache_id": spec.cache_id,
            "descriptor": spec.descriptor,
            "dimension": int(batch.shape[1]),
            "dtype": "float32",
            "completed_count": 0,
            "completed_batches": [],
        }
    else:
        vectors = np.load(partial_vectors, mmap_mode="r+", allow_pickle=False)
    vectors[start:end] = batch
    vectors.flush()
    stored["completed_count"] = end
    stored["completed_batches"].append({
        "start": int(start),
        "end": int(end),
        "sha256": hashlib.sha256(batch.tobytes()).hexdigest(),
    })
    _atomic_json_write(partial_metadata, stored)

    metadata = dict(prior_metadata or _status_metadata(spec, status="miss", hit=False))
    metadata.update({
        "vector_cache_hit": False,
        "vector_cache_status": "building_partial",
        "vector_cache_partial_count": end,
        "vector_cache_partial_dimension": int(batch.shape[1]),
        "vector_cache_partial_batches": len(stored["completed_batches"]),
        "vector_cache_error": "",
    })
    return metadata


def finalize_partial_vector_cache(
    spec: VectorCacheSpec,
    prior_metadata: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Atomically promote a fully checkpointed partial index to the final cache."""
    completed, dimension, stored = load_partial_vector_cache(spec)
    total = int(spec.descriptor["unit_count"])
    if completed != total:
        raise ValueError(f"cannot finalize partial vector cache at {completed}/{total}")
    partial_vectors = spec.directory / "vectors.partial.npy"
    partial_metadata = spec.directory / "partial_metadata.json"
    final_vectors = spec.directory / "vectors.npy"
    final_metadata = spec.directory / "metadata.json"
    final_stored = {
        "schema": VECTOR_CACHE_SCHEMA,
        "cache_id": spec.cache_id,
        "descriptor": spec.descriptor,
        "dimension": dimension,
        "dtype": "float32",
        "vectors_sha256": _file_sha256(partial_vectors),
        "vectors_bytes": int(partial_vectors.stat().st_size),
        "checkpoint_batch_count": len(stored["completed_batches"]),
    }
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    tmp_metadata = spec.directory / f".metadata-{token}.tmp"
    try:
        with tmp_metadata.open("w", encoding="utf-8") as handle:
            json.dump(final_stored, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial_vectors, final_vectors)
        os.replace(tmp_metadata, final_metadata)
        partial_metadata.unlink(missing_ok=True)
    finally:
        if tmp_metadata.exists():
            tmp_metadata.unlink()
    mmap = np.load(final_vectors, mmap_mode="r", allow_pickle=False)
    metadata = dict(prior_metadata or _status_metadata(spec, status="miss", hit=False))
    metadata.update({
        "vector_cache_hit": False,
        "vector_cache_status": "built_and_persisted",
        "vector_cache_dimension": dimension,
        "vector_cache_bytes": int(final_vectors.stat().st_size),
        "vector_cache_checkpoint_batches": len(stored["completed_batches"]),
        "vector_cache_partial_count": 0,
        "vector_cache_error": "",
    })
    return mmap, metadata


def rank_vectors(query_vector: list[float], vectors: Any, *, k: int) -> list[tuple[float, int]]:
    matrix = np.asarray(vectors, dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"cosine rank shape mismatch: query={tuple(query.shape)}, vectors={tuple(matrix.shape)}"
        )
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0 or matrix.shape[0] == 0:
        return []
    norms = np.linalg.norm(matrix, axis=1)
    denominators = norms * query_norm
    scores = np.divide(
        matrix @ query,
        denominators,
        out=np.zeros(matrix.shape[0], dtype=np.float32),
        where=denominators != 0,
    )
    order = np.argsort(-scores, kind="stable")[: max(0, min(k, matrix.shape[0]))]
    return [(float(scores[index]), int(index)) for index in order]


def vector_count(vectors: Any) -> int:
    return int(len(vectors)) if vectors is not None else 0


def fake_vector_cache_metadata() -> dict[str, Any]:
    return {
        "vector_cache_enabled": False,
        "vector_cache_hit": False,
        "vector_cache_status": "not_applicable_fake",
        "vector_cache_key": "",
        "vector_cache_path": "",
        "vector_cache_error": "",
    }


def _status_metadata(spec: VectorCacheSpec, *, status: str, hit: bool) -> dict[str, Any]:
    return {
        "vector_cache_enabled": spec.enabled,
        "vector_cache_hit": hit,
        "vector_cache_status": status,
        "vector_cache_key": spec.cache_id,
        "vector_cache_path": str(spec.directory),
        "vector_cache_error": "",
    }


def _update_digest(digest: Any, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temporary = path.with_name(f".{path.name}-{token}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
