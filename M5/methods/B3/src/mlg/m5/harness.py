from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlg.m5.backend import FakeTextBackend, OpenAIResponsesBackend, TextBackend
from mlg.m5.continuity import load_local_continuity
from mlg.m5.dataset import M5Dataset, han_char_count
from mlg.m5.factory import CONDITIONS, make_memory
from mlg.m5.io import atomic_write_json, atomic_write_text, read_json
from mlg.m5.prompts import build_writer_prompts, packet_payload
from mlg.m5.taskgraph_config import PaperDependencyConfig


@dataclass(frozen=True)
class M5RunConfig:
    m5_root: Path = Path("M5")
    run_root: Path = Path("runs/M5")
    conditions: tuple[str, ...] = ("current_only", "recency_window", "taskgraph")
    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "medium"
    context_token_budget: int = 12000
    chapter_start: int = 1
    chapter_end: int = 320
    workers: int = 3
    replicate: int = 1
    retries: int = 2
    fake: bool = False
    resume: bool = True
    dependency_structural_window: int = 12
    dependency_semantic_threshold: float = 0.35
    dependency_reference_threshold: int = 2
    dependency_semantic_top_k: int = 128
    dependency_candidate_limit: int = 192


def run_conditions(config: M5RunConfig) -> dict[str, Any]:
    unknown = sorted(set(config.conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown M5 conditions: {unknown}")
    if "taskgraph" in config.conditions and not config.fake:
        raise RuntimeError(
            "The generic M5 harness is only a legacy/offline TaskGraph pilot; "
            "formal TaskGraph generation must use tools/run_m5_taskgraph_collab.py"
        )
    dataset = M5Dataset(config.m5_root)
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(config.workers, len(config.conditions))) as pool:
        futures = {
            pool.submit(_run_condition, dataset, config, condition): condition
            for condition in config.conditions
        }
        for future in as_completed(futures):
            condition = futures[future]
            results[condition] = future.result()
    return {"schema": "m5-run-set-v1", "conditions": results}


def _run_condition(dataset: M5Dataset, config: M5RunConfig, condition: str) -> dict[str, Any]:
    run_dir = config.run_root.resolve() / condition / f"replicate_{config.replicate:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    manifest_path = run_dir / "manifest.json"
    fingerprint = dataset.public_fingerprint()
    manifest = _manifest(config, condition, fingerprint)
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("dataset_fingerprint") != fingerprint:
            raise RuntimeError(f"Dataset changed for resumable M5 run: {run_dir}")
        if existing != manifest:
            changed = sorted(key for key in set(existing) | set(manifest) if existing.get(key) != manifest.get(key))
            raise RuntimeError(f"M5 run configuration changed for {run_dir}: {', '.join(changed)}")
        if not config.resume:
            raise FileExistsError(f"M5 run already exists and --no-resume was requested: {run_dir}")
    else:
        atomic_write_json(manifest_path, manifest)

    backend = _backend(config)
    memory = make_memory(
        condition, dataset.root, token_budget=config.context_token_budget, fake=config.fake,
        relationship_backend=backend,
        dependency_config=_dependency_config(config),
    )
    last_completed = 0
    if state_path.exists():
        state = read_json(state_path)
        last_completed = int(state.get("last_completed", 0))
        memory.load_state(state["memory"])
    start = max(config.chapter_start, last_completed + 1)
    if start > 1 and last_completed < start - 1:
        raise RuntimeError("A non-initial M5 chapter requires a checkpoint containing all prior chapters")
    _discard_stale_pending(run_dir, last_completed)
    completed_now = 0
    for chapter_id in range(start, min(config.chapter_end, dataset.total_chapters) + 1):
        _run_chapter(dataset, config, condition, run_dir, memory, backend, chapter_id)
        last_completed = chapter_id
        completed_now += 1
        atomic_write_json(state_path, {
            "schema": "m5-checkpoint-v1",
            "condition": condition,
            "last_completed": last_completed,
            "dataset_fingerprint": fingerprint,
            "memory": memory.to_state(),
        })
        pending = run_dir / "pending" / f"chapter_{chapter_id:03d}.json"
        if pending.exists():
            pending.unlink()
    return {
        "run_dir": str(run_dir), "condition": condition,
        "last_completed": last_completed, "completed_now": completed_now,
    }


def _run_chapter(
    dataset: M5Dataset,
    config: M5RunConfig,
    condition: str,
    run_dir: Path,
    memory,
    backend: TextBackend,
    chapter_id: int,
) -> None:
    chapter = dataset.release(chapter_id)
    pending_path = run_dir / "pending" / f"chapter_{chapter_id:03d}.json"
    output_path = run_dir / "chapters" / f"chapter_{chapter_id:03d}.md"
    if pending_path.exists():
        pending = read_json(pending_path)
        text = output_path.read_text(encoding="utf-8-sig")
        writer_calls = list(pending["writer_calls"])
        context_meta = dict(pending["context_metadata"])
        if memory.name == "taskgraph":
            memory.context_for(chapter)
    else:
        context = memory.context_for(chapter)
        local = load_local_continuity(run_dir, chapter_id)
        system, user = build_writer_prompts(
            dataset, chapter, context, local_continuity=local,
        )
        packet_path = run_dir / "inputs" / f"chapter_{chapter_id:03d}.json"
        atomic_write_json(packet_path, packet_payload(chapter, system, user))
        text, writer_calls = _generate_valid(
            dataset, config, backend, chapter, context, local, system, user,
        )
        method_chapters = list(context.selected_chapters)
        context_meta = {
            **context.metadata,
            "selected_chapters": sorted({
                *method_chapters,
                *([local.chapter_id] if local.chapter_id is not None else []),
            }),
            "method_selected_chapters": method_chapters,
            "local_continuity": local.audit(),
        }
        atomic_write_text(output_path, text.rstrip() + "\n")
        atomic_write_json(pending_path, {
            "schema": "m5-pending-chapter-v1", "chapter_id": chapter_id,
            "writer_calls": writer_calls, "context_metadata": context_meta,
        })
    memory_calls = [_call_record(result) for result in memory.observe(chapter, text, backend)]
    query_calls = list(context_meta.get("api_calls", []))
    atomic_write_json(run_dir / "records" / f"chapter_{chapter_id:03d}.json", {
        "schema": "m5-chapter-record-v1",
        "condition": condition,
        "chapter_id": chapter_id,
        "han_chars": han_char_count(text),
        "selected_chapters": context_meta.get("selected_chapters", []),
        "memory_metadata": context_meta,
        "api_calls": query_calls + writer_calls + memory_calls,
    })


def _generate_valid(dataset, config, backend, chapter, context, local, system, user):
    low, high = [int(item) for item in dataset.global_task.get("allowed_chars_per_chapter", [2000, 3000])]
    calls: list[dict[str, Any]] = []
    feedback = ""
    for attempt in range(config.retries + 1):
        if attempt:
            system, user = build_writer_prompts(
                dataset, chapter, context, local_continuity=local,
                retry_feedback=feedback,
            )
        result = backend.generate(system, user, purpose="chapter")
        calls.append({**_call_record(result), "attempt": attempt + 1})
        count = han_char_count(result.text)
        if low <= count <= high:
            return result.text, calls
        feedback = f"正文汉字数为{count}，必须落在{low}—{high}之间；保持情节要求不变并完整重写。"
    raise RuntimeError(f"Chapter {chapter.chapter_id} failed length validation after {config.retries + 1} attempts")


def _backend(config: M5RunConfig) -> TextBackend:
    if config.fake:
        return FakeTextBackend()
    return OpenAIResponsesBackend(model=config.model, reasoning_effort=config.reasoning_effort)


def _dependency_config(config: M5RunConfig) -> PaperDependencyConfig:
    return PaperDependencyConfig(
        structural_window=config.dependency_structural_window,
        semantic_threshold=config.dependency_semantic_threshold,
        reference_threshold=config.dependency_reference_threshold,
        semantic_top_k=config.dependency_semantic_top_k,
        candidate_limit=config.dependency_candidate_limit,
    )


def _manifest(config: M5RunConfig, condition: str, fingerprint: str) -> dict[str, Any]:
    return {
        "schema": "m5-run-manifest-v1", "condition": condition,
        "model": "fake-m5-writer" if config.fake else config.model,
        "reasoning_effort": config.reasoning_effort,
        "context_token_budget": config.context_token_budget,
        "taskgraph_dependency_config": _dependency_config(config).__dict__,
        "replicate": config.replicate, "run_mode": "fake" if config.fake else "llm",
        "dataset_fingerprint": fingerprint, "future_prompt_access": False,
        "hidden_gold_access": condition == "oracle_retrieval",
    }


def _discard_stale_pending(run_dir: Path, last_completed: int) -> None:
    for path in (run_dir / "pending").glob("chapter_*.json") if (run_dir / "pending").exists() else []:
        chapter_id = int(path.stem.rsplit("_", 1)[-1])
        if chapter_id <= last_completed:
            path.unlink()


def _call_record(result) -> dict[str, Any]:
    payload = result.to_dict()
    payload.pop("text", None)
    return payload
