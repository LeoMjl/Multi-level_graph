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

from mlg.m5.collab_protocol import CollaborationTaskGraphRun
from mlg.m5.collab_guard import guard_collaboration_commit
from mlg.m5.collab_state import CollaborationRunState
from mlg.m5.codex_cli_writer import CodexCliWriter
from mlg.m5.codex_formal_isolation import formal_isolation_preflight
from mlg.m5.dashscope_embedding import DashScopeEmbeddingBackend
from mlg.m5.io import atomic_write_json
from mlg.m5.io import read_json
from mlg.m5.openrouter_embedding import OpenRouterEmbeddingBackend
from mlg.m5.taskgraph_config import PaperDependencyConfig
from mlg.m5.taskgraph_cli_controller import TaskGraphCodexController


DEFAULT_RUN = ROOT / "run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M5 TaskGraph collaboration harness: Codex writes and judges; "
            "the configured embeddings API encodes released node representations only"
        ),
    )
    parser.add_argument(
        "action",
        choices=(
            "run", "resume", "status", "prepare", "apply-prewrite", "submit-chapter",
            "submit-writeback", "guard", "repair", "preflight",
        ),
    )
    parser.add_argument("--chapter", type=int)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--writeback", type=Path)
    parser.add_argument("--chapter-end", type=int, default=320)
    parser.add_argument("--token-budget", type=int, default=12000)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--formal-isolation-root", type=Path)
    parser.add_argument("--m5-root", type=Path, default=ROOT.parents[1])
    parser.add_argument(
        "--embedding-model",
        default="qwen3.7-text-embedding",
    )
    parser.add_argument(
        "--embedding-provider", choices=("dashscope", "openrouter"),
        default="dashscope",
    )
    parser.add_argument("--embedding-api-key-env")
    parser.add_argument(
        "--embedding-base-url",
    )
    parser.add_argument("--embedding-dimensions", type=int, default=2048)
    parser.add_argument(
        "--run-mode", choices=("pilot", "formal"), default="formal",
        help="New runs default to the formal protocol",
    )
    parser.add_argument("--writer-model", default="gpt-5.6-luna")
    parser.add_argument("--writer-effort", default="medium")
    parser.add_argument("--writer-agent-id")
    parser.add_argument("--writer-input-tokens", type=int)
    parser.add_argument("--writer-output-tokens", type=int)
    parser.add_argument("--writer-elapsed-ms", type=float)
    parser.add_argument("--judge-model", default="gpt-5.6-luna")
    parser.add_argument("--judge-effort", default="medium")
    parser.add_argument("--judge-agent-id")
    parser.add_argument("--judge-input-tokens", type=int)
    parser.add_argument("--judge-output-tokens", type=int)
    parser.add_argument("--judge-elapsed-ms", type=float)
    parser.add_argument("--extractor-model", default="gpt-5.6-luna")
    parser.add_argument("--extractor-effort", default="medium")
    parser.add_argument("--extractor-agent-id")
    parser.add_argument("--extractor-input-tokens", type=int)
    parser.add_argument("--extractor-output-tokens", type=int)
    parser.add_argument("--extractor-elapsed-ms", type=float)
    parser.add_argument(
        "--force-unlock", action="store_true",
        help="Remove a stale local lock before repair; use only after confirming no process runs",
    )
    parser.add_argument("--full-state", action="store_true")
    return parser


def require(value, label: str):
    if value is None:
        raise SystemExit(f"{label} is required for this action")
    return value


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    manual_mutations = {
        "prepare", "apply-prewrite", "submit-chapter", "submit-writeback",
    }
    if args.run_mode == "formal" and args.action in manual_mutations:
        raise SystemExit(
            "Formal runs only allow run/resume automation; manual stage actions "
            "are restricted to --run-mode pilot"
        )
    if args.action == "run" and run_dir.is_dir() and any(run_dir.iterdir()):
        raise SystemExit("run requires an empty --run-dir; use resume for existing state")
    if args.action == "resume" and not (run_dir / "state.json").is_file():
        raise SystemExit("resume requires an existing state.json")
    existing_state = (
        read_json(run_dir / "state.json")
        if (run_dir / "state.json").is_file() else None
    )
    if existing_state is not None and args.action not in {"status", "guard"}:
        stored_revision = existing_state.get("config", {}).get("protocol_revision")
        if stored_revision != CollaborationRunState.protocol_revision:
            raise SystemExit(
                f"Historical {stored_revision} run is read-only under "
                f"{CollaborationRunState.protocol_revision}; only status/guard are allowed"
            )
    if args.action == "status":
        if existing_state is None:
            raise SystemExit(f"No collaboration state: {run_dir / 'state.json'}")
        print(json.dumps(
            existing_state if args.full_state else _status_payload(existing_state),
            ensure_ascii=False, indent=2,
        ))
        return
    if args.action == "guard":
        result = guard_collaboration_commit(
            run_dir, require(args.chapter, "--chapter"),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ok"] else 1)
    isolation_evidence = None
    if args.action == "preflight" and args.run_mode != "formal":
        raise SystemExit("preflight is only defined for --run-mode formal")
    if args.run_mode == "formal" and args.action in {"run", "resume", "preflight"}:
        isolation_evidence = _formal_preflight(args, run_dir)
    if args.action == "preflight":
        print(json.dumps(isolation_evidence.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.force_unlock:
        if args.action not in {"repair", "resume"}:
            raise SystemExit("--force-unlock is restricted to repair/resume")
        (run_dir / ".collaboration.lock").unlink(missing_ok=True)
    stored_state = existing_state
    if args.action == "repair" and stored_state is None:
        raise SystemExit("repair requires an existing collaboration state.json")
    if args.action == "repair" and stored_state:
        stored_config = stored_state["config"]
        args.run_mode = stored_state["run_mode"]
        args.embedding_model = stored_config["embedding_model"]
        args.token_budget = int(stored_config["token_budget"])
    if args.action in {"run", "resume", "prepare", "submit-writeback"}:
        embeddings = _embedding_backend(args)
    else:
        embeddings = _MetadataEmbeddingBackend(args.embedding_model)
    if args.action in {"run", "resume"}:
        controller = _controller(args, embeddings, isolation_evidence)
        if isolation_evidence:
            atomic_write_json(
                run_dir / "controller" / "isolation_preflight.json",
                isolation_evidence.to_dict(),
            )
        result = controller.run_to(
            args.chapter_end,
            fresh=args.action == "run",
            force_unlock=args.force_unlock,
            progress=lambda row: print(
                json.dumps(row, ensure_ascii=False), flush=True,
            ),
        )
        print(json.dumps(controller.status(result), ensure_ascii=False, indent=2))
        return
    actors = (
        stored_state.get("actor_provenance", {})
        if args.action == "repair" and stored_state else _actor_provenance(args)
    )
    if args.action == "repair" and stored_state:
        stored_revision = stored_state.get("config", {}).get("protocol_revision")
        if stored_revision != CollaborationRunState.protocol_revision:
            raise RuntimeError(
                f"Historical {stored_revision} run is read-only under "
                f"{CollaborationRunState.protocol_revision}; start a fresh run"
            )
    dependency_config = (
        PaperDependencyConfig(**stored_state["config"]["dependency_config"])
        if args.action == "repair" and stored_state else PaperDependencyConfig()
    )
    run = CollaborationTaskGraphRun(
        args.m5_root, run_dir, embeddings,
        dependency_config=dependency_config,
        token_budget=args.token_budget,
        run_mode=args.run_mode,
        actor_provenance=actors,
        controller_policy=(
            stored_state.get("config", {}).get("controller_policy")
            if stored_state else None
        ),
    )
    if args.action == "prepare":
        result = run.prepare_prewrite_judgment(require(args.chapter, "--chapter"))
    elif args.action == "apply-prewrite":
        result = run.apply_prewrite_judgment(
            require(args.chapter, "--chapter"), require(args.response, "--response"),
        )
    elif args.action == "submit-chapter":
        result = run.submit_chapter(
            require(args.chapter, "--chapter"), require(args.text, "--text"),
        )
    elif args.action == "submit-writeback":
        result = run.submit_writeback(
            require(args.chapter, "--chapter"), require(args.writeback, "--writeback"),
        )
    else:
        result = run.repair_pending_commit(args.chapter)
    if isinstance(result, Path):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _formal_preflight(args, run_dir: Path):
    codex = which("codex.cmd") or which("codex")
    if not codex:
        raise SystemExit("Codex CLI executable was not found")
    base = args.formal_isolation_root or (
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "M5CodexActors"
    )
    run_key = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:20]
    actor_root = base.resolve() / run_key
    forbidden = [
        args.m5_root / "hooks_gold.jsonl",
        args.m5_root / "hook_schedule.jsonl",
        args.m5_root / "prompts" / "chapters_281_320.jsonl",
        args.m5_root / "story_bible.md",
        ROOT / "src" / "mlg" / "m5" / "dataset.py",
    ]
    return formal_isolation_preflight(
        codex,
        repository_root=ROOT,
        actor_root=actor_root,
        forbidden_files=forbidden,
    )


def _controller(args, embeddings, isolation_evidence=None) -> TaskGraphCodexController:
    work_dir = (
        Path(isolation_evidence.actor_root)
        if isolation_evidence else
        args.run_dir.resolve() / "controller" / "isolated_workdir"
    )
    writers: dict[tuple[str, str], CodexCliWriter] = {}

    def writer(model: str, effort: str) -> CodexCliWriter:
        key = (model, effort)
        if key not in writers:
            writers[key] = CodexCliWriter(
                work_dir,
                model=model,
                reasoning_effort=effort,
                timeout_seconds=args.timeout_seconds,
                isolation_evidence=isolation_evidence,
            )
        return writers[key]

    return TaskGraphCodexController(
        args.m5_root, args.run_dir, embeddings,
        judge_writer=writer(args.judge_model, args.judge_effort),
        chapter_writer=writer(args.writer_model, args.writer_effort),
        extractor_writer=writer(args.extractor_model, args.extractor_effort),
        run_mode=args.run_mode,
        retries=args.retries,
        token_budget=args.token_budget,
        dependency_config=PaperDependencyConfig(),
    )


def _embedding_backend(args):
    if args.embedding_provider == "dashscope":
        return DashScopeEmbeddingBackend(
            model=args.embedding_model,
            api_key_env=args.embedding_api_key_env or "DASHSCOPE_API_KEY",
            base_url=(args.embedding_base_url or
                      "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            dimensions=args.embedding_dimensions,
        )
    return OpenRouterEmbeddingBackend(
        model=args.embedding_model,
        api_key_env=args.embedding_api_key_env or "OPENROUTER_API_KEY",
        base_url=args.embedding_base_url or "https://openrouter.ai/api/v1",
    )


class _MetadataEmbeddingBackend:
    """Load/repair a run without requiring a network credential."""

    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, texts, *, purpose: str):
        raise RuntimeError("This action unexpectedly requested embeddings")


def _actor_provenance(args) -> dict[str, dict[str, object]]:
    judge = {
        "model": args.judge_model,
        "reasoning_effort": args.judge_effort,
        "agent_id": args.judge_agent_id,
        "input_tokens": args.judge_input_tokens,
        "output_tokens": args.judge_output_tokens,
        "elapsed_ms": args.judge_elapsed_ms,
    }
    return {
        "prewrite_judge": dict(judge),
        "writer": {
            "model": args.writer_model,
            "reasoning_effort": args.writer_effort,
            "agent_id": args.writer_agent_id,
            "input_tokens": args.writer_input_tokens,
            "output_tokens": args.writer_output_tokens,
            "elapsed_ms": args.writer_elapsed_ms,
        },
        "writeback_extractor": {
            "model": args.extractor_model,
            "reasoning_effort": args.extractor_effort,
            "agent_id": args.extractor_agent_id,
            "input_tokens": args.extractor_input_tokens,
            "output_tokens": args.extractor_output_tokens,
            "elapsed_ms": args.extractor_elapsed_ms,
        },
    }


def _status_payload(state: dict) -> dict:
    tx = state.get("transaction", {})
    return {
        key: state.get(key) for key in (
            "schema", "run_mode", "stage", "last_completed", "active_chapter",
            "dataset_fingerprint", "config_fingerprint", "updated_at",
        )
    } | {
        "transaction_id": tx.get("transaction_id"),
        "chapter_prompt_sha256": tx.get("chapter_prompt_sha256"),
        "artifacts": sorted(tx.get("artifacts", {})),
    }


if __name__ == "__main__":
    main()
