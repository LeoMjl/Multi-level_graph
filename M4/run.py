from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mlg.m4_data import M4_DIR, prepare_m4_data
from mlg.m4_judge import run_judging
from mlg.m4_model import M4Model
from mlg.m4_pipeline import METHODS, build_inputs, run_generation
from mlg.m4_retrieval import read_jsonl
from mlg.m4_report import build_report


METHOD_SLUGS = {
    "full_text": "Full Text", "raptor": "RAPTOR", "graphrag": "GraphRAG",
    "memorystream": "MemoryStream", "taskgraph": "TaskGraph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M4 QuALITY and MultiHop-RAG document QA.")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--model-config", type=Path, default=ROOT / "model_config.example.json")
    parser.add_argument("--datasets", nargs="+", choices=["quality", "multihop_rag"], default=["quality", "multihop_rag"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--judge-batch-size", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_SLUGS), default=list(METHOD_SLUGS))
    parser.add_argument("--limit", type=int, default=0, help="Pilot-only questions per dataset; zero runs the full benchmark.")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Resume from existing retrieval JSONL files.")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.build_only, args.skip_build, args.judge_only, args.report_only)) > 1:
        raise ValueError("Choose at most one of --build-only, --skip-build, --judge-only, or --report-only")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        build_report(args.output)
        print(args.output / "summary.json")
        return
    data_manifest = prepare_m4_data()
    requested = [METHOD_SLUGS[name] for name in args.methods]
    model = M4Model.from_config(args.model_config)
    if not args.skip_build and not args.judge_only and not model.can_call_embedding():
        raise RuntimeError(f"Embedding API is not configured ({model.runtime.embedding_api_key_env})")
    if not args.build_only and not model.can_call_llm():
        raise RuntimeError(f"LLM API is not configured ({model.runtime.api_key_env})")
    audit_path = args.output / "run_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {
        "schema": "m4-target-paper-run-v3", "datasets": {},
    }
    audit.update({
        "schema": "m4-target-paper-run-v3",
        "formal": args.limit == 0, "limit": args.limit, "requested_methods": requested,
        "reader_batch_size": args.batch_size, "judge_batch_size": args.judge_batch_size,
        "skip_build": args.skip_build,
        "data_manifest": data_manifest,
    })
    for dataset in args.datasets:
        methods = [method for method in METHODS[dataset] if method in requested]
        if not methods:
            continue
        dataset_audit = audit["datasets"].get(dataset, {})
        if args.skip_build:
            prepared = []
            for method in methods:
                slug = next(key for key, value in METHOD_SLUGS.items() if value == method)
                retrieval_path = args.output / f"{dataset}_{slug}_retrieval.jsonl"
                if not retrieval_path.exists():
                    raise FileNotFoundError(f"Missing retrieval checkpoint: {retrieval_path}")
                prepared.extend(read_jsonl(retrieval_path))
            document_text = {
                row["doc_id"]: row["body"]
                for row in read_jsonl(M4_DIR / f"{dataset}_documents.jsonl")
            }
        elif not args.judge_only:
            prepared, documents, build_audit = build_inputs(
                dataset, model, args.output, args.model_config,
                methods=methods, query_limit=args.limit,
            )
            dataset_audit["build"] = build_audit
            document_text = documents
        if not args.build_only and not args.judge_only:
            dataset_audit["generation"] = run_generation(
                dataset, prepared, document_text, args.output, args.model_config,
                methods=methods, workers=args.workers, batch_size=args.batch_size, limit=args.limit,
            )
        if not args.build_only:
            dataset_audit["judging"] = run_judging(
                dataset, args.output, args.model_config, methods=methods,
                workers=args.workers, batch_size=args.judge_batch_size, limit=args.limit,
            )
        audit["datasets"][dataset] = dataset_audit
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.build_only and args.limit == 0 and set(args.datasets) == {"quality", "multihop_rag"} and set(requested) == set(sum(METHODS.values(), [])):
        build_report(args.output)
    print(audit_path)


if __name__ == "__main__":
    main()
