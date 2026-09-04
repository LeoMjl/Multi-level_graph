from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from mlg.m4_data import M4_DIR
from mlg.m4_model import M4Model
from mlg.m4_pipeline import METHODS, PROTOCOL_VERSION, _slug, _write_rows
from mlg.m4_retrieval import read_jsonl


def run_judging(
    dataset: str, output_dir: Path, config_path: Path, *, methods: list[str] | None = None,
    workers: int = 6, batch_size: int = 20, limit: int = 0,
) -> dict[str, Any]:
    selected_methods = methods or METHODS[dataset]
    queries = {row["query_id"]: row for row in read_jsonl(M4_DIR / f"{dataset}_queries.jsonl")}
    labels = {row["query_id"]: row for row in read_jsonl(M4_DIR / f"{dataset}_labels.jsonl")}
    selected_ids = set(sorted(queries)[:limit]) if limit else set(queries)
    summaries = {}
    for method in selected_methods:
        predictions = read_jsonl(output_dir / f"{dataset}_{_slug(method)}_predictions.jsonl")
        prediction_by_id = {row["query_id"]: row for row in predictions if row["query_id"] in selected_ids}
        if set(prediction_by_id) != selected_ids:
            raise RuntimeError(f"Cannot judge incomplete {dataset}/{method} predictions")
        path = output_dir / f"{dataset}_{_slug(method)}_judgments.jsonl"
        valid = {}
        if path.exists():
            for row in read_jsonl(path):
                query_id = row.get("query_id")
                prediction = prediction_by_id.get(query_id)
                digest = hashlib.sha256(str(prediction.get("answer", "")).encode("utf-8")).hexdigest() if prediction else ""
                if (
                    prediction is not None
                    and row.get("protocol_version") == PROTOCOL_VERSION
                    and row.get("prediction_sha256") == digest
                ):
                    valid[query_id] = row
            _write_rows(path, [valid[key] for key in sorted(valid)])
        completed = set(valid)
        pending = []
        for query_id in sorted(selected_ids - completed):
            prediction = str(prediction_by_id[query_id]["answer"])
            pending.append({
                "query_id": query_id, "question": queries[query_id]["question"],
                "gold": str(labels[query_id]["answer"]), "prediction": prediction,
                "prediction_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
            })
        batches = [pending[index:index + batch_size] for index in range(0, len(pending), batch_size)]
        failures = _judge_batches(dataset, method, batches, path, config_path, workers)
        total = len(read_jsonl(path)) if path.exists() else 0
        summaries[method] = {"expected": len(selected_ids), "completed": total, "failures": failures, "path": str(path)}
        if failures or total != len(selected_ids):
            raise RuntimeError(f"{dataset}/{method} judgments incomplete: {total}/{len(selected_ids)}")
    return summaries


def _judge_batches(dataset, method, batches, output_path, config_path, workers):
    local = threading.local()

    def execute(batch_id, batch):
        if not hasattr(local, "model"):
            local.model = M4Model.from_config(config_path)
        values, call = local.model.judge_batch(batch)
        return [{
            "dataset": dataset, "method": method, "query_id": item["query_id"],
            "protocol_version": PROTOCOL_VERSION,
            "correct": int(values[item["query_id"]]), "judge_model": local.model.model,
            "prediction_sha256": item["prediction_sha256"], "batch_id": batch_id, "api_call": call,
        } for item in batch]

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, index, batch): index for index, batch in enumerate(batches)}
        for completed, future in enumerate(as_completed(futures), 1):
            batch_id = futures[future]
            try:
                rows = future.result()
                with output_path.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception as exc:
                failures.append({"batch_id": batch_id, "error": str(exc)})
            if completed == len(batches) or completed % 20 == 0:
                print(f"[M4] judge {dataset}/{method}: batches {completed}/{len(batches)}, failures={len(failures)}", flush=True)
    failure_path = output_path.with_suffix(".failures.jsonl")
    if failures:
        _write_rows(failure_path, failures)
    elif failure_path.exists():
        failure_path.unlink()
    return failures
