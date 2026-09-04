from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlg.m4_data import M4_DIR
from mlg.m4_retrieval import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inspectable M4 MultiHop-RAG QA and retrieval records.")
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    output = args.results / "audit_exports"
    output.mkdir(parents=True, exist_ok=True)
    labels = {row["query_id"]: row for row in read_jsonl(M4_DIR / "multihop_rag_labels.jsonl")}
    queries = {row["query_id"]: row for row in read_jsonl(M4_DIR / "multihop_rag_queries.jsonl")}
    for method in ("raptor", "graphrag", "memorystream", "taskgraph"):
        prediction_path = args.results / f"multihop_rag_{method}_predictions.jsonl"
        if not prediction_path.exists():
            continue
        predictions = read_jsonl(prediction_path)
        retrieval_rows, qa_rows = [], []
        for row in predictions:
            label = labels[row["query_id"]]
            query = queries[row["query_id"]]["question"]
            retrieval_rows.append({
                "query": query,
                "question_type": label["question_type"],
                "retrieval_list": [{"text": item["text"]} for item in row["retrieval"]],
                "gold_list": [{"fact": fact} for fact in label["evidence_facts"]],
            })
            qa_rows.append({
                "query": query,
                "question_type": label["question_type"],
                "model_answer": row["answer"],
                "gold_answer": label["answer"],
            })
        for kind, rows in (("retrieval", retrieval_rows), ("qa", qa_rows)):
            path = output / f"{method}_{kind}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(path)


if __name__ == "__main__":
    main()
