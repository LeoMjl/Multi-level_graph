from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


SUBSETS = ("G2_instruction", "G2_category", "G3_instruction")
PASS_LABELS = {
    "AnswerStatus.Solved",
    "AnswerStatus.Unsure",
    "AnswerStatus.Unsolved",
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate complete three-round SoPR and FAC artifacts."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--evaluate-times", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit: dict[str, object] = {}
    complete = True
    expected_rounds = {str(index) for index in range(args.evaluate_times)}
    for subset in SUBSETS:
        ids_path = (
            args.official_root
            / "solvable_queries"
            / "test_query_ids"
            / f"{subset}.json"
        )
        expected_ids = {str(item) for item in load_json(ids_path)}
        pass_path = (
            args.output_root
            / "pass_rate"
            / args.method
            / f"{subset}_{args.method}.json"
        )
        pass_payload = load_json(pass_path)
        if not isinstance(pass_payload, dict):
            raise ValueError(f"Pass-rate artifact is not an object: {pass_path}")
        pass_ids = {str(item) for item in pass_payload}
        incomplete_round_ids = sorted(
            str(query_id)
            for query_id, item in pass_payload.items()
            if not isinstance(item, dict)
            or not isinstance(item.get("is_solved"), dict)
            or set(map(str, item["is_solved"])) != expected_rounds
        )
        invalid_label_ids = sorted(
            str(query_id)
            for query_id, item in pass_payload.items()
            if isinstance(item, dict)
            and isinstance(item.get("is_solved"), dict)
            and any(str(label) not in PASS_LABELS for label in item["is_solved"].values())
        )

        converted_path = (
            args.output_root / "converted" / args.method / f"{subset}.json"
        )
        converted = load_json(converted_path)
        if not isinstance(converted, dict):
            raise ValueError(f"Converted artifact is not an object: {converted_path}")
        expected_queries = Counter(
            str(item.get("query", ""))
            for item in converted.values()
            if isinstance(item, dict)
        )
        fac_path = args.output_root / "fac" / args.method / f"{subset}.csv"
        with fac_path.open("r", encoding="utf-8", newline="") as handle:
            fac_rows = list(csv.DictReader(handle))
        actual_queries = Counter(str(row.get("query", "")) for row in fac_rows)
        empty_evaluations = sum(
            not str(row.get("evaluation", "")).strip() for row in fac_rows
        )
        invalid_evaluations = sum(
            re.search(
                r"\b(?:solved|unsolved)\b",
                str(row.get("evaluation", "")),
                flags=re.IGNORECASE,
            )
            is None
            for row in fac_rows
        )
        subset_complete = (
            expected_ids == pass_ids
            and not incomplete_round_ids
            and not invalid_label_ids
            and len(converted) == len(expected_ids)
            and len(fac_rows) == len(expected_ids)
            and actual_queries == expected_queries
            and empty_evaluations == 0
            and invalid_evaluations == 0
        )
        complete = complete and subset_complete
        audit[subset] = {
            "expected_count": len(expected_ids),
            "pass_rate_count": len(pass_ids),
            "missing_pass_rate_ids": sorted(expected_ids - pass_ids),
            "extra_pass_rate_ids": sorted(pass_ids - expected_ids),
            "incomplete_round_ids": incomplete_round_ids,
            "invalid_label_ids": invalid_label_ids,
            "fac_count": len(fac_rows),
            "fac_query_multiset_matches": actual_queries == expected_queries,
            "empty_fac_evaluations": empty_evaluations,
            "invalid_fac_evaluations": invalid_evaluations,
            "complete": subset_complete,
        }

    payload = {
        "method": args.method,
        "evaluate_times": args.evaluate_times,
        "complete": complete,
        "subsets": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
