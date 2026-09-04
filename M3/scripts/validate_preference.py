from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUBSETS = ("G2_instruction", "G2_category", "G3_instruction")
PROTOCOL = "alternating_candidate_order_v2"


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def tool_names(example: dict) -> list[str]:
    names: list[str] = []
    for tool in example.get("available_tools", []):
        if not isinstance(tool, dict):
            raise ValueError("Tool definition must be an object")
        nested = tool.get("function")
        name = tool.get("name")
        if not name and isinstance(nested, dict):
            name = nested.get("name")
        if not name:
            raise ValueError("Tool definition has no name")
        names.append(str(name))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit complete M3 candidate-vs-CoT preference artifacts."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path)
    parser.add_argument("--preference-root", type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--evaluate-times", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converted_root = args.converted_root or args.output_root / "converted"
    preference_root = args.preference_root or args.output_root / "preference"

    all_complete = True
    candidate_audits: dict[str, object] = {}
    for candidate in args.candidates:
        subset_audits: dict[str, object] = {}
        for subset in SUBSETS:
            ids_path = (
                args.official_root
                / "solvable_queries"
                / "test_query_ids"
                / f"{subset}.json"
            )
            expected_ids = {str(item) for item in load_object(ids_path)}
            ref_path = converted_root / args.reference / f"{subset}.json"
            candidate_path = converted_root / candidate / f"{subset}.json"
            reference = load_object(ref_path)
            candidate_answers = load_object(candidate_path)
            ref_ids = {str(item) for item in reference}
            candidate_ids = {str(item) for item in candidate_answers}

            mismatched_pairs: list[str] = []
            for query_id in sorted(expected_ids & ref_ids & candidate_ids):
                ref_item = reference[query_id]
                candidate_item = candidate_answers[query_id]
                if not isinstance(ref_item, dict) or not isinstance(candidate_item, dict):
                    mismatched_pairs.append(query_id)
                    continue
                if (
                    str(ref_item.get("query", ""))
                    != str(candidate_item.get("query", ""))
                    or tool_names(ref_item) != tool_names(candidate_item)
                ):
                    mismatched_pairs.append(query_id)

            pref_path = preference_root / (
                f"{subset}_{args.reference}_{candidate}.json"
            )
            preference = load_object(pref_path)
            preference_ids = {str(item) for item in preference}
            invalid_preference_ids: list[str] = []
            for query_id, item in preference.items():
                if not isinstance(item, dict) or item.get("_protocol") != PROTOCOL:
                    invalid_preference_ids.append(str(query_id))
                    continue
                ref_votes = item.get(args.reference)
                candidate_votes = item.get(candidate)
                if (
                    not isinstance(ref_votes, int)
                    or not isinstance(candidate_votes, int)
                    or ref_votes < 0
                    or candidate_votes < 0
                    or ref_votes + candidate_votes > args.evaluate_times
                ):
                    invalid_preference_ids.append(str(query_id))
                    continue
                for round_id in range(args.evaluate_times):
                    expected_order = (
                        "reference_first" if round_id % 2 == 0 else "output_first"
                    )
                    if (
                        item.get(f"round_{round_id}") != "complete"
                        or item.get(f"round_{round_id}_order") != expected_order
                    ):
                        invalid_preference_ids.append(str(query_id))
                        break

            csv_path = pref_path.with_suffix(".csv")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            csv_ids = [str(row.get("query_id", "")) for row in rows]
            subset_complete = (
                expected_ids == ref_ids == candidate_ids == preference_ids
                and not mismatched_pairs
                and not invalid_preference_ids
                and len(csv_ids) == len(expected_ids)
                and set(csv_ids) == expected_ids
            )
            all_complete = all_complete and subset_complete
            subset_audits[subset] = {
                "expected_count": len(expected_ids),
                "reference_count": len(ref_ids),
                "candidate_count": len(candidate_ids),
                "preference_count": len(preference_ids),
                "csv_count": len(csv_ids),
                "missing_preference_ids": sorted(expected_ids - preference_ids),
                "extra_preference_ids": sorted(preference_ids - expected_ids),
                "mismatched_pair_ids": mismatched_pairs,
                "invalid_preference_ids": sorted(set(invalid_preference_ids)),
                "complete": subset_complete,
            }
        candidate_audits[candidate] = {
            "complete": all(item["complete"] for item in subset_audits.values()),
            "subsets": subset_audits,
        }

    payload = {
        "reference": args.reference,
        "protocol": PROTOCOL,
        "evaluate_times": args.evaluate_times,
        "complete": all_complete,
        "candidates": candidate_audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
