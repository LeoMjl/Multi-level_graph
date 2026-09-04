from __future__ import annotations

import argparse
import json
from pathlib import Path


SUBSETS = ("G2_instruction", "G2_category", "G3_instruction")


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def tool_names(example: dict) -> list[str]:
    names: list[str] = []
    for tool in example.get("available_tools", []):
        nested = tool.get("function") if isinstance(tool, dict) else None
        name = tool.get("name") if isinstance(tool, dict) else None
        if not name and isinstance(nested, dict):
            name = nested.get("name")
        if not name:
            raise ValueError("Tool definition has no name")
        names.append(str(name))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit M3 converted inputs before pairwise evaluation."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit: dict[str, object] = {}
    all_complete = True
    for candidate in args.candidates:
        subsets: dict[str, object] = {}
        for subset in SUBSETS:
            expected = set(load_object(
                args.official_root / "solvable_queries" / "test_query_ids"
                / f"{subset}.json"
            ))
            reference = load_object(
                args.output_root / "converted" / args.reference / f"{subset}.json"
            )
            candidate_rows = load_object(
                args.output_root / "converted" / candidate / f"{subset}.json"
            )
            ref_ids = set(reference)
            candidate_ids = set(candidate_rows)
            mismatched = []
            for query_id in sorted(expected & ref_ids & candidate_ids):
                ref_item = reference[query_id]
                candidate_item = candidate_rows[query_id]
                if (
                    not isinstance(ref_item, dict)
                    or not isinstance(candidate_item, dict)
                    or str(ref_item.get("query", ""))
                    != str(candidate_item.get("query", ""))
                    or tool_names(ref_item) != tool_names(candidate_item)
                ):
                    mismatched.append(query_id)
            complete = expected == ref_ids == candidate_ids and not mismatched
            all_complete = all_complete and complete
            subsets[subset] = {
                "expected_count": len(expected),
                "reference_count": len(ref_ids),
                "candidate_count": len(candidate_ids),
                "missing_reference_ids": sorted(expected - ref_ids),
                "missing_candidate_ids": sorted(expected - candidate_ids),
                "extra_reference_ids": sorted(ref_ids - expected),
                "extra_candidate_ids": sorted(candidate_ids - expected),
                "mismatched_pair_ids": mismatched,
                "complete": complete,
            }
        audit[candidate] = {"complete": all(
            item["complete"] for item in subsets.values()
        ), "subsets": subsets}

    payload = {
        "reference": args.reference,
        "expected_total": 291,
        "complete": all_complete,
        "candidates": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
