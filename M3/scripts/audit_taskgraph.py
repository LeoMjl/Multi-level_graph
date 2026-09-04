from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from mlg.stabletoolbench.planning import PLAN_PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit semantic liveness of TaskGraph raw artifacts."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-invalid-expansions", type=int, default=0)
    parser.add_argument("--max-invalid-validations", type=int, default=0)
    parser.add_argument("--require-state-update", action="store_true")
    return parser.parse_args()


def audit(raw_root: Path, args: argparse.Namespace) -> dict:
    paths = sorted(raw_root.glob("**/*_MLG.json"))
    totals = Counter()
    invalid_files: dict[str, list[str]] = {
        "json": [],
        "protocol": [],
        "post_termination": [],
        "successful_invalid_expansion": [],
        "invalid_business_stage": [],
    }
    subsets = Counter()
    for path in paths:
        relative = str(path.relative_to(raw_root)).replace("\\", "/")
        subsets[path.parent.name] += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_files["json"].append(relative)
            continue
        totals["parsed"] += 1
        if payload.get("mlg_protocol") != PLAN_PROTOCOL:
            invalid_files["protocol"].append(relative)
        answer = payload.get("answer_generation", {})
        successful = isinstance(answer, dict) and answer.get("valid_data") is True
        totals["successful_tasks"] += int(successful)
        expansions = payload.get("mlg_stage_expansion", {}).get("trace", [])
        invalid_expansions = sum(
            item.get("source") == "invalid" for item in expansions
        )
        repaired_expansions = sum(
            item.get("source") == "model_repaired" for item in expansions
        )
        post_termination = sum(
            item.get("phase") == "post_termination_budget"
            for item in expansions
        )
        totals["invalid_expansions"] += invalid_expansions
        totals["repaired_expansions"] += repaired_expansions
        totals["post_termination_expansions"] += post_termination
        if post_termination:
            invalid_files["post_termination"].append(relative)
        if successful and invalid_expansions:
            invalid_files["successful_invalid_expansion"].append(relative)
        graph = payload.get("mlg_observable_graph", {})
        invalid_business = any(
            node.get("sub_type") == "CoarseStage"
            and node.get("metadata", {}).get("expansion_valid") is False
            for node in graph.get("nodes", [])
            if isinstance(node, dict)
        )
        totals["invalid_business_stage_tasks"] += int(invalid_business)
        if successful and invalid_business:
            invalid_files["invalid_business_stage"].append(relative)
        validations = payload.get("mlg_result_validation", {}).get("trace", [])
        totals["validation_steps"] += len(validations)
        totals["invalid_validations"] += sum(
            item.get("status") == "invalid" for item in validations
        )
        totals["repaired_validations"] += sum(
            item.get("status") == "model_repaired" for item in validations
        )
        totals["parsed_state_updates"] += sum(
            int(item.get("state_update_count", 0) or 0) for item in validations
        )
        updates = payload.get("mlg_state_updates", {}).get("trace", [])
        totals["accepted_state_updates"] += sum(
            bool(item.get("accepted")) for item in updates
        )
    complete = bool(paths) and totals["parsed"] == len(paths)
    complete = complete and not invalid_files["protocol"]
    complete = complete and not invalid_files["post_termination"]
    complete = complete and not invalid_files["successful_invalid_expansion"]
    complete = complete and not invalid_files["invalid_business_stage"]
    complete = complete and (
        totals["invalid_expansions"] <= args.max_invalid_expansions
    )
    complete = complete and (
        totals["invalid_validations"] <= args.max_invalid_validations
    )
    if args.require_state_update:
        complete = complete and totals["accepted_state_updates"] > 0
    return {
        "protocol": PLAN_PROTOCOL,
        "complete": complete,
        "raw_root": str(raw_root),
        "raw_count": len(paths),
        "subsets": dict(sorted(subsets.items())),
        "totals": dict(totals),
        "invalid_files": invalid_files,
    }


def main() -> None:
    args = parse_args()
    result = audit(args.raw_root, args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
