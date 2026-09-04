from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlg.metrics.stabletoolbench import (
    OFFICIAL_SUBSETS,
    load_official_json,
    macro_average,
    official_fac,
    official_sopr,
    official_sowr,
)

M3_G23_SUBSETS = ("G2_instruction", "G2_category", "G3_instruction")
PREFERENCE_PROTOCOL = "alternating_candidate_order_v2"


def validate_preference_artifact(payload: dict, path: Path) -> None:
    invalid = []
    for query_id, item in payload.items():
        if not isinstance(item, dict) or item.get("_protocol") != PREFERENCE_PROTOCOL:
            invalid.append(str(query_id))
            continue
        round_ids = sorted(
            int(match.group(1))
            for key in item
            if (match := re.fullmatch(r"round_(\d+)", str(key)))
        )
        if not round_ids or any(
            item.get(f"round_{round_id}") != "complete"
            or item.get(f"round_{round_id}_order") != (
                "reference_first" if round_id % 2 == 0 else "output_first"
            )
            for round_id in round_ids
        ):
            invalid.append(str(query_id))
    if invalid:
        preview = ", ".join(invalid[:10])
        raise ValueError(
            f"Preference artifact is incomplete or uses a legacy order protocol: "
            f"{path} (invalid query ids: {preview})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import only official StableToolBench evaluator artifacts."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=OFFICIAL_SUBSETS,
        default=list(M3_G23_SUBSETS),
    )
    parser.add_argument("--pass-rate-root", type=Path)
    parser.add_argument("--preference-root", type=Path)
    parser.add_argument("--reference-model", default="")
    parser.add_argument("--fac-root", type=Path)
    parser.add_argument("--sopr-sowr-judge-model", default="")
    parser.add_argument("--sopr-sowr-thinking-mode", default="")
    parser.add_argument("--sopr-sowr-temperature", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics: dict[str, object] = {}
    if args.pass_rate_root:
        sopr = {}
        for subset in args.subsets:
            path = (
                args.pass_rate_root
                / args.method
                / f"{subset}_{args.method}.json"
            )
            sopr[subset] = official_sopr(load_official_json(path))
        metrics["SoPR"] = {
            "subsets": sopr,
            "macro_average": macro_average(sopr),
        }
    if args.preference_root:
        if not args.reference_model:
            raise ValueError("--reference-model is required with --preference-root")
        sowr = {}
        for subset in args.subsets:
            path = (
                args.preference_root
                / f"{subset}_{args.reference_model}_{args.method}.json"
            )
            preference_payload = load_official_json(path)
            validate_preference_artifact(preference_payload, path)
            sowr[subset] = official_sowr(
                preference_payload,
                reference_model=args.reference_model,
                candidate_model=args.method,
            )
        metrics["SoWR"] = {
            "reference_model": args.reference_model,
            "subsets": sowr,
            "macro_average": macro_average(sowr),
        }
    if args.fac_root:
        fac = {
            subset: official_fac(args.fac_root / args.method / f"{subset}.csv")
            for subset in args.subsets
        }
        metrics["FAC"] = {
            "subsets": fac,
            "macro_average": macro_average(fac),
        }
    if not metrics:
        raise ValueError("At least one official metric artifact is required")
    output = {
        "protocol": "official_stabletoolbench",
        "method": args.method,
        "official_subsets": list(args.subsets),
        "metrics": metrics,
    }
    evaluators: dict[str, object] = {}
    if args.pass_rate_root or args.preference_root:
        evaluators["SoPR_SoWR"] = {
            "judge_model": args.sopr_sowr_judge_model,
            "thinking_mode": args.sopr_sowr_thinking_mode,
            "temperature": args.sopr_sowr_temperature,
            "metric_formula": "official_stabletoolbench",
            "preference_protocol": PREFERENCE_PROTOCOL,
            "candidate_order": "alternating_by_round",
        }
    if args.fac_root:
        evaluators["FAC"] = {
            "judge_model": "stabletoolbench/Evaluator",
            "input_scope": "query_and_final_answer",
        }
    output["evaluators"] = evaluators
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
