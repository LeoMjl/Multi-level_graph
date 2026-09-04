from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


SUBSETS = ("G2_instruction", "G2_category", "G3_instruction")


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def function_name(tool: dict) -> str:
    nested = tool.get("function")
    if not isinstance(nested, dict) or not nested.get("name"):
        raise ValueError("Tool definition has no function name")
    return str(nested["name"])


def without_name(tool: dict) -> dict:
    normalized = copy.deepcopy(tool)
    normalized["function"]["name"] = "__FUNCTION_NAME__"
    return normalized


def replace_names(value, mapping: dict[str, str], pattern: re.Pattern | None):
    if isinstance(value, str):
        return pattern.sub(lambda match: mapping[match.group(0)], value) if pattern else value
    if isinstance(value, list):
        return [replace_names(item, mapping, pattern) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_names(item, mapping, pattern)
            for key, item in value.items()
        }
    return value


def normalize_candidate(reference: dict, candidate: dict) -> tuple[dict, int]:
    if str(reference.get("query", "")) != str(candidate.get("query", "")):
        raise ValueError("Candidate query differs from frozen reference")
    ref_tools = reference.get("available_tools", [])
    candidate_tools = candidate.get("available_tools", [])
    if len(ref_tools) != len(candidate_tools):
        raise ValueError("Candidate tool count differs from frozen reference")

    mapping: dict[str, str] = {}
    for ref_tool, candidate_tool in zip(ref_tools, candidate_tools):
        if without_name(ref_tool) != without_name(candidate_tool):
            raise ValueError("Candidate tool schema differs from frozen reference")
        source_name = function_name(candidate_tool)
        target_name = function_name(ref_tool)
        prior = mapping.get(source_name)
        if prior is not None and prior != target_name:
            raise ValueError("One candidate tool name maps to multiple reference names")
        mapping[source_name] = target_name

    changes = sum(source != target for source, target in mapping.items())
    replacements = {source: target for source, target in mapping.items() if source != target}
    pattern = None
    if replacements:
        pattern = re.compile(
            "|".join(re.escape(name) for name in sorted(replacements, key=len, reverse=True))
        )
    normalized = replace_names(copy.deepcopy(candidate), replacements, pattern)
    normalized["available_tools"] = copy.deepcopy(ref_tools)
    return normalized, changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize M3 candidate tool aliases to a frozen reference."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    audit: dict[str, object] = {}
    for subset in SUBSETS:
        reference = load_object(
            args.input_root / args.reference / f"{subset}.json"
        )
        reference_output = args.output_root / args.reference / f"{subset}.json"
        reference_output.parent.mkdir(parents=True, exist_ok=True)
        reference_output.write_text(
            json.dumps(reference, ensure_ascii=False), encoding="utf-8"
        )
        for candidate_name in args.candidates:
            candidate = load_object(
                args.input_root / candidate_name / f"{subset}.json"
            )
            if set(reference) != set(candidate):
                raise ValueError(f"ID mismatch for {candidate_name}/{subset}")
            normalized: dict[str, object] = {}
            changed_queries = 0
            changed_aliases = 0
            for query_id in reference:
                normalized_item, changes = normalize_candidate(
                    reference[query_id], candidate[query_id]
                )
                normalized[query_id] = normalized_item
                changed_queries += int(changes > 0)
                changed_aliases += changes
            output = args.output_root / candidate_name / f"{subset}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(normalized, ensure_ascii=False), encoding="utf-8"
            )
            audit[f"{candidate_name}/{subset}"] = {
                "query_count": len(normalized),
                "queries_with_alias_changes": changed_queries,
                "tool_alias_changes": changed_aliases,
                "query_text_changed": False,
                "tool_schema_changed": False,
                "answer_semantics_changed": False,
            }

    payload = {
        "protocol": "frozen_qwen_tool_alias_normalization_v1",
        "reference": args.reference,
        "complete": True,
        "subsets": audit,
    }
    args.audit.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
