from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def standardize(value: str) -> str:
    normalized = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value)
    normalized = re.sub(r"(_)\1+", "_", normalized).lower().strip("_")
    if normalized and normalized[0].isdigit():
        normalized = "get_" + normalized
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the official StableToolBench query API definitions in "
            "the ToolBench toolenv directory format."
        )
    )
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tools: dict[tuple[str, str], dict[str, Any]] = {}
    for query_path in sorted((args.query_root / "test_instruction").glob("*.json")):
        rows = json.loads(query_path.read_text(encoding="utf-8"))
        for row in rows:
            for api in row.get("api_list", []):
                category = str(api["category_name"])
                tool_name = str(api["tool_name"])
                key = (category, tool_name)
                tool = tools.setdefault(
                    key,
                    {
                        "tool_name": tool_name,
                        "tool_description": "",
                        "api_list": [],
                    },
                )
                api_name = str(api["api_name"])
                existing = {
                    str(item.get("name", "")): item for item in tool["api_list"]
                }
                normalized_api = {
                    "name": api_name,
                    "description": str(api.get("api_description", "")),
                    "required_parameters": list(api.get("required_parameters", [])),
                    "optional_parameters": list(api.get("optional_parameters", [])),
                    "method": str(api.get("method", "")),
                    "template_response": api.get("template_response", {}),
                }
                if api_name in existing and existing[api_name] != normalized_api:
                    raise ValueError(
                        f"Conflicting official API definitions: {category}/"
                        f"{tool_name}/{api_name}"
                    )
                if api_name not in existing:
                    tool["api_list"].append(normalized_api)
    for (category, tool_name), tool in tools.items():
        descriptions = [
            str(api.get("description", "")).strip()
            for api in tool["api_list"]
            if str(api.get("description", "")).strip()
        ]
        tool["tool_description"] = " ".join(dict.fromkeys(descriptions))[:1024]
        category_dir = args.output / category
        category_dir.mkdir(parents=True, exist_ok=True)
        output_path = category_dir / f"{standardize(tool_name)}.json"
        output_path.write_text(
            json.dumps(tool, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "tool_count": len(tools),
                "output": str(args.output),
                "source": str(args.query_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
