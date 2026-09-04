from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a local StableToolBench virtual-server configuration."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-folder", type=Path, required=True)
    parser.add_argument("--cache-folder", type=Path, required=True)
    parser.add_argument("--toolbench-url", default="http://127.0.0.1:8000/rapidapi")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "api_key": args.api_key,
        "api_base": args.api_base,
        "model": args.model,
        "temperature": 0,
        "toolbench_url": args.toolbench_url,
        "tools_folder": str(args.tools_folder.resolve()),
        "cache_folder": str(args.cache_folder.resolve()),
        "is_save": True,
        "port": args.port,
        "log_file": str(args.log_file.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_folder.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model": args.model, "port": args.port}))


if __name__ == "__main__":
    main()
