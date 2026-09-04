from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic StableToolBench smoke subset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"StableToolBench input must be a non-empty list: {args.input}")
    selected = rows[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError("--offset points beyond the available queries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "selected": len(selected),
        "offset": args.offset,
    }))


if __name__ == "__main__":
    main()
