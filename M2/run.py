from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mlg.data.fetch import fetch_datasets
from mlg.data.prepare import prepare_experiment
from mlg.experiments.runner import run_suite


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Run the M2 LoCoMo summarization experiment")
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="Download the official benchmark")
    prepare = commands.add_parser("prepare", help="Prepare local benchmark records")
    prepare.add_argument("--limit", type=int, default=0)
    run = commands.add_parser("run", help="Run all M2 methods")
    run.add_argument(
        "--methods",
        default="ours_full,official_base,official_long_context,official_incremental",
    )
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--seeds", default="1")
    run.add_argument("--model-config", type=Path)
    run.add_argument("--model-profiles", default="")
    run.add_argument("--judge-profile", default="")
    run.add_argument("--run-dir", type=Path, default=ROOT / "results")
    return cli


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "fetch":
        for manifest in fetch_datasets("locomo", "classic"):
            print(f"{manifest['dataset']}: {manifest['status']}")
        return
    if args.command == "prepare":
        print(prepare_experiment("m2", args.limit)[0])
        return
    result = run_suite(
        suite="main",
        experiments=["m2"],
        methods=[item.strip() for item in args.methods.split(",") if item.strip()],
        limit=args.limit,
        seeds=[int(item.strip()) for item in args.seeds.split(",") if item.strip()],
        model_config=args.model_config,
        model_profiles=[item.strip() for item in args.model_profiles.split(",") if item.strip()] or None,
        judge_profile=args.judge_profile,
        run_dir=args.run_dir,
    )
    print(f"run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
