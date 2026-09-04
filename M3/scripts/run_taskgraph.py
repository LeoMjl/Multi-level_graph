from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ours inside the official StableToolBench environment."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--input-query-file", required=True)
    parser.add_argument("--output-answer-file", required=True)
    parser.add_argument("--tool-root-dir", required=True)
    parser.add_argument("--openai-key", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--chatgpt-model", required=True)
    parser.add_argument("--toolbench-key", default="")
    parser.add_argument("--num-thread", type=int, default=1)
    parser.add_argument("--single-chain-max-step", type=int, default=50)
    parser.add_argument("--max-query-count", type=int, default=200)
    parser.add_argument("--max-observation-length", type=int, default=1024)
    parser.add_argument("--observ-compress-method", default="truncate")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--ablation-mode",
        choices=(
            "full", "static_l3", "hard_dependency_only",
            "no_structured_l4", "flat_context",
        ),
        default="full",
    )
    parser.add_argument("--sampling-seed", type=int, default=20260831)
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.set_defaults(
        method="MLG",
        backbone_model="chatgpt_function",
        model_path="",
        lora=False,
        lora_path="",
        max_source_sequence_length=4096,
        max_sequence_length=8192,
        rapidapi_key="",
        use_rapidapi_key=False,
        api_customization=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_root = args.official_root / "toolbench" / "inference"
    if not inference_root.is_dir():
        raise FileNotFoundError(
            f"Official StableToolBench inference package is missing: {inference_root}"
        )
    sys.path.insert(0, str(args.official_root))
    sys.path.insert(0, str(inference_root))

    from toolbench.inference.Downstream_tasks.rapidapi_multithread import (
        pipeline_runner,
    )
    from toolbench.inference.LLM.chatgpt_function_model import ChatGPTFunction

    from mlg.stabletoolbench.ablation import TaskGraphAblation
    from mlg.stabletoolbench.graph_chain import StableToolBenchGraphChain

    ablation = TaskGraphAblation(
        mode=args.ablation_mode,
        sampling_seed=args.sampling_seed,
        action_temperature=args.action_temperature,
    )

    class GraphPipelineRunner(pipeline_runner):
        def method_converter(
            self,
            backbone_model,
            openai_key,
            method,
            env,
            process_id,
            single_chain_max_step=12,
            max_query_count=60,
            callbacks=None,
        ):
            if method != "MLG":
                return super().method_converter(
                    backbone_model,
                    openai_key,
                    method,
                    env,
                    process_id,
                    single_chain_max_step,
                    max_query_count,
                    callbacks,
                )
            llm = ChatGPTFunction(
                model=self.args.chatgpt_model,
                openai_key=openai_key,
                base_url=self.args.base_url,
            )
            chain = StableToolBenchGraphChain(
                llm=llm,
                io_func=env,
                process_id=process_id,
                ablation=ablation,
            )
            result = chain.start(
                pass_at=1,
                single_chain_max_step=single_chain_max_step,
                answer=1,
            )
            return chain, result

    GraphPipelineRunner(args).run()


if __name__ == "__main__":
    main()
