from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class DataSource:
    name: str
    version_group: str
    url: str
    repo_dir: str
    license_note: str
    preferred_revision: str = "main"
    hf_dataset_id: str = ""
    include_files: tuple[str, ...] = ()


SOURCES: dict[str, DataSource] = {
    "longmemeval": DataSource(
        name="longmemeval",
        version_group="classic",
        url="https://github.com/xiaowu0162/longmemeval.git",
        repo_dir="longmemeval",
        license_note="Use upstream repository license and dataset terms.",
        hf_dataset_id="xiaowu0162/longmemeval",
        include_files=("README.md", "longmemeval_s", "longmemeval_oracle"),
    ),
    "longmemeval_v2": DataSource(
        name="longmemeval_v2",
        version_group="latest",
        url="https://github.com/xiaowu0162/LongMemEval-V2.git",
        repo_dir="longmemeval_v2",
        license_note="Official LongMemEval-V2 benchmark; use the released downloader, evaluator, and Apache-2.0 terms.",
        hf_dataset_id="xiaowu0162/longmemeval-v2",
    ),
    "toolbench": DataSource(
        name="toolbench",
        version_group="classic",
        url="https://github.com/openbmb/toolbench.git",
        repo_dir="toolbench",
        license_note="Use OpenBMB ToolBench/ToolLLM dataset and ToolEnv terms.",
        preferred_revision="master",
    ),
    "hotpotqa": DataSource(
        name="hotpotqa",
        version_group="classic",
        url="https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        repo_dir="hotpotqa",
        license_note="Use upstream HotpotQA dataset terms; fallback can use LongBench hotpotqa rows.",
        hf_dataset_id="hotpotqa/hotpot_qa",
    ),
    "qasper": DataSource(
        name="qasper",
        version_group="classic",
        url="https://huggingface.co/datasets/allenai/qasper",
        repo_dir="qasper",
        license_note="Use upstream QASPER dataset terms; fallback can use LongBench qasper rows.",
        hf_dataset_id="allenai/qasper",
    ),
    "locomo": DataSource(
        name="locomo",
        version_group="classic",
        url="https://github.com/snap-research/locomo.git",
        repo_dir="locomo",
        license_note="Official Snap LoCoMo release under CC BY-NC 4.0; cite the ACL 2024 paper and repository.",
    ),
    "stabletoolbench": DataSource(
        name="stabletoolbench",
        version_group="classic",
        url="https://github.com/THUNLP-MT/StableToolBench.git",
        repo_dir="stabletoolbench",
        license_note=(
            "Official StableToolBench solvable-query benchmark, virtual API "
            "environment, and Apache-2.0 terms."
        ),
        preferred_revision="aa4ed9f4737ad98bd706663f01d63623c3427812",
    ),
    "longbench": DataSource(
        name="longbench",
        version_group="classic",
        url="https://github.com/THUDM/LongBench.git",
        repo_dir="longbench",
        license_note="Use upstream LongBench license and dataset terms.",
        hf_dataset_id="zai-org/LongBench",
        include_files=("README.md", "LongBench.py", "data.zip"),
    ),
    "longbench_v2": DataSource(
        name="longbench_v2",
        version_group="latest",
        url="https://github.com/THUDM/LongBench.git",
        repo_dir="longbench_v2",
        license_note="LongBench v2 should be reported as an extension table.",
        preferred_revision="main",
        hf_dataset_id="zai-org/LongBench-v2",
        include_files=("README.md", "data.json"),
    ),
}


def select_sources(dataset: str, version: str) -> list[DataSource]:
    if dataset == "all":
        candidates = list(SOURCES.values())
    else:
        key = dataset.lower()
        if key not in SOURCES:
            raise ValueError(f"Unknown dataset {dataset}. Known: {', '.join(sorted(SOURCES))}, all")
        candidates = [SOURCES[key]]
    if version == "both":
        return candidates
    if version not in {"classic", "latest"}:
        raise ValueError("version must be classic, latest, or both")
    return [src for src in candidates if src.version_group == version]
