MAIN_EXPERIMENTS = ["m1", "m2", "m3", "m4"]
ABLATION_EXPERIMENTS = ["m1", "m2", "m3"]
AUXILIARY_EXPERIMENTS = ["a1", "a2", "a3", "a4", "a5"]
ALL_EXPERIMENTS = MAIN_EXPERIMENTS + AUXILIARY_EXPERIMENTS

# Suite aliases used by current scripts.
CORE_EXPERIMENTS = MAIN_EXPERIMENTS
FULL_EXPERIMENTS = ALL_EXPERIMENTS
LEGACY_EXPERIMENTS = ["exp3", "exp4", "exp5"]

M1_METHODS = [
    "ours_full",
    "atomic_rag",
    "bounded_context",
]

# Official upstream baselines run through scripts/run_longmemeval_v2_official.ps1,
# not through the project-native experiment runner.
M1_OFFICIAL_METHODS = [
    "no_retrieval",
    "rag_query_to_slice_text",
    "agentrunbook_r_text",
]
M1_ALL_METHODS = M1_METHODS + M1_OFFICIAL_METHODS

# M2 follows the official LoCoMo event-summarization protocol.
M2_METHODS = [
    "ours_full",
    "official_base",
    "official_long_context",
    "official_incremental",
]

M3_METHODS = ["ours_full"]

# StableToolBench baselines run through the pinned upstream harness, not through
# the project-native independent-Episode runner.
M3_OFFICIAL_METHODS = [
    "official_stabletoolbench_cot",
    "official_stabletoolbench_dfs",
]
M3_ALL_METHODS = M3_METHODS + M3_OFFICIAL_METHODS

M4_METHODS = ["ours_full"]

# M4 is executed by scripts/run_m4_document_qa.py because its shared document
# indexes do not fit the generic independent-Episode runner.
M4_OFFICIAL_METHODS = [
    "full_text",
    "raptor",
    "graphrag",
    "memorystream",
]
M4_ALL_METHODS = M4_OFFICIAL_METHODS + M4_METHODS
MAIN_METHODS = M1_METHODS + [
    method for method in M2_METHODS + M3_METHODS + M4_METHODS if method not in M1_METHODS
]

ABLATION_METHODS = [
    "ours_full",
    "ours_wo_hier",
    "ours_wo_mainline",
    "ours_wo_dep",
    "ours_wo_active_pool",
]

BASE_METHODS = ["ours_full", "truncated_context", "raw_rag", "plan_and_solve"]

METHOD_DISPLAY_NAMES = {
    "ours_full": "TaskGraph",
    "truncated_context": "Truncated Context (Head+Tail 48K)",
    "bounded_context": "Bounded Context (Model Window)",
    "raw_rag": "Raw RAG",
    "atomic_rag": "Atomic RAG",
    "plan_and_solve": "Plan-and-Solve",
    "official_base": "Official Base (4K)",
    "official_long_context": "Official Long-context",
    "official_incremental": "Official Incremental Summarization",
    "official_stabletoolbench_cot": "Official StableToolBench CoT@1",
    "official_stabletoolbench_dfs": "Official StableToolBench DFS_woFilter_w2",
    "full_text": "Full Text",
    "raptor": "RAPTOR",
    "graphrag": "GraphRAG",
    "memorystream": "MemoryStream",
    "ours_wo_hier": "TaskGraph-w/o-Hier",
    "ours_wo_mainline": "TaskGraph-w/o-Mainline",
    "ours_wo_dep": "TaskGraph-w/o-Dep",
    "ours_wo_active_pool": "TaskGraph-w/o-ActivePool",
}

EXPERIMENT_METRICS = {
    "m1": [
        "official_qa_accuracy",
        "classification_accuracy",
        "domain_accuracy",
        "memory_build_time_ms",
        "memory_query_time_ms",
        "reader_generation_time_ms",
        "end_to_end_time_ms",
        "api_total_tokens",
    ],
    "m2": [
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "factscore_precision",
        "factscore_recall",
        "factscore_f1",
    ],
    "m3": [
        "SoPR",
        "SoWR",
        "FAC",
    ],
    "m4": [
        "binary_judge_accuracy",
        "easy_accuracy",
        "hard_accuracy",
        "inference_accuracy",
        "comparison_accuracy",
        "temporal_accuracy",
        "rouge_l_recall",
    ],
    "a1": ["exact_match", "f1", "evidence_recall", "graph_dependency_recall", "token_usage", "latency_ms"],
    "a2": ["exact_match", "f1", "supporting_fact_recall", "dependency_recall", "token_usage", "latency_ms"],
    "a3": ["f1", "evidence_recall", "evidence_precision", "token_usage", "latency_ms"],
    "a4": ["exact_match", "f1", "rouge_l", "retrieval_accuracy", "token_usage", "latency_ms"],
    "a5": ["retrieval_accuracy", "token_usage", "latency_ms"],
}

PRIMARY_EXPERIMENTS = MAIN_EXPERIMENTS

EXPERIMENT_METHODS = {
    "m1": M1_METHODS,
    "m2": M2_METHODS,
    "m3": M3_METHODS,
    "m4": M4_METHODS,
    "ablation:m1": ABLATION_METHODS,
    "ablation:m2": M2_METHODS,
    "ablation:m3": ABLATION_METHODS,
    "a1": BASE_METHODS + ["ours_wo_dep"],
    "a2": BASE_METHODS + ["ours_wo_dep"],
    "a3": BASE_METHODS + ["ours_wo_dep"],
    "a4": BASE_METHODS,
    "a5": ["ours_full", "truncated_context", "raw_rag"],
    "exp3": ["ours_full", "truncated_context", "raw_rag", "plan_and_solve"],
    "exp4": ["ours_full", "truncated_context", "raw_rag"],
    "exp5": ["ours_full", "truncated_context", "raw_rag"],
}


def experiments_for_suite(suite: str) -> list[str]:
    normalized = suite.lower()
    if normalized in {"main", "core"}:
        return MAIN_EXPERIMENTS
    if normalized == "ablation":
        return ABLATION_EXPERIMENTS
    if normalized == "aux":
        return AUXILIARY_EXPERIMENTS
    if normalized in {"all", "full"}:
        return ALL_EXPERIMENTS
    if normalized == "legacy":
        return LEGACY_EXPERIMENTS
    raise ValueError("suite must be main, ablation, aux, all, core, full, or legacy")


def experiment_role(experiment: str) -> str:
    exp = experiment.lower()
    if exp in MAIN_EXPERIMENTS:
        return "main"
    if exp in AUXILIARY_EXPERIMENTS:
        return "auxiliary"
    return "legacy"


def methods_for_experiment(experiment: str, requested_methods: list[str]) -> list[str]:
    allowed = EXPERIMENT_METHODS.get(experiment.lower(), requested_methods)
    return [method for method in requested_methods if method in allowed]


def methods_for_suite_experiment(suite: str, experiment: str, requested_methods: list[str]) -> list[str]:
    if suite.lower() == "ablation":
        allowed = EXPERIMENT_METHODS.get(f"ablation:{experiment.lower()}", ABLATION_METHODS)
        return [method for method in requested_methods if method in allowed]
    return methods_for_experiment(experiment, requested_methods)


def default_methods_for_suite(suite: str) -> list[str]:
    normalized = suite.lower()
    if normalized == "ablation":
        return list(ABLATION_METHODS)
    if normalized in {"main", "core"}:
        return list(MAIN_METHODS)
    if normalized in {"aux", "all", "full"}:
        return list(BASE_METHODS + ["ours_wo_dep"])
    if normalized == "legacy":
        return ["ours_full", "truncated_context", "raw_rag", "plan_and_solve"]
    return list(MAIN_METHODS)


def allowed_methods_for_suite_experiment(suite: str, experiment: str) -> list[str]:
    if suite.lower() == "ablation":
        return list(EXPERIMENT_METHODS.get(f"ablation:{experiment.lower()}", ABLATION_METHODS))
    return list(EXPERIMENT_METHODS.get(experiment.lower(), default_methods_for_suite(suite)))


def skipped_methods_for_experiment(experiment: str, requested_methods: list[str]) -> list[str]:
    selected = set(methods_for_experiment(experiment, requested_methods))
    return [method for method in requested_methods if method not in selected]


def skipped_methods_for_suite_experiment(suite: str, experiment: str, requested_methods: list[str]) -> list[str]:
    selected = set(methods_for_suite_experiment(suite, experiment, requested_methods))
    return [method for method in requested_methods if method not in selected]
