from __future__ import annotations

import argparse
import json
from pathlib import Path


SUBSETS = (
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)
PROGRESSIVE_PROTOCOL = "taskgraph_progressive_v4"
TASKGRAPH_METHODS = {"ours_progressive"}
RAW_SUFFIXES = {
    "ours_progressive": "MLG",
    "official_stabletoolbench_cot": "CoT@1",
    "official_stabletoolbench_dfs": "DFS_woFilter_w2",
}


def _valid_graph(graph: object) -> bool:
    if not isinstance(graph, dict) or graph.get("schema") != "taskgraph-v1":
        return False
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    return (
        isinstance(nodes, list)
        and isinstance(edges, list)
        and all(isinstance(node, dict) for node in nodes)
        and all(isinstance(edge, dict) for edge in edges)
        and any(
            node.get("sub_type") in {"PlannedStep", "ExpandedStep"}
            for node in nodes
        )
        and any(node.get("sub_type") == "FinalAnswerStep" for node in nodes)
    )


def _valid_progressive_structure(payload: dict) -> bool:
    coarse = payload.get("mlg_coarse_plan")
    expansion = payload.get("mlg_stage_expansion")
    dependency = payload.get("mlg_dependency_construction")
    state_updates = payload.get("mlg_state_updates")
    pruning = payload.get("mlg_context_pruning")
    routing = payload.get("mlg_action_routing")
    validation = payload.get("mlg_result_validation")
    graph = payload.get("mlg_observable_graph")
    tracked = (
        coarse,
        expansion,
        dependency,
        state_updates,
        pruning,
        routing,
        validation,
        graph,
    )
    if not all(isinstance(item, dict) for item in tracked):
        return False
    routing_trace = routing.get("trace")
    expansion_trace = expansion.get("trace")
    writeback_trace = state_updates.get("writeback_trace")
    traces = (
        expansion_trace,
        dependency.get("trace"),
        state_updates.get("trace"),
        pruning.get("trace"),
        routing_trace,
        validation.get("trace"),
    )
    structurally_valid = (
        isinstance(coarse.get("stages"), list)
        and isinstance(coarse.get("global_state"), list)
        and all(isinstance(trace, list) for trace in traces)
        and isinstance(writeback_trace, list)
        and all(isinstance(item, dict) for item in writeback_trace)
        and all(
            isinstance(item, dict)
            for trace in traces
            for item in trace
        )
        and all(
            item.get("executed_tool_call_count", 2) <= 1
            for item in routing_trace
        )
        and _valid_graph(graph)
    )
    if not structurally_valid:
        return False
    if any(
        item.get("phase") == "post_termination_budget"
        for item in expansion_trace
    ):
        return False
    answer = payload.get("answer_generation")
    if isinstance(answer, dict) and answer.get("valid_data") is True:
        if any(item.get("source") == "invalid" for item in expansion_trace):
            return False
        coarse_nodes = [
            node for node in graph.get("nodes", [])
            if node.get("sub_type") == "CoarseStage"
        ]
        if any(
            node.get("metadata", {}).get("expansion_valid") is False
            for node in coarse_nodes
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate raw and converted M3 artifacts against official query IDs."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--method", choices=sorted(RAW_SUFFIXES), default="ours_progressive"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    subset_audit: dict[str, object] = {}
    complete = True
    for subset in SUBSETS:
        query_path = (
            args.official_root
            / "solvable_queries"
            / "test_instruction"
            / f"{subset}.json"
        )
        expected = {
            str(item["query_id"])
            for item in json.loads(query_path.read_text(encoding="utf-8"))
        }
        raw_dir = args.output_root / "raw" / args.method / subset
        suffix = RAW_SUFFIXES[args.method]
        filename_suffix = f"_{suffix}.json"
        raw_paths = {
            path.name[: -len(filename_suffix)]: path
            for path in raw_dir.glob(f"*_{suffix}.json")
        }
        raw_ids = set(raw_paths)
        invalid_json_ids = []
        invalid_protocol_ids = []
        invalid_structure_ids = []
        for query_id, path in raw_paths.items():
            try:
                raw_payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                invalid_json_ids.append(query_id)
                continue
            if args.method in TASKGRAPH_METHODS:
                if raw_payload.get("mlg_protocol") != PROGRESSIVE_PROTOCOL:
                    invalid_protocol_ids.append(query_id)
                elif not _valid_progressive_structure(raw_payload):
                    invalid_structure_ids.append(query_id)
        converted_path = (
            args.output_root / "converted" / args.method / f"{subset}.json"
        )
        converted = json.loads(converted_path.read_text(encoding="utf-8"))
        converted_ids = {str(query_id) for query_id in converted}
        subset_complete = (
            expected == raw_ids == converted_ids
            and not invalid_json_ids
            and not invalid_protocol_ids
            and not invalid_structure_ids
        )
        complete = complete and subset_complete
        subset_audit[subset] = {
            "expected_count": len(expected),
            "raw_count": len(raw_ids),
            "converted_count": len(converted_ids),
            "missing_raw_ids": sorted(expected - raw_ids),
            "extra_raw_ids": sorted(raw_ids - expected),
            "invalid_json_ids": sorted(invalid_json_ids),
            "invalid_protocol_ids": sorted(invalid_protocol_ids),
            "invalid_structure_ids": sorted(invalid_structure_ids),
            "missing_converted_ids": sorted(expected - converted_ids),
            "extra_converted_ids": sorted(converted_ids - expected),
            "complete": subset_complete,
        }

    payload = {
        "method": args.method,
        "required_protocol": (
            PROGRESSIVE_PROTOCOL if args.method in TASKGRAPH_METHODS else None
        ),
        "complete": complete,
        "expected_total": sum(
            int(item["expected_count"]) for item in subset_audit.values()
        ),
        "subsets": subset_audit,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
