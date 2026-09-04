from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CANDIDATES = (
    ("official_stabletoolbench_dfs", "DFS_woFilter_w2"),
    ("ours_progressive", "TaskGraph"),
)
METRICS = ("SoPR", "FAC", "SoWR")


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro(metrics: dict, name: str) -> float | None:
    item = metrics.get(name)
    if not isinstance(item, dict) or "macro_average" not in item:
        return None
    return float(item["macro_average"])


def metric_set(metrics: dict) -> dict[str, float | None]:
    return {name: macro(metrics, name) for name in METRICS}


def delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def display(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build M3 frozen-Qwen-vs-DeepSeek comparison table."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    reference = load_object(args.reference_json)
    if reference.get("status") != "complete":
        raise ValueError("Frozen Qwen reference is not complete")
    qwen_methods = reference.get("methods")
    if not isinstance(qwen_methods, dict):
        raise ValueError("Frozen Qwen reference has no methods")

    cot = metric_set(qwen_methods["official_stabletoolbench_cot"]["metrics"])
    rows: list[dict] = []
    measured_sowr = True
    for method, label in CANDIDATES:
        summary = load_object(args.output_root / f"{method}_summary.json")
        artifact_audit = load_object(
            args.output_root / f"{method}_artifact_audit.json"
        )
        evaluation_audit = load_object(
            args.output_root / f"{method}_evaluation_audit.json"
        )
        if summary.get("method") != method:
            raise ValueError(f"Wrong method in summary for {method}")
        if not artifact_audit.get("complete") or not evaluation_audit.get("complete"):
            raise ValueError(f"Incomplete audit for {method}")

        qwen = metric_set(qwen_methods[method]["metrics"])
        deepseek = metric_set(summary["metrics"])
        measured_sowr = measured_sowr and deepseek["SoWR"] is not None
        rows.append({
            "method": method,
            "display_name": label,
            "qwen3_14b_awq": qwen,
            "deepseek_v4_flash_no_thinking": deepseek,
            "delta_vs_same_method_qwen": {
                name: delta(deepseek[name], qwen[name]) for name in METRICS
            },
            "delta_vs_frozen_qwen_cot": {
                name: delta(deepseek[name], cot[name]) for name in METRICS
            },
        })

    preference_audit = None
    if measured_sowr:
        preference_audit_path = args.output_root / "preference_audit.json"
        preference_audit = load_object(preference_audit_path)
        if (
            not preference_audit.get("complete")
            or preference_audit.get("reference") != "official_stabletoolbench_cot"
            or preference_audit.get("protocol") != "alternating_candidate_order_v2"
            or preference_audit.get("evaluate_times") != 3
        ):
            raise ValueError("Frozen-Qwen-CoT preference audit is incomplete")

    output = {
        "protocol": "official_stabletoolbench_g2_g3_291_frozen_qwen_cot",
        "frozen_baseline": {
            "backbone": "Qwen3-14B-AWQ",
            "method": "CoT@1",
            "metrics": cot,
            "rerun": False,
        },
        "candidate_execution": {
            "backbone": "DeepSeek/deepseek-v4-flash",
            "thinking": "disabled",
            "methods": [label for _, label in CANDIDATES],
        },
        "fixed_judges": {
            "SoPR": "DeepSeek/deepseek-v4-flash, thinking disabled",
            "SoWR": "DeepSeek/deepseek-v4-flash, thinking disabled",
            "FAC": "StableToolBench/Evaluator",
        },
        "primary_endpoint": "FAC",
        "new_sowr_status": (
            "measured_against_frozen_qwen_cot_3_rounds"
            if measured_sowr
            else "not_measured_without_frozen_per_query_cot"
        ),
        "preference_audit": preference_audit,
        "claim_scope": (
            "Backbone sensitivity under a frozen Qwen CoT baseline; model family "
            "and provider change, so this is not pure parameter-count causality."
        ),
        "self_judge_caveat": (
            "The DeepSeek execution backbone shares the SoPR/SoWR judge model name; "
            "FAC is the primary cross-backbone endpoint."
        ),
        "reference_sha256": sha256(args.reference_json),
        "methods": rows,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }

    lines = [
        "# M3 DeepSeek 执行骨干验证（291 题）",
        "",
        "CoT 不重跑，固定使用此前 Qwen3-14B-AWQ 的结果。数值为宏平均百分比。",
        "",
        "| 方法 | 执行骨干 | SoPR | FAC | SoWR | ΔFAC vs Qwen CoT | ΔFAC vs 同方法 Qwen |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| CoT@1（冻结） | Qwen3-14B-AWQ | {display(cot['SoPR'])} | "
        f"{display(cot['FAC'])} | — | 0.00 | — |",
    ]
    for row in rows:
        qwen = row["qwen3_14b_awq"]
        deepseek = row["deepseek_v4_flash_no_thinking"]
        lines.append(
            f"| {row['display_name']}（历史） | Qwen3-14B-AWQ | "
            f"{display(qwen['SoPR'])} | {display(qwen['FAC'])} | "
            f"{display(qwen['SoWR'])} | {display(delta(qwen['FAC'], cot['FAC']))} | 0.00 |"
        )
        lines.append(
            f"| {row['display_name']}（新） | DeepSeek-V4-Flash，无思考 | "
            f"{display(deepseek['SoPR'])} | {display(deepseek['FAC'])} | "
            f"{display(deepseek['SoWR'])} | "
            f"{display(row['delta_vs_frozen_qwen_cot']['FAC'])} | "
            f"{display(row['delta_vs_same_method_qwen']['FAC'])} |"
        )
    if measured_sowr:
        sowr_note = (
            "DeepSeek DFS/TaskGraph 的 SoWR 均以同一批冻结 Qwen3-14B-AWQ CoT@1 "
            "逐题轨迹为 reference，采用三轮 alternating_candidate_order_v2 实测。"
        )
    else:
        sowr_note = (
            "新实验没有可用的冻结逐题 CoT 候选文件，因此不计算新的 SoWR；"
            "历史 Qwen SoWR 仅作为已冻结结果展示。"
        )
    lines.extend([
        "",
        "主判据为 FAC。" + sowr_note + "DeepSeek 同时作为执行模型和 SoPR/SoWR "
        "裁判，相关指标可能存在自评偏差。模型家族与服务实现同时变化，结果不能单独归因于参数规模。",
        "",
    ])
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
