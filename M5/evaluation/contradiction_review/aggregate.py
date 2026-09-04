from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator


HERE = Path(__file__).resolve().parent
EVALUATION = HERE.parent
PACKETS = EVALUATION / "packets"
RESULTS = HERE / "results"
SCHEMA = HERE / "result.schema.json"
ASSIGNMENTS = EVALUATION / "controller" / "assignment_key.json"
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CATEGORY_ORDER = ("person", "item", "time", "space", "world_rule", "causal")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def packet_contract(eid: str):
    packet = PACKETS / eid
    manifest = read_json(packet / "review_manifest.json")
    index_rows = read_json(packet / "hook_chapter_index.json")
    chapters_by_hook = {
        row["hook_id"]: {int(ch) for ch in row["chapters"]} for row in index_rows
    }
    valid_predicates = {}
    for hook in manifest["hooks"]:
        valid_predicates[hook["hook_id"]] = {
            item["predicate_id"]
            for field in ("seed_predicates", "state_requirements")
            for item in hook[field]
        }
    unique_chapters = sorted({ch for values in chapters_by_hook.values() for ch in values})
    chapter_text = {
        ch: (packet / "novel" / f"chapter_{ch:03d}.md").read_text(encoding="utf-8")
        for ch in unique_chapters
    }
    han_chars = sum(len(HAN_RE.findall(text)) for text in chapter_text.values())
    return chapters_by_hook, valid_predicates, unique_chapters, chapter_text, han_chars


def validate_result(eid: str):
    result_path = RESULTS / eid / "result.json"
    require(result_path.exists(), f"missing result: {result_path}")
    result = read_json(result_path)
    schema = read_json(SCHEMA)
    errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda e: list(e.path))
    require(not errors, f"{eid} schema error: {errors[0].message if errors else ''}")
    require(result["anonymous_id"] == eid, f"{eid} anonymous_id mismatch")

    chapters_by_hook, valid_predicates, chapters, texts, han_chars = packet_contract(eid)
    seen_keys = set()
    for item in result["items"]:
        hook = item["hook_id"]
        key = item["conflict_key"]
        require(key.startswith(hook + "|"), f"{eid} conflict_key hook mismatch: {key}")
        require(key not in seen_keys, f"{eid} duplicate conflict_key: {key}")
        seen_keys.add(key)
        require(
            set(item["predicate_ids"]) <= valid_predicates[hook],
            f"{eid} invalid predicate id in {key}",
        )
        source = item["source"]
        target = item["target"]
        source_ch = int(source["chapter_id"])
        target_ch = int(target["chapter_id"])
        require(source_ch < target_ch, f"{eid} non-forward anchors in {key}")
        require(source_ch in chapters_by_hook[hook], f"{eid} source outside hook index: {key}")
        require(target_ch in chapters_by_hook[hook], f"{eid} target outside hook index: {key}")
        require(source["quote"] in texts[source_ch], f"{eid} source quote not exact: {key}")
        require(target["quote"] in texts[target_ch], f"{eid} target quote not exact: {key}")

    categories = Counter(item["category"] for item in result["items"])
    count = len(result["items"])
    density = count / han_chars * 10000 if han_chars else 0.0
    return {
        "anonymous_id": eid,
        "count": count,
        "reviewed_unique_chapters": len(chapters),
        "reviewed_unique_han_chars": han_chars,
        "density_per_10000": round(density, 4),
        "excluded_candidates": len(result["excluded_candidates"]),
        **{f"category_{name}": categories[name] for name in CATEGORY_ORDER},
    }


def write_outputs(rows):
    assignments = read_json(ASSIGNMENTS)["assignments"]
    for row in rows:
        row["method"] = assignments[row["anonymous_id"]]
    order = {name: i for i, name in enumerate(("B0", "B1", "B2", "B3", "B4", "TaskGraph"))}
    rows.sort(key=lambda row: order[row["method"]])

    (HERE / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (HERE / "summary.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# M5 固定钩子轨迹长程明确矛盾重评",
        "",
        "## Material Passport",
        "",
        "- ID: M5-contradiction-reaudit-20260830",
        "- Type: Validation Report",
        "- Verification Status: ANALYZED",
        "- Evidence: six anonymous gpt-5.6-sol/medium reviews with exact-anchor validation",
        "",
        "| 方法 | 明确矛盾数 | 去重审查章节 | 审查汉字 | 长程矛盾/万字 | 排除候选 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["method"]} | {row["count"]} | {row["reviewed_unique_chapters"]} | '
            f'{row["reviewed_unique_han_chars"]:,} | {row["density_per_10000"]:.4f} | '
            f'{row["excluded_candidates"]} |'
        )
    lines += [
        "",
        "## 冲突类型",
        "",
        "| 方法 | 人物 | 物品 | 时间 | 空间 | 世界规则 | 因果 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["method"]} | {row["category_person"]} | {row["category_item"]} | '
            f'{row["category_time"]} | {row["category_space"]} | '
            f'{row["category_world_rule"]} | {row["category_causal"]} |'
        )
    lines += [
        "",
        "口径：同一批10条完整钩子轨迹；同一实体—属性的重复漂移只计一次；",
        "未提及、遗忘、合理状态变化和仅违反金标预期均不计。该结果不是全书穷尽计数。",
        "每部小说由一名模型评审；尚无第二评审或人工仲裁，因此不报告评审者一致性。",
        "",
    ]
    (HERE / "SUMMARY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = [validate_result(f"E{i:02d}") for i in range(1, 7)]
    write_outputs(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
