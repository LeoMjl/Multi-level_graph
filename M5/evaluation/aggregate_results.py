from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent
METHODS = {
    "B0": "E03", "B1": "E05", "B2": "E06",
    "B3": "E01", "B4": "E04", "TaskGraph": "E02",
}
VALID_ROUTES = {"attempted", "auxiliary", "essential"}
CAUSAL_ROUTES = {"auxiliary", "essential"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_contradiction_reaudit(method: str, alias: str) -> dict:
    path = ROOT / "contradiction_review" / "summary.json"
    require(path.exists(), "missing unified contradiction re-audit summary")
    matches = [row for row in load(path) if row["method"] == method]
    require(len(matches) == 1, f"{method} contradiction re-audit row missing or duplicated")
    row = matches[0]
    require(row["anonymous_id"] == alias, f"{method} contradiction re-audit alias mismatch")
    require(int(row["reviewed_unique_chapters"]) == 116, f"{method} review chapter mismatch")
    require(int(row["reviewed_unique_han_chars"]) > 0, f"{method} invalid review denominator")
    return row


def validate_schema(result: dict, alias: str) -> None:
    schema = load(ROOT / "result.schema.json")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(result),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise ValueError(f"{alias} result schema error at {location}: {first.message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text)


def evidence_rows(alias: str) -> dict[tuple[str, int], dict]:
    rows = {}
    path = ROOT / "packets" / alias / "evidence_inputs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows[(item["hook_id"], int(item["chapter_id"]))] = item
    return rows


def anchor_in_text(anchor: dict, text: str) -> bool:
    quote = normalized(str(anchor.get("quote") or ""))
    return len(quote) >= 8 and quote in normalized(text)


def history_anchor_unit(anchor: dict, packet_rows: dict, hook_id: str) -> dict:
    chapter = int(anchor["chapter_id"])
    row = packet_rows.get((hook_id, chapter))
    if row is None:
        raise ValueError(f"history anchor chapter {chapter} is not in the evidence packet")
    unit_id = anchor.get("unit_id")
    units = [unit for unit in row["history_units"] if unit["unit_id"] == unit_id]
    if len(units) != 1 or not anchor_in_text(anchor, units[0]["text"]):
        raise ValueError(f"invalid history anchor {hook_id}/{chapter}/{unit_id}")
    if units[0]["sha256"] != hashlib.sha256(
        units[0]["text"].encode("utf-8")
    ).hexdigest():
        raise ValueError(f"history unit hash mismatch: {unit_id}")
    return units[0]


def validate_story_anchor(anchor: dict, run: Path) -> None:
    chapter = int(anchor["chapter_id"])
    text = (run / "chapters" / f"chapter_{chapter:03d}.md").read_text(encoding="utf-8")
    if not anchor_in_text(anchor, text):
        raise ValueError(f"invalid story anchor in chapter {chapter}")


def validate_story_evidence(anchor: dict, run: Path) -> None:
    validate_story_anchor(anchor, run)


def valid_chain(
    row: dict, manifest_hook: dict, packet_rows: dict, hook_id: str, run: Path,
) -> bool:
    if row["provenance"] not in {"D", "T"}:
        return False
    if row["prompt_disclosed"] or row["prompt_laundered"] or row["conflict"]:
        return False
    if not row["output_used"]:
        return False
    if not row["source_anchors"] or not row["history_anchors"] or not row["output_anchors"]:
        return False
    plant = int(manifest_hook["plant_chapter"])
    scoring = {int(value) for value in manifest_hook["scoring_chapters"]}
    window = range(
        int(manifest_hook["evaluation_window"][0]),
        int(manifest_hook["evaluation_window"][1]) + 1,
    )
    if any(int(anchor["chapter_id"]) not in window for anchor in row["output_anchors"]):
        return False
    for anchor in row["output_anchors"]:
        validate_story_anchor(anchor, run)
    history_anchors = sorted(row["history_anchors"], key=lambda value: value["chapter_id"])
    history_unit_ids = {str(anchor.get("unit_id") or "") for anchor in history_anchors}
    if "" in history_unit_ids:
        return False
    source_anchors_by_unit: dict[str, list[dict]] = {}
    for source_anchor in row["source_anchors"]:
        unit_id = str(source_anchor.get("unit_id") or "")
        if not unit_id or unit_id not in history_unit_ids:
            return False
        source_anchors_by_unit.setdefault(unit_id, []).append(source_anchor)
    if set(source_anchors_by_unit) != history_unit_ids:
        return False
    output_chapters = {int(anchor["chapter_id"]) for anchor in row["output_anchors"]}
    final_history_chapter = int(history_anchors[-1]["chapter_id"])
    if final_history_chapter not in (scoring & output_chapters):
        return False
    carriers = {plant}
    for anchor in history_anchors:
        chapter = int(anchor["chapter_id"])
        packet_row = packet_rows.get((hook_id, chapter))
        unit = history_anchor_unit(anchor, packet_rows, hook_id)
        unit_id = str(anchor["unit_id"])
        unit_sources = {int(value) for value in unit["source_chapters"]}
        source_anchors = source_anchors_by_unit.get(unit_id, [])
        if not source_anchors or any(
            int(source_anchor["chapter_id"]) not in unit_sources
            or int(source_anchor["chapter_id"]) >= chapter
            for source_anchor in source_anchors
        ):
            return False
        for source_anchor in source_anchors:
            validate_story_anchor(source_anchor, run)
        if not carriers.intersection(
            int(source_anchor["chapter_id"]) for source_anchor in source_anchors
        ):
            return False
        if anchor_in_text(anchor, packet_row["current_prompt"]):
            return False
        carriers.add(chapter)
    if row["provenance"] == "D":
        final_unit = history_anchor_unit(history_anchors[-1], packet_rows, hook_id)
        if (final_unit["provenance_cap"] != "D"
                or plant not in {int(value) for value in final_unit["source_chapters"]}):
            return False
    return True


def valid_local_resolution(
    resolution: dict, manifest_hook: dict, packet_rows: dict, hook_id: str, run: Path,
) -> bool:
    basic = (
        resolution["same_problem_resolution"] == "full"
        and resolution["subtype"] != "none"
        and resolution["resource_supported"]
        and resolution["causal_chain_complete"]
        and resolution["state_consistent"]
        and resolution["cost_reasonable"]
        and resolution["continuity_3_chapters"]
        and resolution["forbidden_free"]
    )
    if not basic:
        return False
    anchors = resolution["anchors"]
    roles = {anchor["role"] for anchor in anchors}
    required_roles = {"problem", "resource", "operation", "mechanism", "result", "continuity"}
    if not required_roles <= roles:
        return False
    for anchor in anchors:
        validate_story_evidence(anchor, run)
    window_start, window_end = (int(value) for value in manifest_hook["evaluation_window"])
    in_window_roles = {"operation", "mechanism", "result"}
    if any(
        anchor["role"] in in_window_roles
        and not window_start <= int(anchor["chapter_id"]) <= window_end
        for anchor in anchors
    ):
        return False
    trigger = int(manifest_hook["trigger_chapter"])
    if any(
        anchor["role"] == "problem"
        and int(anchor["chapter_id"]) not in ({trigger} | set(range(window_start, window_end + 1)))
        for anchor in anchors
    ):
        return False
    chapters_by_role = {
        role: [int(anchor["chapter_id"]) for anchor in anchors if anchor["role"] == role]
        for role in roles
    }
    operation_chapter = min(chapters_by_role["operation"])
    result_chapter = max(chapters_by_role["result"])
    if min(chapters_by_role["resource"]) > operation_chapter:
        return False
    mechanism_chapter = min(chapters_by_role["mechanism"])
    if (min(chapters_by_role["problem"]) > operation_chapter
            or operation_chapter > result_chapter
            or mechanism_chapter > result_chapter):
        return False
    continuity_limit = min(320, window_end + 3)
    if not any(
        result_chapter < chapter <= continuity_limit
        for chapter in chapters_by_role["continuity"]
    ):
        return False

    prompt_supported = bool(resolution["prompt_anchors"])
    for anchor in resolution["prompt_anchors"]:
        packet_row = packet_rows.get((hook_id, int(anchor["chapter_id"])))
        if (
            packet_row is None
            or not anchor_in_text(anchor, packet_row["current_prompt"])
        ):
            prompt_supported = False
            break
    historical_supported = bool(resolution["history_anchors"])
    for anchor in resolution["history_anchors"]:
        try:
            unit = history_anchor_unit(anchor, packet_rows, hook_id)
        except (KeyError, TypeError, ValueError):
            historical_supported = False
            break
        source_values = {
            int(value) for value in unit["source_chapters"]
        }
        if not source_values or min(source_values) >= trigger:
            historical_supported = False
            break
    verification_supported = bool(chapters_by_role.get("verification")) and any(
        chapter <= operation_chapter for chapter in chapters_by_role["verification"]
    )
    support_count = sum((prompt_supported, historical_supported, verification_supported))
    subtype = resolution["subtype"]
    return (
        (subtype == "prompt_scaffolded" and prompt_supported)
        or (subtype == "historical_alternative" and historical_supported)
        or (subtype == "window_verified" and verification_supported)
        or (subtype == "mixed" and support_count >= 2)
    )


def derive_category(hook: dict, manifest_hook: dict, packet_rows: dict, run: Path) -> str:
    checks = {item["predicate_id"]: item for item in hook["predicate_provenance"]}
    seed_ids = {item["predicate_id"] for item in manifest_hook["seed_predicates"]}
    attr_ids = {item["key_id"] for item in manifest_hook["attribution_min_keys"]}
    require(
        len(checks) == len(hook["predicate_provenance"])
        and seed_ids | attr_ids == set(checks),
        f"{hook['hook_id']} predicate IDs are missing or duplicated",
    )
    valid = {
        predicate_id: valid_chain(
            item, manifest_hook, packet_rows, hook["hook_id"], run
        )
        for predicate_id, item in checks.items()
    }
    eligible_seed = {
        predicate_id for predicate_id in seed_ids
        if not checks[predicate_id]["prompt_disclosed"]
    }
    a_min = any(valid[predicate_id] for predicate_id in attr_ids)
    a_full = (
        a_min and bool(eligible_seed)
        and all(valid[predicate_id] for predicate_id in eligible_seed)
    )
    expected_status = "full" if a_full else "min" if a_min else (
        "unidentifiable" if not eligible_seed else "none"
    )
    require(
        hook["attribution"] == {
            "a_min": a_min, "a_full": a_full, "status": expected_status,
        },
        f"{hook['hook_id']} attribution does not match its evidence chains",
    )
    gold = hook["gold"]
    if (a_full and hook["target_route"] in CAUSAL_ROUTES
            and all(gold.values())):
        return "G"
    if a_min and hook["target_route"] in VALID_ROUTES:
        return "P"
    if not a_min and valid_local_resolution(
        hook["local_resolution"], manifest_hook, packet_rows, hook["hook_id"], run
    ):
        return "L"
    return "F"


def derive_evidence_metrics(
    hooks: list[dict], manifest_hooks: dict[str, dict],
    packet_rows: dict, run: Path,
) -> dict:
    direct = transitive = none = conflict = excluded_prompt = 0
    macro_rates = []
    for hook in hooks:
        hook_id = hook["hook_id"]
        checks = {item["predicate_id"]: item for item in hook["predicate_provenance"]}
        seed_ids = {
            item["predicate_id"] for item in manifest_hooks[hook_id]["seed_predicates"]
        }
        eligible = [
            checks[predicate_id] for predicate_id in seed_ids
            if not checks[predicate_id]["prompt_disclosed"]
        ]
        excluded_prompt += len(seed_ids) - len(eligible)
        hook_hits = 0
        for item in eligible:
            if item["conflict"]:
                conflict += 1
            if valid_chain(item, manifest_hooks[hook_id], packet_rows, hook_id, run):
                if item["provenance"] == "D":
                    direct += 1
                else:
                    transitive += 1
                hook_hits += 1
            else:
                none += 1
        if eligible:
            macro_rates.append(hook_hits / len(eligible))
    scoring_rows = [row for row in packet_rows.values() if row["is_scoring_chapter"]]
    average_k = (
        sum(int(row["history_unit_count"]) for row in scoring_rows) / len(scoring_rows)
        if scoring_rows else 0.0
    )
    return {
        "macro_average": sum(macro_rates) / len(macro_rates) if macro_rates else 0.0,
        "direct": direct, "transitive": transitive, "none": none,
        "conflict": conflict, "excluded_prompt": excluded_prompt,
        "total": direct + transitive + none,
        "average_k": average_k,
    }


def normalize(method: str, alias: str) -> tuple[dict, dict[str, str]]:
    result = load(ROOT / "results" / alias / "result.json")
    objective = load(ROOT / "packets" / alias / "objective_metrics.json")
    manifest_path = ROOT / "packets" / alias / "review_manifest.json"
    manifest = load(manifest_path)
    protocol_path = ROOT.parent / "prompts" / "model_review_prompt.md"
    if result.get("schema_version") != "m5-review":
        raise RuntimeError(
            f"{alias} does not use the current review schema; rerun its reviewer "
            "with model_review_prompt.md before aggregation"
        )
    validate_schema(result, alias)
    require(result["anonymous_id"] == alias, f"{alias} anonymous ID mismatch")
    require(result["reviewer_model"] == "gpt-5.6-sol", f"{alias} reviewer mismatch")
    require(result["reasoning_effort"] == "medium", f"{alias} effort mismatch")
    require(
        result["protocol_sha256"] == sha256(protocol_path),
        f"{alias} protocol hash mismatch",
    )
    require(
        result["review_manifest_sha256"] == sha256(manifest_path),
        f"{alias} manifest hash mismatch",
    )
    require(
        manifest.get("protocol_sha256") == result["protocol_sha256"],
        f"{alias} manifest/protocol mismatch",
    )
    require(
        objective["protocol_sha256"] == result["protocol_sha256"],
        f"{alias} objective/protocol mismatch",
    )
    require(
        objective["review_manifest_sha256"] == result["review_manifest_sha256"],
        f"{alias} objective/manifest mismatch",
    )
    hooks = result["hooks"]
    require(
        len(hooks) == 10 and len(result["continuity_transition_scores"]) == 32,
        f"{alias} has an incomplete review sample",
    )
    manifest_hooks = {item["hook_id"]: item for item in manifest["hooks"]}
    packet_rows = evidence_rows(alias)
    run = ROOT.parent / method / "run"
    categories = {}
    for item in hooks:
        hook_id = item["hook_id"]
        derived = derive_category(item, manifest_hooks[hook_id], packet_rows, run)
        require(
            item["derived_category"] == derived,
            f"{alias}/{hook_id} reviewer category is not rule-derived",
        )
        categories[hook_id] = derived
    require(
        sorted(categories) == [f"H{x:02d}" for x in range(1, 11)],
        f"{alias} hook IDs are missing or duplicated",
    )
    counts = Counter(categories.values())
    require(
        sum(counts.values()) == 10 and set(counts) <= {"G", "P", "L", "F"},
        f"{alias} contains invalid derived categories",
    )

    plot = float(result["plot_coherence"]["total"])
    long_range = float(result["long_range_narrative_quality"]["total"])
    contradiction = result["contradictions"]
    initial_contradiction_count = int(contradiction["count"])
    require(
        initial_contradiction_count == len(contradiction["items"]),
        f"{alias} contradiction count mismatch",
    )
    reaudit = load_contradiction_reaudit(method, alias)
    contradiction_count = int(reaudit["count"])
    sample_chars = int(reaudit["reviewed_unique_han_chars"])
    contradiction_density = contradiction_count / sample_chars * 10000
    require(
        abs(contradiction_density - float(reaudit["density_per_10000"])) < 5e-5,
        f"{alias} contradiction re-audit density mismatch",
    )

    evidence = result["evidence_recall_at_k"]
    calculated_evidence = derive_evidence_metrics(hooks, manifest_hooks, packet_rows, run)
    evidence_macro = float(evidence["macro_average"])
    micro = evidence["micro"]
    direct = int(micro["direct"])
    transitive = int(micro["transitive"])
    none = int(micro["none"])
    conflict = int(micro["conflict"])
    excluded_prompt = int(micro["excluded_prompt"])
    micro_hits = direct + transitive
    micro_total = int(micro["total"])
    require(
        direct + transitive + none == micro_total and conflict <= none,
        f"{alias} evidence totals are internally inconsistent",
    )
    average_k = float(evidence["average_k"])
    require(
        abs(evidence_macro - calculated_evidence["macro_average"]) < 1e-6,
        f"{alias} evidence macro was not derived from predicate chains",
    )
    require(
        abs(average_k - calculated_evidence["average_k"]) < 1e-6,
        f"{alias} average k does not match physical history units",
    )
    require(
        {
            "direct": direct, "transitive": transitive, "none": none,
            "conflict": conflict, "excluded_prompt": excluded_prompt,
            "total": micro_total,
        } == {key: calculated_evidence[key] for key in (
            "direct", "transitive", "none", "conflict", "excluded_prompt", "total"
        )},
        f"{alias} evidence micro counts were not rule-derived",
    )
    state = result["state_fidelity"]
    state_num = int(state["numerator"])
    state_den = int(state["denominator"])
    require(state_den == 21, f"{alias} state denominator must be 21")

    hook_aware = counts["G"] + counts["P"]
    total_tokens = int(objective["total_token_proxy"])
    unit_cost = total_tokens / hook_aware if hook_aware else None
    row = {
        "method": method, "anonymous_id": alias,
        "G": counts["G"], "P": counts["P"], "L": counts["L"], "F": counts["F"],
        "gold_rate": counts["G"] / 10,
        "partial_hook_adherence_rate": counts["P"] / 10,
        "hook_aware_progress_rate": hook_aware / 10,
        "local_resolution_rate": counts["L"] / 10,
        "plot_coherence": round(plot, 2), "long_range_narrative_quality": round(long_range, 2),
        "contradictions": contradiction_count,
        "contradiction_sample_han_chars": sample_chars,
        "contradiction_sample_unique_chapters": int(reaudit["reviewed_unique_chapters"]),
        "contradiction_density_per_10000": round(contradiction_density, 4),
        "initial_continuity_contradictions": initial_contradiction_count,
        "total_token_proxy": total_tokens,
        "tokens_per_hook_aware_progress": round(unit_cost, 2) if unit_cost is not None else None,
        "evidence_recall_at_k": round(evidence_macro, 4),
        "evidence_micro_hits": micro_hits, "evidence_micro_total": micro_total,
        "evidence_direct": direct, "evidence_transitive": transitive,
        "evidence_none": none, "evidence_conflict": conflict,
        "evidence_excluded_prompt": excluded_prompt,
        "average_k": round(average_k, 4),
        "state_pass": state_num, "state_total": state_den,
        "state_fidelity": round(state_num / state_den, 4),
        "evaluator_agreement": "NA_pending_human_annotation",
    }
    return row, categories


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def write_method_report(alias: str, row: dict, categories: dict[str, str]) -> None:
    result = load(ROOT / "results" / alias / "result.json")
    hooks = {item["hook_id"]: item for item in result["hooks"]}
    cost = (
        "NA"
        if row["tokens_per_hook_aware_progress"] is None
        else f'{row["tokens_per_hook_aware_progress"]:,.0f}'
    )
    lines = [
        "## Material Passport", "",
        "- Mode: model review / descriptive validation",
        f'- Reviewer: `{result["reviewer_model"]}` / `{result["reasoning_effort"]}`',
        f'- Schema: `{result["schema_version"]}`',
        "- Verification Status: `ANALYZED`", "",
        f"# {alias} 匿名模型评审报告", "", "## 正文指标", "",
        "| Gold | Partial Hook Adherence | Hook-aware Progress | Plot | Long-range | 长程矛盾/万字 | token/可归因推进 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f'| {pct(row["gold_rate"])} | '
        f'{pct(row["partial_hook_adherence_rate"])} | '
        f'{pct(row["hook_aware_progress_rate"])} | '
        f'{row["plot_coherence"]:.2f} | '
        f'{row["long_range_narrative_quality"]:.2f} | '
        f'{row["contradiction_density_per_10000"]:.4f} | {cost} |',
        "", "## 附录指标", "",
        "| Local | Memory Recall | D/T/N/冲突 | 提示排除 | 微覆盖 | 平均k | State Fidelity | Agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f'| {pct(row["local_resolution_rate"])} | '
        f'{row["evidence_recall_at_k"]:.4f} | '
        f'{row["evidence_direct"]}/{row["evidence_transitive"]}/'
        f'{row["evidence_none"]}/{row["evidence_conflict"]} | '
        f'{row["evidence_excluded_prompt"]} | '
        f'{row["evidence_micro_hits"]}/{row["evidence_micro_total"]} | '
        f'{row["average_k"]:.4f} | '
        f'{row["state_pass"]}/{row["state_total"]} '
        f'({row["state_fidelity"]:.4f}) | NA（待人工） |',
        "", "## 逐钩子类别", "",
        "| Hook | 类别 | A_min/A_full | Reason codes |",
        "|---|---:|---:|---|",
    ]
    for hook_id in sorted(categories):
        hook = hooks[hook_id]
        attribution = hook["attribution"]
        reasons = ", ".join(hook["reason_codes"]) or "—"
        lines.append(
            f'| {hook_id} | {categories[hook_id]} | '
            f'{str(attribution["a_min"]).lower()}/'
            f'{str(attribution["a_full"]).lower()} | {reasons} |'
        )
    lines += [
        "", "## 口径说明", "",
        "- P（Partial Hook Adherence）要求至少一个目标特异历史事实具有可验证来源链并被正文使用。",
        "- Hook-aware Progress=(G+P)/10；token/可归因推进=总token proxy/(G+P)。",
        "- L仅表示局部提示、近期上下文或窗口内验证支持的合理处理，只在附录报告。",
        "- L不提高长期兑现满足度；Plot可独立较高，不能单独证明长期记忆。",
        "- 长程矛盾/万字按统一重评协议审查10条固定钩子轨迹；未提及或遗忘不计矛盾。",
        "- Agreement待人工标签后计算。", "",
    ]
    (ROOT / "results" / alias / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    rows, hooks = [], {}
    for method, alias in METHODS.items():
        row, categories = normalize(method, alias)
        rows.append(row)
        hooks[method] = categories
        write_method_report(alias, row, categories)
    (ROOT / "summary.json").write_text(
        json.dumps({"schema": "m5-model-review-summary", "rows": rows},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (ROOT / "hook_categories.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method"] + [f"H{x:02d}" for x in range(1, 11)])
        writer.writeheader()
        for method in METHODS:
            writer.writerow({"method": method, **hooks[method]})

    lines = [
        "## Material Passport", "", "- Mode: model review / descriptive validation",
        "- Primary review: six independent `gpt-5.6-sol` agents, `medium` reasoning",
        "- Contradiction re-audit: six anonymous `gpt-5.6-sol` agents, `medium` reasoning",
        "- Metric correction: evidence audit plus two matched-rubric `gpt-5.6-sol` re-reviewers",
        "- Verification Status: `ANALYZED`", "- Replicates: one completed novel per method", "",
        "# M5六方法模型盲评汇总", "", "## 正文指标", "",
        "| 方法 | Gold | Partial Hook Adherence | Hook-aware Progress | Plot | Long-range | 长程矛盾/万字 | token/可归因推进 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cost = "NA" if r["tokens_per_hook_aware_progress"] is None else f'{r["tokens_per_hook_aware_progress"]:,.0f}'
        long_range = f'{r["long_range_narrative_quality"]:.2f}'
        lines.append(f'| {r["method"]} | {pct(r["gold_rate"])} | '
                     f'{pct(r["partial_hook_adherence_rate"])} | '
                     f'{pct(r["hook_aware_progress_rate"])} | {r["plot_coherence"]:.2f} | '
                     f'{long_range} | '
                     f'{r["contradiction_density_per_10000"]:.4f} | {cost} |')
    lines += ["", "Long-range按统一四分量评审；Local只影响问题解决合理性，不提高兑现满足度。", ""]
    lines += ["Plot是独立叙事质量指标，方法间高低不直接证明长期记忆。", "",
              "## 附录指标", "",
              "| 方法 | Local | Memory Recall | D/T/N/冲突 | 提示排除 | 微覆盖 | 平均k | State Fidelity | Agreement |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'| {r["method"]} | {pct(r["local_resolution_rate"])} | '
                     f'{r["evidence_recall_at_k"]:.4f} | '
                     f'{r["evidence_direct"]}/{r["evidence_transitive"]}/'
                     f'{r["evidence_none"]}/{r["evidence_conflict"]} | '
                     f'{r["evidence_excluded_prompt"]} | '
                     f'{r["evidence_micro_hits"]}/{r["evidence_micro_total"]} | '
                     f'{r["average_k"]:.4f} | {r["state_pass"]}/{r["state_total"]} '
                     f'({r["state_fidelity"]:.4f}) | NA（待人工） |')
    lines += ["", "## 附录：逐钩子类别", "",
              "| 方法 | " + " | ".join(f"H{x:02d}" for x in range(1, 11)) + " |",
              "|---|" + "---:|" * 10]
    for method in METHODS:
        lines.append("| " + method + " | " + " | ".join(hooks[method][f"H{x:02d}"] for x in range(1, 11)) + " |")
    best_gold = max(row["gold_rate"] for row in rows)
    best_plot = max(row["plot_coherence"] for row in rows)
    best_recall = max(row["evidence_recall_at_k"] for row in rows)
    gold_names = "、".join(row["method"] for row in rows if row["gold_rate"] == best_gold)
    plot_names = "、".join(row["method"] for row in rows if row["plot_coherence"] == best_plot)
    recall_names = "、".join(row["method"] for row in rows if row["evidence_recall_at_k"] == best_recall)
    lines += ["", "## 描述性结论", "",
              f"- 可归因Gold最高：{gold_names}（{pct(best_gold)}）。",
              f"- Plot Coherence最高：{plot_names}（{best_plot:.2f}）。",
              f"- Memory-supported Recall最高：{recall_names}（{best_recall:.4f}）。",
              "- Partial Hook Adherence表示至少召回并使用一个目标特异历史事实，但没有达到完整Gold。",
              "- Hook-aware Progress=(G+P)/10，只表示可归因的钩子推进，不等同于完整解决率。",
              "- 只由当前提示、近期上下文或窗口验证形成的完整方案记L，并仅在附录报告。",
              "- Long-range已按v3专项复评；L不得提高payoff分项，终卷四条全局线与钩子兑现分开评分。",
              "- 长程矛盾密度来自统一重评：每部审查同一10条固定钩子轨迹、116个去重章节；不是全书穷尽计数。",
              "- 未提及、遗忘、合理状态变化与仅违反金标预期不计矛盾；旧32章缝计数已停止用于方法比较。",
              "- 这是单次运行、每书一名模型评审的描述性结果；Agreement须待人工标签后计算。", ""]
    (ROOT / "SUMMARY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
