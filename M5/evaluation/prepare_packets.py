from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

M5 = Path(__file__).resolve().parents[1]
EVAL = M5 / "evaluation"
ASSIGNMENTS = {
    "E01": "B3",
    "E02": "TaskGraph",
    "E03": "B0",
    "E04": "B4",
    "E05": "B1",
    "E06": "B2",
}
TRANSITIONS = [
    2, 11, 21, 40, 41, 51, 61, 80, 81, 91, 101, 120, 121, 131, 141, 160,
    161, 171, 181, 200, 201, 211, 221, 240, 241, 251, 261, 280,
    281, 291, 301, 320,
]
ANON_RE = re.compile(
    r"current_only|recency_window|running_summary|flat_vector_rag|"
    r"hierarchical_summary|taskgraph",
    re.IGNORECASE,
)
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ATTRIBUTION_KEYS = {
    "H01": ["铜盘身份、拳头大小圆形与七凹/三角定位结构"],
    "H02": ["04:17固定停时", "每分钟反向跳回七秒"],
    "H03": ["三短针、一长针、两短针", "跟蓝线走，不要数门"],
    "H04": ["A-17对污染样显紫、正常样保持绿色的标定规则"],
    "H05": ["三短两长", "北塔不亮、南门让潮"],
    "H06": ["S.X.-07手套身份", "五个指尖垫加掌心垫共六个触点"],
    "H07": ["琥珀色圆框", "Y形裂纹与偏振显青线"],
    "H08": ["九个结", "一一二三五八十三二十一与左右交替"],
    "H09": ["13.7兆赫静默载波", "二七二节奏"],
    "H10": ["苏弦惯用左手", "逆向起笔、短横、末笔向左收"],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_chapter_prompts() -> dict[int, dict]:
    prompts: dict[int, dict] = {}
    for path in sorted((M5 / "prompts").glob("chapters_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                prompts[int(item["chapter_id"])] = item
    if sorted(prompts) != list(range(1, 321)):
        raise RuntimeError("Expected exactly 320 chapter prompts")
    return prompts


def chapter_contract(item: dict) -> str:
    include = "\n".join(f"- {value}" for value in item.get("must_include", []))
    avoid = "\n".join(f"- {value}" for value in item.get("must_avoid", []))
    return (
        f"只撰写第{item['chapter_id']}章。\n"
        f"章节标题：{item['title']}\n"
        f"本章目标：{item['chapter_goal']}\n"
        f"必须包含：\n{include}\n"
        f"必须避免：\n{avoid}\n"
        f"目标长度：约{item['target_chars']}个汉字，允许区间2000—3000个汉字。"
    )


def split_writer_input(user: str, contract: str) -> tuple[str, str]:
    boundaries = [
        "\n\n本章开场承接材料",
        "\n\n此前已经建立",
        "\n\n其他当前有效事实",
    ]
    positions = [user.find(value) for value in boundaries if user.find(value) >= 0]
    if positions:
        split_at = min(positions)
        current_prompt = user[:split_at].strip()
        tail = user[split_at:].lstrip()
    else:
        start = user.find(contract)
        if start < 0:
            raise RuntimeError("Writer input does not contain a recognizable contract")
        current_prompt = user[start:start + len(contract)].strip()
        tail = user[start + len(contract):]
    end_positions = [
        tail.find(marker) for marker in (
            "叙事一致性规则：", "以下当前章要求具有最高优先级"
        ) if tail.find(marker) >= 0
    ]
    end = min(end_positions) if end_positions else -1
    if end < 0:
        raise RuntimeError("Writer input does not contain the consistency marker")
    history = tail[:end].strip()
    return current_prompt, history


def source_chapters(method: str, run: Path, record: dict) -> list[int]:
    selected = record.get("selected_chapters")
    if selected is None:
        selected = (record.get("memory_metadata") or {}).get("selected_chapters")
    if selected is not None:
        return sorted({int(value) for value in selected})
    if method != "TaskGraph":
        return []
    graph_ref = Path(str(record.get("prewrite_graph") or ""))
    graph_path = graph_ref if graph_ref.is_absolute() else run / graph_ref
    if not graph_path.is_file():
        return []
    graph = read_json(graph_path)
    selected_ids = set(record.get("selected_node_ids") or [])
    chapters = {
        int((node.get("metadata") or {}).get("chapter_id"))
        for node in graph.get("nodes", [])
        if node.get("node_id") in selected_ids
        and (node.get("metadata") or {}).get("chapter_id") is not None
    }
    local = record.get("local_continuity") or {}
    if local.get("source_chapter") is not None:
        chapters.add(int(local["source_chapter"]))
    return sorted(chapters)


def _marked_units(history: str, pattern: str) -> list[tuple[re.Match, str]]:
    matches = list(re.finditer(pattern, history, flags=re.MULTILINE))
    return [
        (match, history[
            match.start():matches[index + 1].start() if index + 1 < len(matches) else None
        ].strip())
        for index, match in enumerate(matches)
    ]


_AUDIT_SENTENCE_BREAK_RE = re.compile(r"[\r\n。！？；!?;]+")


def _normalize_audit_text(text: str) -> str:
    """Normalize Chinese prose without weakening factual substring checks."""
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    normalized = normalized.strip("\ufeff\ufffd").strip()
    return "".join(
        char.casefold() for char in normalized
        if not char.isspace()
        and not unicodedata.category(char).startswith(("P", "Z"))
    )


def _is_grounded_text(source: str, candidate: str) -> bool:
    """Accept exact normalized text or ordered sentence fragments.

    Token chunk boundaries can leave a replacement character at either edge,
    while graph source quotes can join adjacent source sentences.  Interior
    wording must still be present verbatim after Chinese punctuation/width
    normalization.
    """
    source_normalized = _normalize_audit_text(source)
    candidate_normalized = _normalize_audit_text(candidate)
    if len(candidate_normalized) >= 8 and candidate_normalized in source_normalized:
        return True
    fragments = [
        _normalize_audit_text(fragment)
        for fragment in _AUDIT_SENTENCE_BREAK_RE.split(str(candidate))
    ]
    fragments = [fragment for fragment in fragments if len(fragment) >= 8]
    if not fragments:
        return False
    position = 0
    for fragment in fragments:
        found = source_normalized.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
    return True


def _chapter_source(run: Path, chapter: int) -> str:
    path = run / "chapters" / f"chapter_{chapter:03d}.md"
    if not path.is_file():
        raise RuntimeError(f"Missing claimed source chapter {chapter}: {path}")
    return path.read_text(encoding="utf-8")


def _require_grounded(
    *, method: str, current_chapter: int, source_chapter: int,
    source: str, candidates: list[tuple[str, str]], identifier: str,
) -> None:
    for _, candidate in candidates:
        if candidate.strip() and _is_grounded_text(source, candidate):
            return
    labels = ", ".join(label for label, _ in candidates)
    raise RuntimeError(
        f"{method} chapter {current_chapter} {identifier} is not grounded in "
        f"claimed source chapter {source_chapter} ({labels})"
    )


def _taskgraph_source_verification(
    run: Path, chapter: object, metadata: dict, node: dict,
) -> tuple[bool, str]:
    if chapter is None:
        return False, "missing_metadata_chapter_id"
    source_chapter = int(chapter)
    path = run / "chapters" / f"chapter_{source_chapter:03d}.md"
    if not path.is_file():
        return False, "missing_source_chapter_file"
    source = path.read_text(encoding="utf-8")
    source_quote = str(metadata.get("source_quote") or "")
    if source_quote.strip() and _is_grounded_text(source, source_quote):
        return True, "verified_source_quote"
    node_value = str(node.get("value") or "")
    if node_value.strip() and _is_grounded_text(source, node_value):
        return True, "verified_node_value"
    return False, "source_text_not_found"


def split_history_units(
    method: str, run: Path, record: dict, history: str,
) -> list[dict]:
    if method == "B0" or not history.strip():
        return []
    units: list[dict] = []
    if method == "B1":
        selected = (record.get("memory_metadata") or {}).get("selected_chapters") or []
        prior_position = -1
        for chapter in selected:
            text = (run / "chapters" / f"chapter_{int(chapter):03d}.md").read_text(
                encoding="utf-8"
            ).strip()
            position = history.find(text)
            if position < 0 or position <= prior_position:
                raise RuntimeError(f"B1 history does not contain selected chapter {chapter}")
            prior_position = position
            units.append({
                "source_chapters": [int(chapter)], "text": text,
                "unit_kind": "raw_chapter", "provenance_cap": "D",
            })
    elif method == "B2":
        units.append({
            "source_chapters": source_chapters(method, run, record),
            "text": history.strip(),
            "unit_kind": "running_summary", "provenance_cap": "T",
        })
    elif method == "B3":
        marked = _marked_units(history, r"^\[历史第(\d+)章片段(\d+)\]\s*$")
        if not marked:
            raise RuntimeError("B3 history has no retriever chunk markers")
        observed = [(int(match.group(1)), int(match.group(2))) for match, _ in marked]
        expected = [
            (int(item["chapter_id"]), int(item["chunk_id"]))
            for item in (record.get("memory_metadata") or {}).get("ranked_candidates", [])
            if item.get("selected")
        ]
        if observed != expected:
            raise RuntimeError("B3 prompt chunks do not match selected retriever chunks")
        for match, text in marked:
            chapter = int(match.group(1))
            body = text.split("\n", 1)[1] if "\n" in text else ""
            _require_grounded(
                method=method, current_chapter=int(record["chapter_id"]),
                source_chapter=chapter, source=_chapter_source(run, chapter),
                candidates=[("retrieved chunk", body)],
                identifier=f"retrieved chunk {match.group(2)}",
            )
            units.append({
                "source_chapters": [chapter], "text": text,
                "unit_kind": "retrieved_chunk", "provenance_cap": "D",
            })
    elif method == "B4":
        marked = _marked_units(history, r"^\[第(\d+)(卷累计摘要|章摘要)\]\s*$")
        if not marked:
            raise RuntimeError("B4 history has no summary block markers")
        coverage = (record.get("memory_metadata") or {}).get("volume_coverage") or {}
        included = (record.get("memory_metadata") or {}).get("included_blocks") or []
        observed = [
            (int(match.group(1)), "volume" if match.group(2) == "卷累计摘要" else "chapter")
            for match, _ in marked
        ]
        expected = [(int(item["id"]), item["level"]) for item in included]
        if observed != expected:
            raise RuntimeError("B4 prompt blocks do not match included summaries")
        for match, text in marked:
            number = int(match.group(1))
            sources = (coverage.get(str(number)) or coverage.get(number) or [])
            if match.group(2) == "章摘要":
                sources = [number]
            units.append({
                "source_chapters": sorted({int(value) for value in sources}),
                "text": text,
                "unit_kind": (
                    "volume_summary" if match.group(2) == "卷累计摘要"
                    else "chapter_summary"
                ),
                "provenance_cap": "T",
            })
    elif method == "TaskGraph":
        marker = "其他当前有效事实："
        if marker not in history:
            raise RuntimeError("TaskGraph history has no fact-section marker")
        continuity, facts = history.split(marker, 1)
        continuity = continuity.strip()
        local = record.get("local_continuity") or {}
        if continuity:
            units.append({
                "source_chapters": [int(local["source_chapter"])],
                "text": continuity,
                "unit_kind": "local_continuity", "provenance_cap": "D",
            })
        graph_ref = Path(str(record.get("prewrite_graph") or ""))
        graph_path = graph_ref if graph_ref.is_absolute() else run / graph_ref
        graph = read_json(graph_path)
        selected_ids = set(record.get("selected_node_ids") or [])
        nodes = {
            node["node_id"]: node for node in graph.get("nodes", [])
            if node.get("node_id") in selected_ids
        }
        fact_blocks = [block.strip() for block in re.split(r"\n\s*\n", facts) if block.strip()]
        selected_order = list(record.get("selected_node_ids") or [])
        if len(fact_blocks) != len(selected_order):
            raise RuntimeError("TaskGraph prompt block count differs from selected nodes")
        for block, node_id in zip(fact_blocks, selected_order):
            node = nodes[node_id]
            if str(node.get("value") or "").strip() not in block:
                raise RuntimeError("TaskGraph prompt block order differs from selected nodes")
            chapter = (node.get("metadata") or {}).get("chapter_id")
            metadata = node.get("metadata") or {}
            source_verified, verification_reason = _taskgraph_source_verification(
                run, chapter, metadata, node,
            )
            units.append({
                "source_chapters": [] if chapter is None else [int(chapter)],
                "text": block,
                "unit_kind": "graph_projection", "provenance_cap": "T",
                "source_verified": source_verified,
                "source_verification_reason": verification_reason,
            })
    else:
        raise RuntimeError(f"Unsupported method for history splitting: {method}")
    return units


def predicate_rows(hook: dict, field: str, label: str) -> list[dict]:
    return [
        {"predicate_id": f"{hook['hook_id']}.{label}.{index:02d}", "text": text}
        for index, text in enumerate(hook.get(field, []), start=1)
    ]


def calls_for(record: dict) -> list[dict]:
    if record.get("actor_provenance"):
        calls: list[dict] = []
        for actor in record["actor_provenance"].values():
            calls.extend(actor.get("invocations") or [])
        calls.extend(record.get("external_api_calls") or [])
        return calls
    calls = record.get("api_calls") or []
    return calls if isinstance(calls, list) else [calls]


def objective_metrics(method: str, run: Path) -> dict:
    totals = defaultdict(float)
    purposes = defaultdict(lambda: {"calls": 0, "tokens": 0})
    writer_attempts = 0
    chars: list[int] = []
    for path in sorted((run / "records").glob("chapter_*.json")):
        record = read_json(path)
        chars.append(int(record.get("han_chars") or 0))
        for call in calls_for(record):
            input_tokens = int(call.get("input_tokens") or 0)
            output_tokens = int(call.get("output_tokens") or 0)
            call_tokens = int(call.get("total_tokens") or input_tokens + output_tokens)
            purpose = str(call.get("purpose") or "unknown")
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["total_tokens"] += call_tokens
            totals["elapsed_ms"] += float(call.get("elapsed_ms") or 0)
            purposes[purpose]["calls"] += 1
            purposes[purpose]["tokens"] += call_tokens
            if purpose in {"chapter", "writer"}:
                writer_attempts += 1
    return {
        "chapters": len(chars),
        "total_han_chars": sum(chars),
        "min_han_chars": min(chars),
        "max_han_chars": max(chars),
        "mechanical_completion_count_2000_3500": sum(2000 <= x <= 3500 for x in chars),
        "writer_band_count_2000_2500": sum(2000 <= x <= 2500 for x in chars),
        "total_token_proxy": int(totals["total_tokens"]),
        "input_token_proxy": int(totals["input_tokens"]),
        "output_token_proxy": int(totals["output_tokens"]),
        "elapsed_ms_sum": round(totals["elapsed_ms"], 3),
        "writer_attempts": writer_attempts,
        "writer_retries": max(0, writer_attempts - len(chars)),
        "purpose_breakdown": dict(sorted(purposes.items())),
        "token_note": "offline token proxy from persisted records; not a provider bill",
    }


def main() -> None:
    hooks = [json.loads(x) for x in (M5 / "hooks_gold.jsonl").read_text(encoding="utf-8").splitlines() if x]
    schedule = [json.loads(x) for x in (M5 / "hook_schedule.jsonl").read_text(encoding="utf-8").splitlines() if x]
    prompts = load_chapter_prompts()
    schedule_by_id = {x["hook_id"]: x for x in schedule}
    protocol = M5 / "prompts" / "model_review_prompt.md"
    key = {"schema": "m5-model-review-assignments", "assignments": ASSIGNMENTS}
    (EVAL / "controller").mkdir(parents=True, exist_ok=True)
    write_json(EVAL / "controller" / "assignment_key.json", key)
    for alias, method in ASSIGNMENTS.items():
        packet = EVAL / "packets" / alias
        packet.mkdir(parents=True, exist_ok=True)
        run = M5 / "methods" / method / "run"
        metrics = objective_metrics(method, run)
        metrics["protocol_sha256"] = sha256(protocol)
        write_json(packet / "objective_metrics.json", metrics)

        manifest_hooks = []
        for hook in hooks:
            sched = schedule_by_id[hook["hook_id"]]
            evidence_chapters = sorted(set(
                hook.get("intermediate_chapters", [])
                + [sched["trigger_chapter"]]
                + list(range(sched["evaluation_window"][0],
                             sched["evaluation_window"][1] + 1))
                + sched["scoring_chapters"]
            ))
            manifest_hooks.append({
                "hook_id": hook["hook_id"],
                "plant_chapter": hook["plant_chapter"],
                "intermediate_chapters": hook.get("intermediate_chapters", []),
                "trigger_chapter": sched["trigger_chapter"],
                "evaluation_window": sched["evaluation_window"],
                "scoring_chapters": sched["scoring_chapters"],
                "evidence_chapters": evidence_chapters,
                "seed_predicates": predicate_rows(hook, "seed_predicates", "seed"),
                "state_requirements": predicate_rows(hook, "state_requirements", "state"),
                "payoff_predicates": predicate_rows(
                    hook, "required_payoff_predicates", "payoff"
                ),
                "forbidden_outcomes": predicate_rows(
                    hook, "forbidden_outcomes", "forbidden"
                ),
                "attribution_min_keys": [
                    {"key_id": f"{hook['hook_id']}.attr.{index:02d}", "text": text}
                    for index, text in enumerate(
                        ATTRIBUTION_KEYS[hook["hook_id"]], start=1
                    )
                ],
            })
        review_manifest = {
            "schema_version": "m5-review-manifest",
            "protocol_sha256": sha256(protocol),
            "hooks": manifest_hooks,
        }
        write_json(packet / "review_manifest.json", review_manifest)
        metrics["review_manifest_sha256"] = sha256(packet / "review_manifest.json")
        write_json(packet / "objective_metrics.json", metrics)

        sample_chars = 0
        with (packet / "continuity_samples.jsonl").open("w", encoding="utf-8") as fh:
            for chapter in TRANSITIONS:
                previous = (run / "chapters" / f"chapter_{chapter - 1:03d}.md").read_text(encoding="utf-8")
                current = (run / "chapters" / f"chapter_{chapter:03d}.md").read_text(encoding="utf-8")
                prev_excerpt, curr_excerpt = previous[-1000:], current[:1000]
                count = len(HAN_RE.findall(prev_excerpt + curr_excerpt))
                sample_chars += count
                fh.write(json.dumps({"target_chapter": chapter, "previous_end": prev_excerpt,
                                     "current_start": curr_excerpt, "han_chars": count},
                                    ensure_ascii=False) + "\n")
        metrics["continuity_sample_han_chars"] = sample_chars
        write_json(packet / "objective_metrics.json", metrics)

        with (packet / "evidence_inputs.jsonl").open("w", encoding="utf-8") as fh:
            for hook in manifest_hooks:
                for chapter in hook["evidence_chapters"]:
                    record = read_json(run / "records" / f"chapter_{chapter:03d}.json")
                    writer_ref = str(record.get("writer_input") or "")
                    writer_path = Path(writer_ref)
                    if not writer_path.is_absolute():
                        writer_path = run / writer_path
                    raw_input = (writer_path.read_text(encoding="utf-8")
                                 if writer_path.is_file() else writer_ref)
                    try:
                        parsed_input = json.loads(raw_input)
                    except json.JSONDecodeError:
                        parsed_input = None
                    writer_visible = (str(parsed_input.get("user") or "")
                                      if isinstance(parsed_input, dict) else raw_input)
                    try:
                        current_prompt, history = split_writer_input(
                            writer_visible, chapter_contract(prompts[chapter])
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"Cannot split writer input for {method} chapter {chapter}: {exc}"
                        ) from exc
                    if method == "B0":
                        history = ""
                    anonymized_input = ANON_RE.sub("[ANON]", writer_visible)
                    raw_units = split_history_units(method, run, record, history)
                    unit_rows = []
                    for unit_index, unit in enumerate(raw_units, start=1):
                        unit_text = ANON_RE.sub("[ANON]", unit["text"])
                        if any(int(source) >= chapter for source in unit["source_chapters"]):
                            raise RuntimeError(
                                f"Future or same-chapter history in {method} chapter {chapter}"
                            )
                        unit_row = {
                            "unit_id": (
                                f"{hook['hook_id']}.C{chapter:03d}.history."
                                f"{unit_index:02d}"
                            ),
                            "source_chapters": unit["source_chapters"],
                            "unit_kind": unit["unit_kind"],
                            "provenance_cap": unit["provenance_cap"],
                            "text": unit_text,
                            "sha256": text_sha256(unit_text),
                        }
                        if "source_verified" in unit:
                            unit_row["source_verified"] = unit["source_verified"]
                            unit_row["source_verification_reason"] = unit[
                                "source_verification_reason"
                            ]
                        unit_rows.append(unit_row)
                    fh.write(json.dumps({
                        "hook_id": hook["hook_id"],
                        "chapter_id": chapter,
                        "is_scoring_chapter": chapter in hook["scoring_chapters"],
                        "current_prompt": current_prompt,
                        "history_units": unit_rows,
                        "history_unit_count": len(unit_rows),
                        "writer_input": anonymized_input,
                    }, ensure_ascii=False) + "\n")

        index = []
        for hook in hooks:
            sched = schedule_by_id[hook["hook_id"]]
            end = sched["evaluation_window"][1]
            chapters = sorted(set([hook["plant_chapter"], sched["trigger_chapter"]]
                                  + hook["intermediate_chapters"]
                                  + list(range(sched["evaluation_window"][0], end + 1))
                                  + list(range(end + 1, min(320, end + 3) + 1))))
            index.append({"hook_id": hook["hook_id"], "chapters": chapters,
                          "files": [f"novel/chapter_{x:03d}.md" for x in chapters]})
        write_json(packet / "hook_chapter_index.json", index)
        (packet / "README.md").write_text(
            f"# Anonymous review packet {alias}\n\n"
            "Follow `../../../prompts/model_review_prompt.md`. Do not inspect link metadata or "
            "`../../controller/`. Write only to `../../results/" + alias + "/`.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
