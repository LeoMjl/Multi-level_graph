from __future__ import annotations

import json
import ast
import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterator

from mlg.config import RAW_DIR
from mlg.schemas import Episode, Message


STABLETOOLBENCH_SOLVABLE_SUBSETS = {
    "G1_instruction": 163,
    "G1_category": 153,
    "G1_tool": 158,
    "G2_instruction": 106,
    "G2_category": 124,
    "G3_instruction": 61,
}


def load_json_or_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return rows
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                rows.extend(dict(row) for row in csv.DictReader(f))
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict):
                            rows.append(item)
                    if limit and len(rows) >= limit:
                        break
        elif path.suffix.lower() == ".json":
            streamed_rows, is_array = _load_json_array_stream(path, limit=limit)
            if is_array:
                rows.extend(streamed_rows)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    rows.extend(item for item in payload if isinstance(item, dict))
                elif isinstance(payload, dict):
                    for value in payload.values():
                        if isinstance(value, list):
                            rows.extend(item for item in value if isinstance(item, dict))
                    if not rows:
                        rows.append(payload)
        elif path.suffix.lower() == ".parquet":
            try:
                import pandas as pd

                frame = pd.read_parquet(path)
                rows.extend(frame.to_dict(orient="records"))
            except Exception:
                return []
        else:
            text = path.read_text(encoding="utf-8")
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    rows.extend(item for item in payload if isinstance(item, dict))
                elif isinstance(payload, dict):
                    rows.append(payload)
            except json.JSONDecodeError:
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
    except Exception:
        return []
    for row in rows:
        row.setdefault("_source_file", path.name)
        row.setdefault("_source_stem", path.stem)
    return rows[:limit] if limit else rows


def _load_json_array_stream(path: Path, *, limit: int = 0) -> tuple[list[dict[str, Any]], bool]:
    """Read a top-level JSON array without materializing a multi-GB file at once."""
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return [], False
            if not char.isspace():
                first = char
                break
        if first != "[":
            return [], False

        buffer = ""
        eof = False
        while True:
            if not eof:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            while buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return rows, True
            if not buffer and eof:
                raise ValueError(f"Unterminated JSON array: {path}")
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                continue
            buffer = buffer[end:]
            if isinstance(item, dict):
                rows.append(item)
                if limit and len(rows) >= limit:
                    return rows, True


def _iter_json_array_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one object at a time from a top-level JSON array."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return
            if not char.isspace():
                if char != "[":
                    raise ValueError(f"Expected a top-level JSON array: {path}")
                break
        buffer = ""
        eof = False
        while True:
            if not eof:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            while buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            if not buffer and eof:
                raise ValueError(f"Unterminated JSON array: {path}")
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                continue
            buffer = buffer[end:]
            if isinstance(item, dict):
                item.setdefault("_source_file", path.name)
                item.setdefault("_source_stem", path.stem)
                yield item


def find_records(repo_name: str, patterns: tuple[str, ...], limit: int = 0) -> list[dict[str, Any]]:
    root = RAW_DIR / "repos" / repo_name
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for pattern in patterns:
        for path in root.rglob(pattern):
            rows.extend(load_json_or_jsonl(path, limit=max(0, limit - len(rows)) if limit else 0))
            if limit and len(rows) >= limit:
                return rows[:limit]
    return rows


def adapt_longmemeval(limit: int = 0) -> list[Episode]:
    records = find_records("longmemeval", ("*.jsonl", "*.json", "longmemeval_*"), limit=limit * 3 if limit else 0)
    episodes: list[Episode] = []
    for idx, row in enumerate(records):
        question = first_text(row, "question", "query", "input")
        answer = first_answer(row)
        history = coerce_longmemeval_history(row) or coerce_history(row, "history", "haystack", "context", "conversation")
        if not question or not answer or not history:
            continue
        gold_evidence = extract_longmemeval_evidence(row)
        episodes.append(
            Episode(
                episode_id=str(row.get("id") or row.get("sample_id") or f"longmemeval_{idx}"),
                dataset="LongMemEval",
                split=str(row.get("split", "test")),
                history=history,
                query=question,
                answers=[answer],
                gold_evidence=gold_evidence,
                metadata={"source_row_keys": sorted(row.keys())},
            )
        )
        if limit and len(episodes) >= limit:
            break
    return episodes


def adapt_stabletoolbench(limit: int = 0) -> list[Episode]:
    root = RAW_DIR / "repos" / "stabletoolbench"
    records: list[dict[str, Any]] = []
    for rel in ("test_sft/id_low.json", "test_cot/id_low.json", "reference/id_low.jsonl"):
        records.extend(load_json_or_jsonl(root / rel, limit=max(0, (limit * 3) - len(records)) if limit else 0))
        if limit and len(records) >= limit * 3:
            break
    if not records:
        records = find_records("stabletoolbench", ("*.jsonl", "*.json"), limit=limit * 3 if limit else 0)
    episodes: list[Episode] = []
    for idx, row in enumerate(records):
        instruction = first_text(row, "instruction", "query", "question", "input")
        if not instruction and isinstance(row.get("messages"), list):
            instruction = " ".join(
                str(msg.get("content", ""))
                for msg in row["messages"]
                if isinstance(msg, dict) and msg.get("role") in {"user", "human"}
            )
        if not instruction:
            continue
        parsed = parse_stabletoolbench_instruction(instruction)
        tools = row.get("tools") or row.get("available_tools") or row.get("apis") or row.get("functions") or []
        if not tools and parsed.get("api_doc"):
            api_doc = parsed["api_doc"]
            tools = [
                {
                    "name": api_doc.get("tool_name", ""),
                    "api_name": api_doc.get("api_name", ""),
                    "category": api_doc.get("tool_category", ""),
                    "description": api_doc.get("api_description", ""),
                }
            ]
        expected = stabletoolbench_expected_answer(row)
        query = instruction
        if parsed.get("request"):
            query = (
                "Produce the API response for this StableToolBench request.\n"
                f"Request: {json.dumps(parsed['request'], ensure_ascii=False)}"
            )
        tool_context = f"Available tools: {json.dumps(tools, ensure_ascii=False)}"
        if parsed.get("api_doc"):
            tool_context += f"\nAPI doc: {json.dumps(parsed['api_doc'], ensure_ascii=False)}"
        episodes.append(
            Episode(
                episode_id=str(row.get("id") or row.get("query_id") or f"stabletoolbench_{idx}"),
                dataset="StableToolBench",
                split=str(row.get("split", "test")),
                history=[Message(role="system", content=tool_context, turn_index=0)],
                query=query,
                answers=[str(expected)] if expected else [],
                metadata={
                    "tools": tools,
                    "request": parsed.get("request", {}),
                    "solvable": row.get("solvable", True),
                },
            )
        )
        if limit and len(episodes) >= limit:
            break
    return episodes


def stabletoolbench_expected_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "final_answer", "expected_answer", "output"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    return ""


def parse_stabletoolbench_instruction(instruction: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    api_marker = "API doc:"
    request_marker = "Request:"
    if api_marker not in instruction:
        return result
    api_start = instruction.find(api_marker) + len(api_marker)
    request_start = instruction.find(request_marker, api_start)
    api_text = instruction[api_start:request_start if request_start >= 0 else len(instruction)].strip()
    api_doc = parse_pythonish_dict(api_text)
    if api_doc:
        result["api_doc"] = api_doc
    if request_start >= 0:
        request_text = instruction[request_start + len(request_marker) :].strip()
        request = parse_pythonish_dict(request_text)
        if request:
            result["request"] = request
    return result


def parse_pythonish_dict(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    snippet = text[start : end + 1]
    try:
        value = ast.literal_eval(snippet)
    except (SyntaxError, ValueError):
        try:
            value = json.loads(snippet)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def adapt_longbench(limit: int = 0) -> list[Episode]:
    records = find_longbench_records(limit=limit * 2 if limit else 0)
    episodes: list[Episode] = []
    for idx, row in enumerate(records):
        context = first_text(row, "context", "article", "passage")
        query = first_text(row, "input", "question", "query")
        answers = row.get("answers") or row.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        if not context or not query:
            continue
        subset = str(row.get("dataset") or row.get("subset") or row.get("_source_stem") or "")
        episodes.append(
            Episode(
                episode_id=str(row.get("id") or row.get("_id") or f"longbench_{idx}"),
                dataset="LongBench",
                split=str(row.get("split", "test")),
                history=[Message(role="user", content=context, turn_index=1)],
                query=query,
                answers=[str(item) for item in answers],
                metadata={
                    "task_type": str(row.get("task_type") or infer_longbench_task_type(subset)),
                    "subset": subset,
                    "source_file": row.get("_source_file", ""),
                    "context_length": len(context),
                },
            )
        )
        if limit and len(episodes) >= limit:
            break
    return episodes


LONGMEMEVAL_V2_CATEGORY_ORDER = [
    "static-environment",
    "dynamic-environment",
    "procedure",
    "errors-gotchas",
    "static-environment-abs",
    "dynamic-environment-abs",
    "procedure-abs",
]


def adapt_m1_longmemeval_v2(limit: int = 0) -> list[Episode]:
    data_root = _longmemeval_v2_data_root()
    if data_root is None:
        raise FileNotFoundError(
            "LongMemEval-V2 official questions.jsonl, trajectories.jsonl, and haystacks are unavailable under "
            f"{RAW_DIR / 'repos' / 'longmemeval_v2'}. Fetch V2 explicitly; the M1 adapter will not "
            "relabel classic LongMemEval as V2."
        )

    tier = os.getenv("MLG_LONGMEMEVAL_V2_TIER", "small").strip().lower()
    if tier not in {"small", "medium"}:
        raise ValueError("MLG_LONGMEMEVAL_V2_TIER must be 'small' or 'medium'")
    released_questions = load_json_or_jsonl(data_root / "questions.jsonl")
    if not released_questions:
        raise ValueError(f"LongMemEval-V2 questions are empty: {data_root / 'questions.jsonl'}")
    # Generic project methods currently send text-only OpenAI-compatible requests.
    # Keep their M1 comparison on the released text-question subset; the upstream
    # launcher remains the canonical path for the 29 screenshot questions.
    questions = [question for question in released_questions if not question.get("image")]
    excluded_image_questions = len(released_questions) - len(questions)
    if not limit and tier == "small" and len(questions) != 422:
        raise ValueError(
            "LongMemEval-V2-Text small-tier protocol requires exactly 422 text questions; "
            f"found {len(questions)}"
        )
    questions = _stratified_longmemeval_v2_questions(questions, limit=limit)
    haystack_path = data_root / "haystacks" / f"lme_v2_{tier}.json"
    haystacks = json.loads(haystack_path.read_text(encoding="utf-8"))
    if not isinstance(haystacks, dict):
        raise ValueError(f"LongMemEval-V2 haystack must be a question-id mapping: {haystack_path}")

    selected_ids: set[str] = set()
    for question in questions:
        question_id = str(question.get("id", ""))
        trajectory_ids = haystacks.get(question_id)
        if not isinstance(trajectory_ids, list) or not trajectory_ids:
            raise ValueError(f"LongMemEval-V2 question {question_id} has no {tier} haystack")
        selected_ids.update(str(item) for item in trajectory_ids)
    trajectories = _load_longmemeval_v2_trajectories(data_root / "trajectories.jsonl", selected_ids)

    histories: dict[tuple[str, ...], list[Message]] = {}
    context_lengths: dict[tuple[str, ...], int] = {}
    episodes: list[Episode] = []
    for question in questions:
        question_id = str(question["id"])
        trajectory_ids = tuple(str(item) for item in haystacks[question_id])
        history = histories.get(trajectory_ids)
        if history is None:
            history = []
            turn_index = 1
            for trajectory_id in trajectory_ids:
                trajectory = trajectories[trajectory_id]
                for message in _longmemeval_v2_trajectory_messages(trajectory):
                    message.turn_index = turn_index
                    history.append(message)
                    turn_index += 1
            histories[trajectory_ids] = history
            context_lengths[trajectory_ids] = sum(len(message.content) for message in history)
        image_value = question.get("image")
        image_path = str((data_root / image_value).resolve()) if isinstance(image_value, str) and image_value else ""
        episodes.append(Episode(
            episode_id=question_id,
            dataset="LongMemEval-V2-Text",
            split="test",
            history=history,
            query=str(question["question"]),
            answers=[str(question.get("answer", ""))],
            metadata={
                "memory_category": str(question.get("question_type", "")),
                "task_type": "long_agent_memory",
                "context_length": context_lengths[trajectory_ids],
                "source_file": "questions.jsonl",
                "source_subset": f"lme_v2_{tier}_text_questions",
                "original_sample_id": question_id,
                "oracle_subset": False,
                "official_benchmark": True,
                "benchmark_variant": "official_text_question_accessibility_tree",
                "haystack_tier": tier,
                "haystack_trajectory_ids": list(trajectory_ids),
                "trajectory_count": len(trajectory_ids),
                "domain": str(question.get("domain", "")),
                "environment": str(question.get("environment", "")),
                "eval_function": str(question.get("eval_function", "")),
                "question_image": image_path,
                "multimodal_question": False,
                "excluded_multimodal_question_count": excluded_image_questions,
                "sampling_strategy": "all" if not limit else "deterministic_stratified_round_robin",
            },
        ))
    return episodes


def _longmemeval_v2_data_root() -> Path | None:
    base = RAW_DIR / "repos" / "longmemeval_v2"
    for candidate in (base / "dataset", base):
        required = (
            candidate / "questions.jsonl",
            candidate / "trajectories.jsonl",
            candidate / "haystacks" / "lme_v2_small.json",
            candidate / "haystacks" / "lme_v2_medium.json",
        )
        if all(path.is_file() for path in required):
            return candidate
    return None


def _stratified_longmemeval_v2_questions(
    questions: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not limit or limit >= len(questions):
        return questions
    order = {category: idx for idx, category in enumerate(LONGMEMEVAL_V2_CATEGORY_ORDER)}
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for question in questions:
        key = (str(question.get("question_type", "")), str(question.get("domain", "")))
        strata.setdefault(key, []).append(question)
    for rows in strata.values():
        rows.sort(key=lambda row: hashlib.sha256(str(row.get("id", "")).encode("utf-8")).hexdigest())
    keys = sorted(strata, key=lambda key: (order.get(key[0], len(order)), key[1]))
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if depth < len(strata[key]):
                selected.append(strata[key][depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def _load_longmemeval_v2_trajectories(
    path: Path,
    selected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            trajectory_id = str(row.get("id", ""))
            if trajectory_id in selected_ids:
                if trajectory_id in trajectories:
                    raise ValueError(f"Duplicate LongMemEval-V2 trajectory id {trajectory_id} at line {line_no}")
                trajectories[trajectory_id] = row
                if len(trajectories) == len(selected_ids):
                    break
    missing = selected_ids - set(trajectories)
    if missing:
        raise ValueError(f"LongMemEval-V2 haystack references missing trajectories: {sorted(missing)[:10]}")
    return trajectories


def _longmemeval_v2_trajectory_messages(trajectory: dict[str, Any]) -> list[Message]:
    trajectory_id = str(trajectory.get("id", ""))
    domain = str(trajectory.get("domain", ""))
    messages = [Message(
        role="user",
        content=(
            f"[trajectory_id={trajectory_id}]\n"
            f"Domain: {domain}\n"
            f"Environment: {trajectory.get('environment', '')}\n"
            f"Goal: {trajectory.get('goal', '')}\n"
            f"Outcome: {trajectory.get('outcome', '')}\n"
            f"Start URL: {trajectory.get('start_url', '')}"
        ),
        metadata={"trajectory_id": trajectory_id, "domain": domain, "record_type": "trajectory_header"},
    )]
    states = trajectory.get("states")
    if not isinstance(states, list):
        return messages
    for ordinal, state in enumerate(states):
        if not isinstance(state, dict):
            continue
        state_index = state.get("state_index", ordinal)
        action = state.get("action")
        action_text = json.dumps(action, ensure_ascii=False, default=str) if action is not None else "NONE"
        messages.append(Message(
            role="tool",
            content=(
                f"[trajectory_id={trajectory_id} state_index={state_index} step={state.get('step', '')}]\n"
                f"URL: {state.get('url', '')}\n"
                f"Action: {action_text}\n"
                f"Thought: {state.get('thought', '')}\n"
                f"Accessibility tree:\n{state.get('accessibility_tree', '')}\n"
                f"Screenshot: {state.get('screenshot', '')}"
            ),
            metadata={
                "trajectory_id": trajectory_id,
                "state_index": state_index,
                "domain": domain,
                "record_type": "trajectory_state",
            },
        ))
    return messages


def adapt_m2_locomo_event_summarization(limit: int = 0) -> list[Episode]:
    """Adapt the official LoCoMo speaker-level event summarization benchmark."""
    source = RAW_DIR / "repos" / "locomo" / "repository" / "data" / "locomo10.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Official LoCoMo data is missing: {source}. Run `mlg data fetch --dataset locomo`."
        )
    records = load_json_or_jsonl(source)
    episodes: list[Episode] = []
    for row in records:
        conversation = row.get("conversation")
        event_summary = row.get("event_summary")
        if not isinstance(conversation, dict) or not isinstance(event_summary, dict):
            raise ValueError("Official LoCoMo row is missing conversation/event_summary")
        sample_id = str(row.get("sample_id", "")).strip()
        sessions = _locomo_session_numbers(conversation)
        history = _official_locomo_history(conversation, sessions)
        if not sample_id or not sessions or not history:
            raise ValueError("Official LoCoMo row has an invalid sample_id or empty conversation")
        speakers = [
            str(conversation.get("speaker_a", "")).strip(),
            str(conversation.get("speaker_b", "")).strip(),
        ]
        dates = [
            str(conversation.get(f"session_{number}_date_time", "")).strip()
            for number in sessions
        ]
        timeframe = f"{dates[0]} through {dates[-1]}"
        for speaker in speakers:
            gold_events = _official_locomo_events(event_summary, sessions, speaker)
            if not speaker or not gold_events:
                raise ValueError(f"Official LoCoMo target has no events: {sample_id}/{speaker}")
            query = (
                f"Summarize the significant life events of {speaker} from {timeframe}. "
                "List only events supported by the conversation, in chronological order. "
                "Include dates or time references and preserve explicit causal connections when present."
            )
            gold_summary = "\n".join(
                f"- {event['date']}: {event['fact']}" for event in gold_events
            )
            episodes.append(
                Episode(
                    episode_id=f"m2_{sample_id}_{_slug(speaker)}",
                    dataset="LoCoMo Event Summarization (official)",
                    split="test",
                    history=list(history),
                    query=query,
                    answers=[gold_summary],
                    metadata={
                        "task_type": "locomo_event_summarization",
                        "benchmark_type": "official_locomo_event_summarization",
                        "parent_episode_id": sample_id,
                        "target_speaker": speaker,
                        "timeframe": timeframe,
                        "session_count": len(sessions),
                        "gold_events": gold_events,
                        "source_file": "locomo10.json",
                    },
                )
            )
            if limit and len(episodes) >= limit:
                return episodes
    return episodes


def _locomo_session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(key))
        if match and isinstance(value, list):
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def _official_locomo_history(conversation: dict[str, Any], sessions: list[int]) -> list[Message]:
    speaker_a = str(conversation.get("speaker_a", "")).strip()
    history: list[Message] = []
    for session_number in sessions:
        session_id = f"session_{session_number}"
        date = str(conversation.get(f"{session_id}_date_time", "")).strip()
        for turn in conversation.get(session_id, []):
            if not isinstance(turn, dict):
                continue
            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", "")).strip()
            caption = str(turn.get("blip_caption", "")).strip()
            if caption:
                text = f"{text}\n[Image caption: {caption}]"
            history.append(
                Message(
                    role="user" if speaker == speaker_a else "assistant",
                    content=f"[{session_id} | {date} | {speaker}]\n{text}",
                    turn_index=len(history) + 1,
                    metadata={
                        "session_id": session_id,
                        "session_date": date,
                        "speaker": speaker,
                    },
                )
            )
    return history


def _official_locomo_events(
    event_summary: dict[str, Any],
    sessions: list[int],
    speaker: str,
) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for session_number in sessions:
        item = event_summary.get(f"events_session_{session_number}", {})
        if not isinstance(item, dict):
            continue
        date = str(item.get("date", "")).strip()
        facts = item.get(speaker, [])
        if not isinstance(facts, list):
            continue
        events.extend(
            {"date": date, "fact": str(fact).strip()}
            for fact in facts
            if str(fact).strip()
        )
    return events


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def adapt_m3_toolbench(limit: int = 0) -> list[Episode]:
    root = (
        RAW_DIR
        / "repos"
        / "stabletoolbench"
        / "repository"
        / "solvable_queries"
    )
    if not root.is_dir():
        raise FileNotFoundError(
            "Official StableToolBench checkout is missing. Run "
            "`mlg data fetch --dataset stabletoolbench`."
        )
    by_subset: dict[str, list[Episode]] = {}
    seen_ids: set[str] = set()
    for subset, expected_count in STABLETOOLBENCH_SOLVABLE_SUBSETS.items():
        query_path = root / "test_instruction" / f"{subset}.json"
        id_path = root / "test_query_ids" / f"{subset}.json"
        rows = json.loads(query_path.read_text(encoding="utf-8"))
        official_ids = json.loads(id_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(
                f"Official StableToolBench {subset} must contain "
                f"{expected_count} solvable queries"
            )
        if not isinstance(official_ids, dict):
            raise ValueError(f"Invalid StableToolBench ID manifest: {id_path}")
        episodes: list[Episode] = []
        for row in rows:
            query_id = str(row.get("query_id", "")).strip()
            query = str(row.get("query", "")).strip()
            tools = row.get("api_list")
            if not query_id or query_id not in official_ids:
                raise ValueError(f"Unregistered query ID in {query_path}: {query_id}")
            if query_id in seen_ids:
                raise ValueError(f"Duplicate StableToolBench query ID: {query_id}")
            if not query or not isinstance(tools, list) or not tools:
                raise ValueError(f"Incomplete StableToolBench query: {query_id}")
            seen_ids.add(query_id)
            episodes.append(
                Episode(
                    episode_id=f"stabletoolbench:{subset}:{query_id}",
                    dataset="StableToolBench",
                    split=subset,
                    history=[
                        Message(
                            role="system",
                            content=(
                                "Official available API definitions:\n"
                                + json.dumps(tools, ensure_ascii=False)
                            ),
                            turn_index=0,
                        )
                    ],
                    query=query,
                    metadata={
                        "query_id": query_id,
                        "tools": tools,
                        "solvable": True,
                        "source_subset": subset,
                        "source_file": str(query_path),
                        "task_type": "stabletoolbench_tool_use",
                    },
                )
            )
        by_subset[subset] = episodes
    expected_total = sum(STABLETOOLBENCH_SOLVABLE_SUBSETS.values())
    if len(seen_ids) != expected_total:
        raise ValueError(
            f"Official StableToolBench must contain {expected_total} unique queries"
        )
    if not limit or limit >= len(seen_ids):
        return [
            episode
            for subset in STABLETOOLBENCH_SOLVABLE_SUBSETS
            for episode in by_subset[subset]
        ]
    return _stabletoolbench_stratified_sample(by_subset, limit)


def _stabletoolbench_stratified_sample(
    by_subset: dict[str, list[Episode]],
    limit: int,
) -> list[Episode]:
    """Select a deterministic proportional pilot over all official subsets."""
    total = sum(len(rows) for rows in by_subset.values())
    target = max(0, min(limit, total))
    quotas = {
        subset: target * len(rows) / total
        for subset, rows in by_subset.items()
    }
    counts = {subset: int(quota) for subset, quota in quotas.items()}
    remainder = target - sum(counts.values())
    ranked = sorted(
        by_subset,
        key=lambda subset: (
            -(quotas[subset] - counts[subset]),
            list(STABLETOOLBENCH_SOLVABLE_SUBSETS).index(subset),
        ),
    )
    for subset in ranked[:remainder]:
        counts[subset] += 1
    return [
        episode
        for subset in STABLETOOLBENCH_SOLVABLE_SUBSETS
        for episode in by_subset[subset][: counts[subset]]
    ]


def adapt_a1_longmemeval_cleaned(limit: int = 0) -> list[Episode]:
    episodes = adapt_longmemeval(limit=limit)
    for episode in episodes:
        episode.episode_id = f"a1_{episode.episode_id}"
        episode.dataset = "LongMemEval-cleaned"
        episode.metadata["task_type"] = "traditional_long_memory_qa"
        episode.gold_dependencies = [{"source": item, "target": episode.query} for item in episode.gold_evidence[:3]]
    return episodes


def adapt_a2_hotpotqa(limit: int = 0) -> list[Episode]:
    records = find_records("hotpotqa", ("*.jsonl", "*.json", "*.parquet"), limit=limit * 3 if limit else 0)
    if not records:
        records = [row for row in find_longbench_records(limit=limit * 4 if limit else 0) if "hotpot" in str(row.get("_source_stem", "")).lower()]
    return _adapt_evidence_qa(records, "HotpotQA", "multihop_dependency", limit)


def adapt_a3_qasper(limit: int = 0) -> list[Episode]:
    records = find_records("qasper", ("*.jsonl", "*.json", "*.parquet"), limit=limit * 3 if limit else 0)
    if not records:
        records = [row for row in find_longbench_records(limit=limit * 4 if limit else 0) if "qasper" in str(row.get("_source_stem", "")).lower()]
    episodes: list[Episode] = []
    for idx, row in enumerate(_flatten_qasper_records(records)):
        context = _qasper_context(row)
        qas = row.get("qas") if isinstance(row.get("qas"), list) else []
        if not qas:
            fallback = _adapt_evidence_qa([row], "QASPER", "scientific_evidence_grounding", 1)
            episodes.extend(fallback)
            continue
        for qa in qas:
            if not isinstance(qa, dict):
                continue
            question = first_text(qa, "question", "query")
            answers = qa.get("answers") if isinstance(qa.get("answers"), list) else []
            answer_text, evidence = _qasper_answer_and_evidence(answers)
            if not question or not answer_text or not context:
                continue
            episodes.append(
                Episode(
                    episode_id=str(qa.get("question_id") or row.get("id") or row.get("paper_id") or f"qasper_{idx}_{len(episodes)}"),
                    dataset="QASPER",
                    split=str(row.get("split", "validation")),
                    history=[Message(role="user", content=context, turn_index=1)],
                    query=question,
                    answers=[answer_text],
                    gold_evidence=evidence,
                    gold_dependencies=[{"source": item, "target": question} for item in evidence[:3]],
                    metadata={"task_type": "scientific_evidence_grounding", "source_file": row.get("_source_file", "")},
                )
            )
            if limit and len(episodes) >= limit:
                return episodes
    return episodes


def adapt_a4_longbench(limit: int = 0) -> list[Episode]:
    episodes = adapt_longbench(limit=limit)
    for episode in episodes:
        episode.episode_id = f"a4_{episode.episode_id}"
        episode.metadata["task_type"] = episode.metadata.get("task_type", "long_context_generalization")
    return episodes


def adapt_a5_needle_source(limit: int = 0) -> list[Episode]:
    return adapt_longbench(limit=limit)


def find_longbench_records(limit: int = 0) -> list[dict[str, Any]]:
    root = RAW_DIR / "repos" / "longbench"
    if not root.exists():
        return []
    paths = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))
    if not paths:
        return []
    if not limit:
        records: list[dict[str, Any]] = []
        for path in paths:
            records.extend(load_json_or_jsonl(path))
        return records
    per_file = max(1, limit)
    buckets = [load_json_or_jsonl(path, limit=per_file) for path in paths]
    records = []
    while len(records) < limit and any(buckets):
        for bucket in buckets:
            if bucket:
                records.append(bucket.pop(0))
                if len(records) >= limit:
                    break
    return records


def infer_longbench_task_type(subset: str) -> str:
    normalized = subset.lower()
    if normalized in {"gov_report", "qmsum", "multi_news", "vcsum", "samsum"}:
        return "summarization"
    if normalized in {"lcc", "repobench-p"}:
        return "code"
    if "passage_retrieval" in normalized:
        return "retrieval"
    if "passage_count" in normalized:
        return "counting"
    if normalized in {"trec", "lsht"}:
        return "classification"
    return "qa"


def _source_split(row: dict[str, Any], *, default: str) -> str:
    explicit = str(row.get("split", "")).strip().lower()
    if explicit:
        return explicit
    source = " ".join(
        str(row.get(key, "")).lower()
        for key in ("_source_file", "_source_stem", "source_file")
    )
    for split in ("train", "validation", "valid", "dev", "test"):
        if split in source:
            return "validation" if split in {"valid", "dev"} else split
    return default


def _adapt_evidence_qa(records: list[dict[str, Any]], dataset: str, task_type: str, limit: int = 0) -> list[Episode]:
    episodes: list[Episode] = []
    for idx, row in enumerate(records):
        context = first_text(row, "context", "article", "passage", "paragraphs")
        if not context and isinstance(row.get("context"), list):
            context = "\n".join(str(item) for item in row["context"])
        query = first_text(row, "question", "query", "input")
        answers = row.get("answers") or row.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        evidence = _extract_supporting_facts(row) or extract_longmemeval_evidence(row)
        if not evidence and context:
            evidence = [context[:500]]
        if not context or not query:
            continue
        episodes.append(
            Episode(
                episode_id=str(row.get("id") or row.get("_id") or f"{dataset.lower()}_{idx}"),
                dataset=dataset,
                split=str(row.get("split", "validation")),
                history=[Message(role="user", content=context, turn_index=1)],
                query=query,
                answers=[str(item) for item in answers],
                gold_evidence=evidence,
                gold_dependencies=[{"source": item, "target": query} for item in evidence[:3]],
                metadata={"task_type": task_type, "source_file": row.get("_source_file", "")},
            )
        )
        if limit and len(episodes) >= limit:
            break
    return episodes


def _flatten_qasper_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in records:
        if "qas" in row:
            flattened.append(row)
            continue
        for key, value in row.items():
            if isinstance(value, dict) and "qas" in value:
                item = dict(value)
                item.setdefault("paper_id", key)
                item.setdefault("_source_file", row.get("_source_file", ""))
                flattened.append(item)
    return flattened


def _qasper_context(row: dict[str, Any]) -> str:
    parts = [first_text(row, "title"), first_text(row, "abstract")]
    full_text = row.get("full_text")
    if isinstance(full_text, list):
        for section in full_text:
            if not isinstance(section, dict):
                continue
            section_name = first_text(section, "section_name")
            paragraphs = section.get("paragraphs") or []
            if section_name:
                parts.append(section_name)
            if isinstance(paragraphs, list):
                parts.extend(str(paragraph) for paragraph in paragraphs[:8])
    return "\n".join(part for part in parts if part)[:60000]


def _qasper_answer_and_evidence(answers: list[Any]) -> tuple[str, list[str]]:
    for item in answers:
        if not isinstance(item, dict):
            continue
        answer = item.get("answer", item)
        if not isinstance(answer, dict):
            continue
        if answer.get("unanswerable") is True:
            continue
        spans = answer.get("extractive_spans") or []
        free_form = str(answer.get("free_form_answer") or "").strip()
        yes_no = answer.get("yes_no")
        evidence = [str(e) for e in (answer.get("evidence") or []) if str(e).strip()]
        if spans:
            return "; ".join(str(span) for span in spans), evidence
        if free_form:
            return free_form, evidence
        if yes_no is not None:
            return str(yes_no), evidence
    return "", []


def _extract_supporting_facts(row: dict[str, Any]) -> list[str]:
    value = row.get("supporting_facts") or row.get("evidence") or row.get("evidences")
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, (list, tuple)):
                result.append(" ".join(str(x) for x in item))
            elif isinstance(item, dict):
                result.append(first_text(item, "text", "sentence", "paragraph", "title"))
        return dedupe_keep_order(result)
    return []


def first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return " ".join(str(item) for item in value[:5])
        if isinstance(value, dict) and value:
            return json.dumps(value, ensure_ascii=False, default=str)
        if value is not None and value.__class__.__name__ == "ndarray":
            return " ".join(str(item) for item in value[:5])
    return ""


def first_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "answers", "target", "output", "final_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return str(value[0])
    return ""


def coerce_history(row: dict[str, Any], *keys: str) -> list[Message]:
    for key in keys:
        value = row.get(key)
        messages = value_to_messages(value)
        if messages:
            return messages
    text = first_text(row, "context", "input_context")
    return [Message(role="user", content=text, turn_index=1)] if text else []


def value_to_messages(value: Any) -> list[Message]:
    messages: list[Message] = []
    if isinstance(value, str) and value.strip():
        return [Message(role="user", content=value.strip(), turn_index=1)]
    if isinstance(value, list):
        for idx, item in enumerate(value, start=1):
            if isinstance(item, str):
                content = item
                role = "user"
            elif isinstance(item, dict):
                content = first_text(item, "content", "text", "utterance", "message")
                role = str(item.get("role") or item.get("speaker") or "user")
            else:
                content = str(item)
                role = "user"
            if content:
                messages.append(Message(role=role, content=content, turn_index=idx))
    return messages


def coerce_longmemeval_history(row: dict[str, Any]) -> list[Message]:
    sessions = row.get("haystack_sessions")
    if not isinstance(sessions, list):
        return []
    messages: list[Message] = []
    turn_idx = 1
    for session in sessions:
        if not isinstance(session, list):
            continue
        for msg in session:
            if not isinstance(msg, dict):
                continue
            content = first_text(msg, "content", "text", "utterance")
            if not content:
                continue
            messages.append(Message(role=str(msg.get("role", "user")), content=content, turn_index=turn_idx))
            turn_idx += 1
    return messages


def extract_longmemeval_evidence(row: dict[str, Any]) -> list[str]:
    explicit = first_text(row, "evidence", "evidence_text", "supporting_facts")
    if explicit:
        return [explicit]
    sessions = row.get("haystack_sessions")
    if not isinstance(sessions, list):
        return []
    answer = first_answer(row).strip()
    answer_lower = answer.lower()
    answer_session_ids = {str(item) for item in row.get("answer_session_ids", [])}
    haystack_session_ids = [str(item) for item in row.get("haystack_session_ids", [])]
    marked_evidence: list[str] = []
    session_evidence: list[str] = []
    for idx, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        session_id = haystack_session_ids[idx] if idx < len(haystack_session_ids) else ""
        session_messages = [
            first_text(msg, "content", "text", "utterance")
            for msg in session
            if isinstance(msg, dict)
        ]
        for msg, content in zip((item for item in session if isinstance(item, dict)), session_messages):
            if content and msg.get("has_answer") is True:
                marked_evidence.append(content)
        if session_id in answer_session_ids:
            answer_hits = [content for content in session_messages if answer_lower and answer_lower in content.lower()]
            session_evidence.extend(answer_hits or [content for content in session_messages if content])
    evidence = marked_evidence or session_evidence
    if not evidence and answer_lower:
        for session in sessions:
            if not isinstance(session, list):
                continue
            for msg in session:
                if not isinstance(msg, dict):
                    continue
                content = first_text(msg, "content", "text", "utterance")
                if answer_lower in content.lower():
                    evidence.append(content)
    return dedupe_keep_order(evidence)


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
