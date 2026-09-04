from __future__ import annotations

import re
from typing import Any

from mlg.graph import EdgeType, NodeLevel, NodeStatus, TaskGraph
from mlg.methods.base import pack_text_items, trim_text


LOCOMO_GRAPH_KIND = "locomo_event_graph_v3"
LOCOMO_BUILD_OUTPUT_TOKENS = 16_384
LOCOMO_BUILD_CONTEXT_TOKENS = 80_000
LOCOMO_QUERY_CONTEXT_TOKENS = 16_384


def locomo_graph_build_context(payload: dict, *, max_tokens: int) -> tuple[str, dict]:
    """Pack only the query-independent conversation into the graph builder."""
    items = []
    for ordinal, message in enumerate(payload.get("history", []), start=1):
        metadata = message.get("metadata", {})
        turn = int(message.get("turn_index") or ordinal)
        session_id = str(metadata.get("session_id", "session_unknown"))
        date = str(metadata.get("session_date", ""))
        speaker = str(metadata.get("speaker", ""))
        body = _message_body(str(message.get("content", "")))
        items.append(f"[turn={turn} session={session_id} date={date} speaker={speaker}]\n{body}")
    return pack_text_items(items, max_tokens=max_tokens)


def locomo_graph_build_prompts(context: str) -> tuple[str, str]:
    system = (
        "Build a reusable, query-independent event memory from the supplied LoCoMo conversation. "
        "Extract significant real-world or life events for every named participant, not for one "
        "requested target. Omit greetings, routine chat, repetitions, and unsupported inference. "
        "Preserve explicit dates. Add a dependency only when the conversation explicitly states "
        "that one event caused, enabled, or motivated another. Evidence turn numbers must refer "
        "to the supplied conversation."
    )
    user = (
        f"Conversation:\n{context}\n\n"
        'Return JSON only as {"events":[{"id":"E1","speaker":"exact participant name",'
        '"date":"explicit date or empty","event":"concise standalone event",'
        '"evidence_turns":[1],"depends_on":["E0"]}]}. '
        "List events chronologically, use at most two evidence turns per event, and keep each event "
        "to one concise sentence. depends_on must contain only earlier event IDs."
    )
    return system, user


def normalize_locomo_graph_events(raw: dict, payload: dict) -> list[dict[str, Any]]:
    candidates = raw.get("events", []) if isinstance(raw, dict) else []
    if not isinstance(candidates, list):
        return []
    messages = list(payload.get("history", []))
    by_turn = {
        int(message.get("turn_index") or ordinal): message
        for ordinal, message in enumerate(messages, start=1)
    }
    canonical_speakers: dict[str, str] = {}
    for message in messages:
        speaker = str(message.get("metadata", {}).get("speaker", "")).strip()
        if speaker:
            canonical_speakers.setdefault(speaker.casefold(), speaker)
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    valid_source_ids: set[str] = set()
    for ordinal, item in enumerate(candidates[:160], start=1):
        if not isinstance(item, dict):
            continue
        speaker = canonical_speakers.get(str(item.get("speaker", "")).strip().casefold(), "")
        event = str(item.get("event", item.get("fact", item.get("content", "")))).strip()
        if not speaker or not event:
            continue
        evidence_turns = _valid_turns(item.get("evidence_turns", []), by_turn)[:2]
        date = str(item.get("date", "")).strip()
        if not date and evidence_turns:
            date = str(by_turn[evidence_turns[0]].get("metadata", {}).get("session_date", ""))
        dedupe_key = (speaker.casefold(), date.casefold(), re.sub(r"\W+", " ", event.casefold()).strip())
        if dedupe_key in seen:
            continue
        source_id = str(item.get("id", f"E{ordinal}")).strip() or f"E{ordinal}"
        depends_on = item.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        output.append(
            {
                "source_id": source_id,
                "speaker": speaker,
                "date": date,
                "event": trim_text(event, max_chars=600),
                "evidence_turns": evidence_turns,
                "depends_on": [str(value).strip() for value in depends_on if str(value).strip()],
            }
        )
        valid_source_ids.add(source_id)
        seen.add(dedupe_key)
    for event in output:
        event["depends_on"] = [value for value in event["depends_on"] if value in valid_source_ids]
    return output


def deterministic_locomo_graph_events(payload: dict) -> list[dict[str, Any]]:
    """Small deterministic substitute used only by fake-LLM smoke tests."""
    events = []
    for ordinal, message in enumerate(payload.get("history", []), start=1):
        metadata = message.get("metadata", {})
        body = _message_body(str(message.get("content", ""))).strip()
        speaker = str(metadata.get("speaker", "")).strip()
        if speaker and body:
            turn = int(message.get("turn_index") or ordinal)
            events.append(
                {
                    "source_id": f"E{len(events) + 1}",
                    "speaker": speaker,
                    "date": str(metadata.get("session_date", "")),
                    "event": trim_text(body, max_chars=600),
                    "evidence_turns": [turn],
                    "depends_on": [],
                }
            )
    return events


def build_locomo_event_graph(payload: dict, events: list[dict[str, Any]]) -> TaskGraph:
    """Persist compact events and the complete raw evidence without query data."""
    graph = TaskGraph()
    task_id = graph.add_node(
        NodeLevel.L1,
        content="Official LoCoMo reusable event memory",
        turn_index=0,
        path="T1",
        status=NodeStatus.ACTIVE,
        metadata={
            "graph_kind": LOCOMO_GRAPH_KIND,
            "source_message_count": len(payload.get("history", [])),
            "event_count": len(events),
        },
    )
    evidence_by_turn: dict[int, str] = {}
    for ordinal, message in enumerate(payload.get("history", []), start=1):
        metadata = message.get("metadata", {})
        turn = int(message.get("turn_index") or ordinal)
        evidence_id = graph.add_node(
            NodeLevel.L4,
            content=f"Raw evidence turn {turn}",
            turn_index=turn,
            path=f"T1.Evidence.Turn{turn}",
            status=NodeStatus.DONE,
            value=str(message.get("content", "")),
            sub_type="RawDialogueEvidence",
            metadata={
                "session_id": str(metadata.get("session_id", "session_unknown")),
                "session_date": str(metadata.get("session_date", "")),
                "speaker": str(metadata.get("speaker", "")),
                "source_turn": turn,
            },
        )
        graph.add_edge(task_id, evidence_id, EdgeType.INCLUSION)
        evidence_by_turn[turn] = evidence_id

    speakers: list[str] = []
    for event in events:
        speaker = str(event.get("speaker", "")).strip()
        if speaker and speaker.casefold() not in {item.casefold() for item in speakers}:
            speakers.append(speaker)
    timeline_by_speaker: dict[str, str] = {}
    for speaker_index, speaker in enumerate(speakers, start=1):
        timeline_id = graph.add_node(
            NodeLevel.L2,
            content=f"{speaker} significant-event timeline",
            turn_index=0,
            path=f"T1.Person{speaker_index}",
            status=NodeStatus.DONE,
            value=speaker,
            sub_type="PersonEventTimeline",
            metadata={"speaker": speaker},
        )
        graph.add_edge(task_id, timeline_id, EdgeType.INCLUSION)
        timeline_by_speaker[speaker.casefold()] = timeline_id

    node_by_source_id: dict[str, str] = {}
    previous_by_speaker: dict[str, str] = {}
    event_ordinal_by_speaker: dict[str, int] = {}
    pending_dependencies: list[tuple[str, list[str]]] = []
    for event in events:
        speaker = str(event["speaker"])
        speaker_key = speaker.casefold()
        timeline_id = timeline_by_speaker[speaker_key]
        event_ordinal_by_speaker[speaker_key] = event_ordinal_by_speaker.get(speaker_key, 0) + 1
        event_ordinal = event_ordinal_by_speaker[speaker_key]
        evidence_turns = [int(value) for value in event.get("evidence_turns", [])]
        turn = min(evidence_turns) if evidence_turns else event_ordinal
        event_id = graph.add_node(
            NodeLevel.L3,
            content=f"{speaker} event {event_ordinal}",
            turn_index=turn,
            path=f"{graph.nodes[timeline_id].path}.Event{event_ordinal}",
            status=NodeStatus.DONE,
            value=str(event["event"]),
            sub_type="SignificantEvent",
            metadata={
                "speaker": speaker,
                "date": str(event.get("date", "")),
                "evidence_turns": evidence_turns,
                "source_event_id": str(event.get("source_id", "")),
            },
        )
        graph.add_edge(timeline_id, event_id, EdgeType.INCLUSION)
        previous = previous_by_speaker.get(speaker_key)
        if previous:
            graph.add_edge(previous, event_id, EdgeType.MAINLINE, {"reason": "chronology"})
        previous_by_speaker[speaker_key] = event_id
        source_id = str(event.get("source_id", ""))
        if source_id:
            node_by_source_id[source_id] = event_id
        pending_dependencies.append((event_id, list(event.get("depends_on", []))))
        for evidence_turn in evidence_turns:
            evidence_id = evidence_by_turn.get(evidence_turn)
            if evidence_id:
                graph.add_edge(event_id, evidence_id, EdgeType.INCLUSION)

    for event_id, source_dependencies in pending_dependencies:
        for source_dependency in source_dependencies:
            dependency_id = node_by_source_id.get(source_dependency)
            if dependency_id and dependency_id != event_id:
                graph.add_edge(
                    dependency_id,
                    event_id,
                    EdgeType.DEPENDENCY,
                    {"reason": "explicit_causal_relation"},
                )
    return graph


def render_locomo_target_context(
    graph: TaskGraph,
    target_speaker: str,
    *,
    max_tokens: int = LOCOMO_QUERY_CONTEXT_TOKENS,
) -> tuple[str, dict]:
    """Project only one person's event nodes plus short grounded excerpts."""
    target_key = target_speaker.strip().casefold()
    events = sorted(
        (
            node
            for node in graph.nodes.values()
            if node.level == NodeLevel.L3
            and str(node.metadata.get("speaker", "")).casefold() == target_key
        ),
        key=lambda node: (node.turn_index, node.node_id),
    )
    display_id = {node.node_id: f"E{index}" for index, node in enumerate(events, start=1)}
    items: list[str] = []
    evidence_count = 0
    for event in events:
        dependency_nodes = graph.dependencies(event.node_id)
        dependencies = [display_id[node.node_id] for node in dependency_nodes if node.node_id in display_id]
        date = str(event.metadata.get("date", "")).strip() or "date not stated"
        relation = f" depends_on={','.join(dependencies)}" if dependencies else ""
        lines = [f"[{display_id[event.node_id]} | {date}{relation}] {event.value or event.content}"]
        for dependency in dependency_nodes:
            if dependency.node_id in display_id:
                continue
            dependency_date = str(dependency.metadata.get("date", "")).strip() or "date not stated"
            dependency_speaker = str(dependency.metadata.get("speaker", "")).strip()
            lines.append(
                "  causal_parent: "
                f"[{dependency_date} | {dependency_speaker}] {dependency.value or dependency.content}"
            )
        evidence_nodes = sorted(
            graph.children(event.node_id, level=NodeLevel.L4),
            key=lambda node: (node.turn_index, node.node_id),
        )[:2]
        for evidence in evidence_nodes:
            evidence_count += 1
            body = _message_body(evidence.value or evidence.content)
            lines.append(f"  evidence(T{evidence.turn_index}): {trim_text(body, max_chars=320)}")
        items.append("\n".join(lines))
    context, packing = pack_text_items(items, max_tokens=max_tokens)
    packing.update(
        {
            "available_events": len(events),
            "packed_events": min(len(events), int(packing.get("packed_items", len(events)))),
            "available_evidence_excerpts": evidence_count,
        }
    )
    return context, packing


def _valid_turns(raw_turns: object, by_turn: dict[int, dict]) -> list[int]:
    if not isinstance(raw_turns, list):
        return []
    output: list[int] = []
    for value in raw_turns:
        try:
            turn = int(value)
        except (TypeError, ValueError):
            continue
        if turn in by_turn and turn not in output:
            output.append(turn)
    return output


def _message_body(content: str) -> str:
    lines = content.splitlines()
    if lines and re.match(r"^\[session_[^\]]+\]$", lines[0].strip(), flags=re.IGNORECASE):
        return "\n".join(lines[1:]).strip()
    return content.strip()
