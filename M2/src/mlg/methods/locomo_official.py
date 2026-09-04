from __future__ import annotations

from collections import OrderedDict

from mlg.methods.base import (
    Method,
    formatted_history_items,
    pack_text_items,
    parse_prediction_json,
    token_encoder,
)


BASE_CONTEXT_TOKENS = 4_096


def locomo_event_summary_contract() -> str:
    return (
        'Return JSON only with this schema: {"answer":["one chronological significant '
        'event per array item for the requested speaker"], "evidence":["brief supporting '
        'conversation excerpts"]}. Do not include routine conversation details or unsupported facts.'
    )


def parse_locomo_event_summary(
    raw: dict,
) -> tuple[str, list[str], str, list[dict], dict]:
    """Normalize the LoCoMo JSON event array into evaluator-ready bullet text."""
    normalized = dict(raw)
    answer = raw.get("answer", raw.get("final_answer", ""))
    if isinstance(answer, list):
        events = [_locomo_event_text(item) for item in answer]
        normalized["answer"] = "\n".join(f"- {event}" for event in events if event)
    return parse_prediction_json(normalized)


def _locomo_event_text(item: object) -> str:
    if isinstance(item, dict):
        date = str(item.get("date", "")).strip()
        fact = str(
            item.get("fact", item.get("event", item.get("content", item.get("text", ""))))
        ).strip()
        return f"{date}: {fact}".strip(": ")
    return str(item).strip()


class _OfficialLoCoMoSummaryMethod(Method):
    context_policy = ""

    def _one_shot(
        self,
        payload: dict,
        context: str,
        packing: dict,
    ) -> tuple[str, list[str], str, list[dict], dict]:
        if self.can_call_llm():
            system = (
                "You are evaluating the official LoCoMo event summarization task. "
                "Use only the supplied dialogue and image captions. Select significant life events, "
                "retain temporal and causal connections, and follow the requested speaker and timeframe."
            )
            user = (
                f"Task:\n{payload['query']}\n\nDialogue:\n{context}\n\n"
                + locomo_event_summary_contract()
            )
            with self.measure_phase("reader_generation_time_ms"):
                raw = self.chat_json(system, user)
            answer, evidence, _, _, metadata = parse_locomo_event_summary(raw)
            metadata.update({"context_policy": self.context_policy, "context_packing": packing})
            if not answer:
                metadata.update(
                    {
                        "status": "failed",
                        "failure_stage": "reader_generation",
                        "failure_reason": str(raw.get("_error", "reader returned an empty summary")),
                    }
                )
            return answer, evidence, "", [], metadata
        if not self.fake_llm:
            return "", [], "", [], {
                "context_policy": self.context_policy,
                "context_packing": packing,
                "status": "failed",
                "failure_stage": "reader_generation",
                "failure_reason": f"LLM API is not configured ({self.runtime.api_key_env})",
            }
        target = str(payload.get("metadata", {}).get("target_speaker", "")).lower()
        lines = [
            item.split("\n", 1)[-1].strip()
            for item in formatted_history_items(payload)
            if target and target in item.lower()
        ][:12]
        return "\n".join(f"- {line}" for line in lines), lines[:3], "", [], {
            "context_policy": f"fake_{self.context_policy}",
            "context_packing": packing,
        }


class OfficialBaseContextMethod(_OfficialLoCoMoSummaryMethod):
    """Official Base setup: omit earlier dialogue outside a constrained window."""

    name = "official_base"
    context_policy = "official_base_earlier_dialogues_omitted_4k"

    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        context, packing = _tail_session_context(payload, BASE_CONTEXT_TOKENS)
        return self._one_shot(payload, context, packing)


class OfficialLongContextMethod(_OfficialLoCoMoSummaryMethod):
    """Official Long-context setup: expose as much chronological dialogue as possible."""

    name = "official_long_context"
    context_policy = "official_long_context_maximum_dialogue"

    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        budget = max(4_096, self.runtime.context_window_tokens - self.runtime.max_tokens - 2_048)
        context, packing = pack_text_items(formatted_history_items(payload), max_tokens=budget)
        return self._one_shot(payload, context, packing)


class OfficialIncrementalSummarizationMethod(_OfficialLoCoMoSummaryMethod):
    """Official iterative setup from the LoCoMo paper's session-summary prompt."""

    name = "official_incremental"
    context_policy = "official_incremental_session_summarization"

    def _predict_from_payload(self, payload: dict) -> tuple[str, list[str], str, list[dict], dict]:
        sessions: OrderedDict[str, list[dict]] = OrderedDict()
        for message in payload.get("history", []):
            session_id = str(message.get("metadata", {}).get("session_id", "unknown"))
            sessions.setdefault(session_id, []).append(message)
        if not self.can_call_llm():
            if self.fake_llm:
                context, packing = pack_text_items(
                    formatted_history_items(payload),
                    max_tokens=BASE_CONTEXT_TOKENS,
                )
                return self._one_shot(payload, context, packing)
            return self._one_shot(payload, "", {"total_sessions": len(sessions)})

        summary = ""
        evidence: list[str] = []
        for session_id, messages in sessions.items():
            date = str(messages[0].get("metadata", {}).get("session_date", ""))
            dialogue = "\n\n".join(
                f"[{item.get('role', '')}] {item.get('content', '')}" for item in messages
            )
            system = (
                "Apply the official LoCoMo incremental summarization protocol. "
                "Keep only significant events for the requested speaker, with time references and causal links."
            )
            user = (
                f"Task: {payload['query']}\n"
                f"In previous interactions: {summary or 'none'}\n"
                f"The current session is {session_id} at {date}:\n{dialogue}\n\n"
                "Update the event summary so far. " + locomo_event_summary_contract()
            )
            with self.measure_phase("reader_generation_time_ms"):
                raw = self.chat_json(system, user)
            updated, session_evidence, _, _, _ = parse_locomo_event_summary(raw)
            if not updated:
                return "", [], "", [], {
                    "context_policy": self.context_policy,
                    "completed_sessions": len(evidence),
                    "total_sessions": len(sessions),
                    "status": "failed",
                    "failure_stage": "reader_generation",
                    "failure_reason": str(raw.get("_error", "incremental summary was empty")),
                }
            summary = updated
            evidence.extend(session_evidence)
        return summary, evidence, "", [], {
            "context_policy": self.context_policy,
            "completed_sessions": len(sessions),
            "total_sessions": len(sessions),
        }


def _tail_session_context(payload: dict, max_tokens: int) -> tuple[str, dict]:
    sessions: OrderedDict[str, list[str]] = OrderedDict()
    for message in payload.get("history", []):
        session_id = str(message.get("metadata", {}).get("session_id", "unknown"))
        rendered = (
            f"[turn={message.get('turn_index', '')} role={message.get('role', 'user')}]\n"
            f"{message.get('content', '')}"
        )
        sessions.setdefault(session_id, []).append(rendered)
    blocks = ["\n\n".join(items) for items in sessions.values()]
    encoder = token_encoder()
    separator = encoder.encode("\n\n")
    selected: list[list[int]] = []
    used = 0
    partial_latest = False
    for block in reversed(blocks):
        encoded = encoder.encode(block)
        required = len(encoded) + (len(separator) if selected else 0)
        if used + required <= max_tokens:
            selected.append(encoded)
            used += required
            continue
        if not selected:
            selected.append(encoded[-max_tokens:])
            used = max_tokens
            partial_latest = True
        break
    selected.reverse()
    output: list[int] = []
    for encoded in selected:
        if output:
            output.extend(separator)
        output.extend(encoded)
    return encoder.decode(output), {
        "total_sessions": len(blocks),
        "packed_sessions": len(selected),
        "packed_tokens_estimate": len(output),
        "truncated": len(selected) < len(blocks) or partial_latest,
        "partial_latest_session": partial_latest,
        "omission": "earlier_dialogue_sessions",
        "tokenizer": "tiktoken:o200k_base_estimate",
    }
