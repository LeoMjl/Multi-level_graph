from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    model: str
    elapsed_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    response_id: str = ""
    purpose: str = "chapter"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextBackend(Protocol):
    model: str

    def generate(self, system: str, user: str, *, purpose: str = "chapter") -> GenerationResult: ...


class OpenAIResponsesBackend:
    """Official Responses API backend for the M5 writer and memory updater."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "medium",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "",
        max_output_tokens: int = 12000,
        timeout_seconds: float = 600.0,
    ) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"M5 writer requires {api_key_env}; no fallback provider is used")
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds, "max_retries": 5}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def generate(self, system: str, user: str, *, purpose: str = "chapter") -> GenerationResult:
        started = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=self.max_output_tokens,
        )
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        total_tokens = _usage_value(usage, "total_tokens")
        return GenerationResult(
            text=str(getattr(response, "output_text", "") or "").strip(),
            model=self.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            response_id=str(getattr(response, "id", "") or ""),
            purpose=purpose,
        )


class FakeTextBackend:
    """Deterministic offline backend for harness validation, never for results."""

    model = "fake-m5-writer"

    def generate(self, system: str, user: str, *, purpose: str = "chapter") -> GenerationResult:
        started = time.perf_counter()
        if purpose == "dependency_judge":
            payload = json.loads(user)
            text = json.dumps({"targets": [
                {
                    "target_id": target["target_id"],
                    "dependencies": [
                        {
                            "source_id": candidate["node_id"],
                            "dependency_type": (
                                "state_continuity"
                                if candidate.get("stable_key_match")
                                else "contextual_relevance"
                            ),
                            "confidence": 1.0,
                            "priority": index + 1,
                            "reason": "离线测试保留全部候选",
                        }
                        for index, candidate in enumerate(target.get("candidates", []))
                    ],
                }
                for target in payload.get("targets", [])
            ]}, ensure_ascii=False)
        elif purpose == "memory":
            text = json.dumps({
                "summary": "本章推进了当前目标，人物状态与物品变化均以正文为准。",
                "facts": ["章节已完成", "后续必须保持时间、地点、人物知识与物品状态一致"],
                "entities": ["林澈"],
                "open_threads": ["继续核验当前异常的原因"],
            }, ensure_ascii=False)
        else:
            marker = _extract_marker(user, "章节标题：") or "测试章节"
            body = (
                "潮声贴着混凝土外壁缓慢移动。林澈依照当前记录复核仪表、位置与时间，"
                "没有把尚未证实的猜测当作结论。他逐项完成现场检查，并把变化写入本地记录。"
            )
            text = f"# {marker}\n\n" + (body * 34)
        return GenerationResult(
            text=text,
            model=self.model,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=len(system + user) // 2,
            output_tokens=len(text) // 2,
            total_tokens=len(system + user + text) // 2,
            purpose=purpose,
        )


def _usage_value(usage: Any, key: str) -> int | None:
    value = getattr(usage, key, None) if usage is not None else None
    return int(value) if value is not None else None


def _extract_marker(text: str, marker: str) -> str:
    for line in text.splitlines():
        if line.startswith(marker):
            return line[len(marker):].strip()
    return ""
