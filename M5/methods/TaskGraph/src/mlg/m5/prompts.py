from __future__ import annotations

from mlg.m5.continuity import LocalContinuity
from mlg.m5.dataset import ChapterPrompt, M5Dataset
from mlg.m5.length_policy import WRITER_REQUESTED_HAN_CHARS
from mlg.m5.memory import MemoryContext


def build_writer_prompts(
    dataset: M5Dataset,
    chapter: ChapterPrompt,
    memory: MemoryContext,
    *,
    local_continuity: LocalContinuity | None = None,
    repeat_current_contract: bool = True,
    retry_feedback: str = "",
) -> tuple[str, str]:
    requirements = "\n".join(f"- {item}" for item in dataset.global_task.get("global_requirements", []))
    system = (
        "你是长篇小说正文作者。只能依据故事设定、当前章要求和提供的既有事实写作；"
        "不得补造人物尚未获得的信息，也不得输出内部结构、工作过程或审阅说明。\n\n"
        f"小说：{dataset.global_task.get('novel_title', '')}\n"
        f"叙事视角：{dataset.global_task.get('narrative_person', '')}\n"
        f"风格：{dataset.global_task.get('style', '')}\n"
        f"全局要求：\n{requirements}\n\n"
        f"故事设定：\n{dataset.story_bible}"
    )
    contract = _chapter_contract(chapter)
    local = local_continuity or LocalContinuity()
    previous = local.text.strip() or "（本章为开篇，没有上一章正文。）"
    history = memory.text.strip() or "（没有额外的更早事实；不得自行补造。）"
    repeat = (
        "以下当前章要求具有最高优先级，写作前再次核对：\n"
        f"{contract}\n\n"
        if repeat_current_contract
        else ""
    )
    retry = f"\n\n上一次输出未通过机械校验，请修正：{retry_feedback}" if retry_feedback else ""
    user = (
        f"{contract}\n\n"
        "本章开场承接材料（必须直接延续其时间、地点、人物状态与未完成动作）：\n"
        f"{previous}\n\n"
        f"其他当前有效事实：\n{history}\n\n"
        "叙事一致性规则：人物可以回忆、比对或据此安排核验，但必须遵守其知识边界。"
        "观察、推断和确认要明确区分：只有证据足够时才能把推断写成结论；"
        "若本章要求中的因果措辞超过当前证据，只能表现为人物的暂定判断，并保留可检验的替代解释。\n\n"
        f"{repeat}"
        "输出格式：第一行仅写章节标题，其后直接写连续正文；不得输出提纲、解释、摘要、字数统计或任何正文外附加信息。"
        f"{retry}"
    )
    return system, user


def packet_payload(
    chapter: ChapterPrompt,
    system: str,
    user: str,
) -> dict:
    return {
        "schema": "m5-writer-packet-v2",
        "chapter_id": chapter.chapter_id,
        "system": system,
        "user": user,
    }


def _chapter_contract(chapter: ChapterPrompt) -> str:
    requested_low, requested_high = WRITER_REQUESTED_HAN_CHARS
    include = "\n".join(f"- {item}" for item in chapter.must_include) or "- 无额外项目"
    avoid = "\n".join(f"- {item}" for item in chapter.must_avoid) or "- 无额外项目"
    return (
        f"只撰写第{chapter.chapter_id}章。\n"
        f"章节标题：{chapter.title}\n"
        f"本章目标：{chapter.chapter_goal}\n"
        f"必须包含：\n{include}\n"
        f"必须避免：\n{avoid}\n"
        f"正文长度要求：{requested_low}—{requested_high}个汉字。"
    )
