from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChapterPrompt:
    chapter_id: int
    volume_id: int
    title: str
    chapter_goal: str
    must_include: tuple[str, ...]
    must_avoid: tuple[str, ...]
    target_chars: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChapterPrompt":
        return cls(
            chapter_id=int(raw["chapter_id"]),
            volume_id=int(raw["volume_id"]),
            title=str(raw["title"]),
            chapter_goal=str(raw["chapter_goal"]),
            must_include=tuple(str(item) for item in raw.get("must_include", [])),
            must_avoid=tuple(str(item) for item in raw.get("must_avoid", [])),
            target_chars=int(raw.get("target_chars", 2500)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "volume_id": self.volume_id,
            "title": self.title,
            "chapter_goal": self.chapter_goal,
            "must_include": list(self.must_include),
            "must_avoid": list(self.must_avoid),
            "target_chars": self.target_chars,
        }


class M5Dataset:
    """Public generation-side view; hidden hook files are never loaded."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.global_task = json.loads((self.root / "global_task.json").read_text(encoding="utf-8-sig"))
        self.story_bible = (self.root / "story_bible.md").read_text(encoding="utf-8-sig")
        self._chapters = self._load_chapters()
        expected = int(self.global_task.get("total_chapters", 320))
        if sorted(self._chapters) != list(range(1, expected + 1)):
            raise ValueError(f"M5 prompts must contain exactly chapters 1..{expected}")

    def _load_chapters(self) -> dict[int, ChapterPrompt]:
        chapters: dict[int, ChapterPrompt] = {}
        prompt_dir = self.root / "prompts"
        for path in sorted(prompt_dir.glob("chapters_*.jsonl")):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                prompt = ChapterPrompt.from_dict(json.loads(line))
                if prompt.chapter_id in chapters:
                    raise ValueError(f"Duplicate M5 chapter: {prompt.chapter_id}")
                chapters[prompt.chapter_id] = prompt
        return chapters

    @property
    def total_chapters(self) -> int:
        return len(self._chapters)

    def release(self, chapter_id: int) -> ChapterPrompt:
        """Release exactly one current chapter to the scheduler."""
        try:
            return self._chapters[chapter_id]
        except KeyError as exc:
            raise ValueError(f"Unknown chapter {chapter_id}") from exc

    def volume_briefs(self) -> dict[int, str]:
        """Return the public, substantive objective for every L2 volume."""
        numerals = {name: index for index, name in enumerate("一二三四五六七八", 1)}
        pattern = re.compile(
            r"^### 第([一二三四五六七八])卷：([^\n]+)\n\n([^\n]+)",
            flags=re.MULTILINE,
        )
        briefs = {
            numerals[numeral]: f"{title.strip()}：{objective.strip()}"
            for numeral, title, objective in pattern.findall(self.story_bible)
        }
        expected = int(self.global_task.get("total_volumes", 8))
        if sorted(briefs) != list(range(1, expected + 1)):
            raise ValueError("Story bible must define one public objective per volume")
        return briefs

    def public_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.global_task, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(self.story_bible.encode("utf-8"))
        for chapter_id in sorted(self._chapters):
            payload = json.dumps(self._chapters[chapter_id].to_dict(), ensure_ascii=False, sort_keys=True)
            digest.update(payload.encode("utf-8"))
        return digest.hexdigest()


def han_char_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
