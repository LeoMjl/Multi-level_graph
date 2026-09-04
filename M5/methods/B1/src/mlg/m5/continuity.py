from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from mlg.m5.memory import count_tokens, truncate_tokens


LOCAL_CONTINUITY_TOKEN_BUDGET = 6000


@dataclass(frozen=True)
class LocalContinuity:
    chapter_id: int | None = None
    text: str = ""
    source_sha256: str = ""
    source_tokens: int = 0
    prompt_tokens: int = 0
    truncated: bool = False

    def audit(self) -> dict[str, object]:
        return {
            "source_chapter": self.chapter_id,
            "source_sha256": self.source_sha256,
            "source_tokens": self.source_tokens,
            "prompt_tokens": self.prompt_tokens,
            "truncated": self.truncated,
        }


def load_local_continuity(
    run_dir: Path,
    chapter_id: int,
    *,
    token_budget: int = LOCAL_CONTINUITY_TOKEN_BUDGET,
) -> LocalContinuity:
    """Load exactly t-1 from committed output; never infer it from method memory."""
    if chapter_id <= 1:
        return LocalContinuity()
    path = run_dir / "chapters" / f"chapter_{chapter_id - 1:03d}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing committed previous chapter for chapter {chapter_id}: {path}"
        )
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError(f"Previous chapter is empty: {path}")
    source_tokens = count_tokens(raw)
    text = truncate_tokens(raw, token_budget, keep_end=True)
    return LocalContinuity(
        chapter_id=chapter_id - 1,
        text=text,
        source_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        source_tokens=source_tokens,
        prompt_tokens=count_tokens(text),
        truncated=source_tokens > token_budget,
    )
