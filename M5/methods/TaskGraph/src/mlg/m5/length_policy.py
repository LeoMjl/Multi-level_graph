from __future__ import annotations

from typing import Any


LENGTH_POLICY_SCHEMA = "m5-chapter-length-policy-v1"

# What every current-condition writer is explicitly asked to produce.
WRITER_REQUESTED_HAN_CHARS = (2000, 2500)

# A deliberately tolerant mechanical gate. Text inside this band is accepted
# without a retry even when it exceeds the writer-facing target.
MECHANICAL_ACCEPTED_HAN_CHARS = (2000, 3500)

# Historical TaskGraph artifacts were generated and checked with one shared
# 2000..3000 band. The commit guard keeps that interpretation read-only.
LEGACY_WRITER_REQUESTED_HAN_CHARS = (2000, 3000)
LEGACY_MECHANICAL_ACCEPTED_HAN_CHARS = (2000, 3000)

LENGTH_BAND_POLICY_REVISION = (
    "prewrite-three-channel-union-length-band-v12"
)
PREVIOUS_LENGTH_POLICY_REVISION = (
    "prewrite-three-channel-union-lifecycle-retry-v13"
)
CURRENT_LENGTH_POLICY_REVISION = (
    "prewrite-three-channel-union-qwen-embedding-v14"
)


def chapter_length_policy_payload() -> dict[str, Any]:
    return {
        "schema": LENGTH_POLICY_SCHEMA,
        "writer_requested_han_chars": list(WRITER_REQUESTED_HAN_CHARS),
        "mechanical_accepted_han_chars": list(MECHANICAL_ACCEPTED_HAN_CHARS),
        "count_scope": "full_writer_output_including_title_line",
    }


def is_mechanically_accepted(count: int) -> bool:
    low, high = MECHANICAL_ACCEPTED_HAN_CHARS
    return low <= count <= high


def ranges_for_protocol(revision: object) -> tuple[tuple[int, int], tuple[int, int]]:
    if revision in {
        LENGTH_BAND_POLICY_REVISION,
        PREVIOUS_LENGTH_POLICY_REVISION,
        CURRENT_LENGTH_POLICY_REVISION,
    }:
        return WRITER_REQUESTED_HAN_CHARS, MECHANICAL_ACCEPTED_HAN_CHARS
    return (
        LEGACY_WRITER_REQUESTED_HAN_CHARS,
        LEGACY_MECHANICAL_ACCEPTED_HAN_CHARS,
    )
