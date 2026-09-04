from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperDependencyConfig:
    """Task-specific values for the paper's Vc/Cdep/Vr dependency method."""

    structural_window: int = 12
    semantic_threshold: float = 0.35
    reference_threshold: int = 2
    semantic_top_k: int = 128
    candidate_limit: int = 192

    def __post_init__(self) -> None:
        if self.structural_window < 0:
            raise ValueError("structural_window must be non-negative")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be between 0 and 1")
        if self.reference_threshold < 0:
            raise ValueError("reference_threshold must be non-negative")
        if self.semantic_top_k < 1 or self.candidate_limit < 1:
            raise ValueError("dependency candidate limits must be positive")
