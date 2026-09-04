from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperDependencyConfig:
    """Task-specific values for the paper's Vc/Cdep/Vr dependency method."""

    structural_window: int = 12
    semantic_threshold: float = 0.35
    semantic_top_k: int = 128
    candidate_limit: int = 192
    active_pool_limit: int = 100
    activity_half_life: float = 40.0
    activity_unresolved_weight: float = 0.45
    activity_frequency_weight: float = 0.30
    activity_importance_weight: float = 0.25
    activity_frequency_saturation: float = 3.0

    def __post_init__(self) -> None:
        if self.structural_window < 0:
            raise ValueError("structural_window must be non-negative")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be between 0 and 1")
        if self.semantic_top_k < 1 or self.candidate_limit < 1:
            raise ValueError("dependency candidate limits must be positive")
        if not 1 <= self.active_pool_limit <= 100:
            raise ValueError("active_pool_limit must be between 1 and 100")
        if self.activity_half_life <= 0:
            raise ValueError("activity_half_life must be positive")
        weights = (
            self.activity_unresolved_weight,
            self.activity_frequency_weight,
            self.activity_importance_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("activity weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("activity weights must sum to one")
        if self.activity_frequency_saturation <= 0:
            raise ValueError("activity_frequency_saturation must be positive")
