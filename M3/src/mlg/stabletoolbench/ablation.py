from __future__ import annotations

from dataclasses import asdict, dataclass


ABLATION_PROTOCOL = "taskgraph_ablation_v1"
ABLATION_MODES = (
    "full",
    "static_l3",
    "hard_dependency_only",
    "no_structured_l4",
    "flat_context",
)


@dataclass(frozen=True)
class TaskGraphAblation:
    """One-factor-at-a-time M3 ablation configuration."""

    mode: str = "full"
    sampling_seed: int = 20260831
    action_temperature: float = 0.0
    action_context_chars: int = 6000

    def __post_init__(self) -> None:
        if self.mode not in ABLATION_MODES:
            raise ValueError(
                f"unsupported ablation mode {self.mode!r}; "
                f"expected one of {ABLATION_MODES}"
            )
        if self.action_temperature != 0.0:
            raise ValueError("formal ablations require action_temperature=0")
        if self.action_context_chars != 6000:
            raise ValueError("formal ablations require a 6000-character context budget")

    @property
    def expand_all_l3_before_execution(self) -> bool:
        return self.mode == "static_l3"

    @property
    def hard_dependencies_only(self) -> bool:
        return self.mode == "hard_dependency_only"

    @property
    def materialize_structured_l4(self) -> bool:
        return self.mode != "no_structured_l4"

    @property
    def flat_action_context(self) -> bool:
        return self.mode == "flat_context"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update({
            "protocol": ABLATION_PROTOCOL,
            "expansion_policy": (
                "static_pre_execution"
                if self.expand_all_l3_before_execution
                else "online_progressive"
            ),
            "dependency_policy": (
                "explicit_hard_only"
                if self.hard_dependencies_only
                else "three_channel_union_then_agent_filter"
            ),
            "structured_l4": self.materialize_structured_l4,
            "action_context": (
                "chronological_flat_history"
                if self.flat_action_context
                else "step_anchored_pruned_subgraph"
            ),
        })
        return payload
