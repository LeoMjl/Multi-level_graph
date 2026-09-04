"""M5 controlled long-form generation harness with lazy public imports."""

from typing import Any

__all__ = ["M5RunConfig", "run_conditions"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from mlg.m5.harness import M5RunConfig, run_conditions
        return {"M5RunConfig": M5RunConfig, "run_conditions": run_conditions}[name]
    raise AttributeError(name)
