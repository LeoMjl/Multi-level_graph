"""Allow slow Windows CLI initialization without weakening isolation checks."""
from __future__ import annotations

PREFLIGHT_TIMEOUT_SECONDS = 300


class PreflightSubprocess:
    """Proxy only the isolation module's sandbox probe, not model calls."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) > 1
                and command[1] == 'sandbox' and kwargs.get('timeout') == 60):
            kwargs['timeout'] = PREFLIGHT_TIMEOUT_SECONDS
        return self.original.run(command, *args, **kwargs)


def configure_preflight_timeout():
    from mlg.m5 import codex_formal_isolation

    if not isinstance(codex_formal_isolation.subprocess, PreflightSubprocess):
        codex_formal_isolation.subprocess = PreflightSubprocess(
            codex_formal_isolation.subprocess)
    return PREFLIGHT_TIMEOUT_SECONDS
