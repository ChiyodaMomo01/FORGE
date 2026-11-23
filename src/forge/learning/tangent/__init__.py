"""Identify, invert and validate interval responses using public arrays."""

from __future__ import annotations

import importlib

__all__ = [
    "ClosedLoopResponse",
    "IntervalTrace",
    "SplitTangent",
    "inverse_actions",
    "long_rollout_check",
    "output_difference",
    "validate_response",
]

_EXPORTS = {
    "ClosedLoopResponse": ".inverse",
    "inverse_actions": ".inverse",
    "SplitTangent": ".model",
    "IntervalTrace": ".traces",
    "output_difference": ".traces",
    "long_rollout_check": ".validation",
    "validate_response": ".validation",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    if name in ("fitting", "gain_clamp", "inverse", "model", "refine", "traces", "validation"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
