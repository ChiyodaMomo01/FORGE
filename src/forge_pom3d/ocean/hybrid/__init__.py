"""Shared fixed-core initialization, input packing, and state injection."""

from .core_pool import build_initialized_core
from .state import (
    StateNormalizer,
    inject_core_state,
    pack_pom_batch,
    pack_state_tensors,
)

__all__ = [
    "StateNormalizer",
    "build_initialized_core",
    "inject_core_state",
    "pack_pom_batch",
    "pack_state_tensors",
]
