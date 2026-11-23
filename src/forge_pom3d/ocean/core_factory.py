"""Registry for the semi-implicit 3D Princeton Ocean Model dynamical core."""

from __future__ import annotations

from .contracts import CANONICAL_CORE_NAME, normalize_core_name
from .semi_implicit import POM2K_Core as SemiImplicitCore

CORE_REGISTRY = {CANONICAL_CORE_NAME: SemiImplicitCore}


def resolve_core_class(name: str) -> type[SemiImplicitCore]:
    canonical = normalize_core_name(name)
    return CORE_REGISTRY[canonical]


__all__ = ["CORE_REGISTRY", "resolve_core_class"]
