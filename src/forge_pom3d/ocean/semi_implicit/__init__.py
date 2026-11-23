"""Semi-implicit 3D Princeton Ocean Model dynamical core."""

from .core import POM2K_Core, SemiImplicitPOM2KCore

SEMI_IMPLICIT_NUMERICS_VERSION = POM2K_Core.NUMERICS_VERSION

__all__ = [
    "POM2K_Core",
    "SEMI_IMPLICIT_NUMERICS_VERSION",
    "SemiImplicitPOM2KCore",
]
