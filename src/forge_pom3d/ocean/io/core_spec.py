"""Numerical identities and verified runtime settings for the semi-implicit 3D POM core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import (
    CANONICAL_CORE_NAME,
    CANONICAL_NUMERICS_VERSION,
    normalize_core_name,
)
from ..core_factory import resolve_core_class


@dataclass(frozen=True)
class CoreSpec:
    name: str
    numerics_version: str
    physical_step_seconds: int = 600
    external_step_seconds: int = 20
    split_count: int = 30
    alpha: float = 0.333
    smoth: float = 0.15
    horcon: float = 0.05

    @property
    def core_class(self) -> Any:
        return resolve_core_class(self.name)

    def configure(self, core: Any) -> None:
        core.spatial_shapiro_enabled = False
        core.barotropic_shapiro = 0.0
        core.tracer_shapiro = 0.0
        core.velocity_shapiro = 0.0
        core.eta_solver_rel_tol = 1e-5
        core.eta_solver_preconditioner = "spectral"
        core.horcon = self.horcon
        core.alpha = self.alpha
        core.smoth = self.smoth
        core.prtd1 = core.prtd2 = 1e9
        core.iprint = 10**9
        if (
            core.NUMERICS_VERSION != self.numerics_version
            or float(core.dti) != float(self.physical_step_seconds)
            or float(core.dte) != float(self.external_step_seconds)
            or int(core.isplit) != int(self.split_count)
            or abs(float(core.smoth) - float(self.smoth)) > 1e-12
            or int(core.kb) != 16
            or int(core.kbm1) != 15
        ):
            raise ValueError(f"{self.name} core does not match the registered numerical settings")


CORE_SPECS: dict[str, CoreSpec] = {
    CANONICAL_CORE_NAME: CoreSpec(
        name=CANONICAL_CORE_NAME,
        numerics_version=CANONICAL_NUMERICS_VERSION,
        physical_step_seconds=600,
        external_step_seconds=20,
        split_count=30,
        alpha=0.333,
        smoth=0.15,
        horcon=0.05,
    ),
}


def core_spec(name: str = CANONICAL_CORE_NAME) -> CoreSpec:
    canonical = normalize_core_name(name)
    return CORE_SPECS[canonical]


__all__ = ["CORE_SPECS", "CoreSpec", "core_spec"]
