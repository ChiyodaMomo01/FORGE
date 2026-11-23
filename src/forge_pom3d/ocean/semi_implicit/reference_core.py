from __future__ import annotations

from .baroclinic import BaroclinicMixin
from .base import BaseCore
from .case_policy import CasePolicyMixin
from .cases import CasesMixin
from .checkpointing import CheckpointingMixin
from .diagnostics import DiagnosticsMixin
from .grid import GridMixin
from .implicit_barotropic import ImplicitBarotropicMixin
from .initialization import InitializationMixin
from .momentum import MomentumMixin
from .reference_base import ReferenceCoreBase
from .time_loop import TimeLoopMixin
from .tracers import TracersMixin


class POM2K_Core(
    GridMixin,
    CasePolicyMixin,
    CasesMixin,
    InitializationMixin,
    ImplicitBarotropicMixin,
    BaroclinicMixin,
    TracersMixin,
    MomentumMixin,
    DiagnosticsMixin,
    CheckpointingMixin,
    TimeLoopMixin,
    ReferenceCoreBase,
    BaseCore,
):
    pass
