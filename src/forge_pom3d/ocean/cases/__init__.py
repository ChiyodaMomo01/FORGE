"""Case configuration registry for the 3D ocean model."""

from .catalog import (
    CASE_SETTINGS_REGISTRY,
    CASE_SPEC_REGISTRY,
    get_case_settings,
    resolve_case_spec,
)
from .common import CaseSettings, CaseSetup, CaseSpec

__all__ = [
    "CASE_SETTINGS_REGISTRY",
    "CASE_SPEC_REGISTRY",
    "CaseSettings",
    "CaseSetup",
    "CaseSpec",
    "get_case_settings",
    "resolve_case_spec",
]
