"""Canonical identifiers, contract validation, and environment helpers for 3D POM."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

CANONICAL_CORE_NAME = "semi_implicit"
CANONICAL_NUMERICS_VERSION = "semi-implicit-dg-v2-barotropic-coupled"
CANONICAL_GENERATION_VARIANT = "semi-implicit-barotropic-coupled-no-spatial-shapiro"
CANONICAL_CONTRACT_VERSION = "forge-pom3d-v1-endpoint"
CANONICAL_MODEL_CONTRACT_VERSION = "semi-implicit-endpoint-v1"

_BOOL_FALSE_TOKENS = frozenset({"0", "false", "off", "no"})
_BOOL_TRUE_TOKENS = frozenset({"1", "true", "on", "yes"})


def _normalize_bool_token(raw: str, var_name: str) -> str:
    token = raw.strip().lower()
    if token in _BOOL_FALSE_TOKENS:
        return "0"
    if token in _BOOL_TRUE_TOKENS:
        return "1"
    raise ValueError(f"{var_name} must be one of 0/1, false/true, off/on, or no/yes; got {raw!r}")


def resolve_env_var(
    canonical_name: str,
    default: str = "",
    environ: Mapping[str, str] | None = None,
    *,
    kind: str = "str",
) -> str:
    """Resolve a canonical `FORGE_POM3D_*` environment variable without mutating `os.environ`."""
    env = os.environ if environ is None else environ
    if canonical_name in env:
        raw_val = str(env[canonical_name])
        if kind == "bool":
            _normalize_bool_token(raw_val, canonical_name)
        return raw_val
    return default


def normalize_core_name(name: Any) -> str:
    """Validate and return the canonical ocean core identifier (`'semi_implicit'`)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Unsupported ocean core {name!r}; expected {CANONICAL_CORE_NAME!r}")
    token = name.strip()
    if token == CANONICAL_CORE_NAME:
        return CANONICAL_CORE_NAME
    raise ValueError(f"Unsupported ocean core {name!r}; expected {CANONICAL_CORE_NAME!r}")


def validate_stored_core_identity(
    requested_core: Any,
    numerics_version: Any,
) -> tuple[str, str]:
    """Validate frozen dataset or provenance core/numerics pair without mutating files."""
    if not isinstance(requested_core, str) or not isinstance(numerics_version, str):
        raise ValueError(
            f"Dataset core identity missing or invalid: core={requested_core!r}, "
            f"numerics_version={numerics_version!r}"
        )
    core_tok = requested_core.strip()
    num_tok = numerics_version.strip()

    if core_tok == CANONICAL_CORE_NAME and num_tok == CANONICAL_NUMERICS_VERSION:
        return CANONICAL_CORE_NAME, CANONICAL_NUMERICS_VERSION

    raise ValueError(
        f"Dataset core/numerics mismatch or unsupported spec: "
        f"core={requested_core!r}, numerics_version={numerics_version!r}"
    )


def validate_restart_contract(
    stored_contract: Any,
    expected_contract: str = CANONICAL_CONTRACT_VERSION,
) -> str:
    """Validate restart checkpoint contract string against the canonical contract."""
    if isinstance(stored_contract, str) and stored_contract.strip() == expected_contract:
        return expected_contract
    raise ValueError(
        f"Restart contract mismatch: got {stored_contract!r}, expected {expected_contract!r}"
    )


def normalize_generation_variant(variant: Any) -> str | None:
    """Normalize dataset generation variant tag for metadata reporting."""
    if variant is None:
        return None
    if isinstance(variant, str):
        return variant.strip()
    return str(variant)


__all__ = [
    "CANONICAL_CONTRACT_VERSION",
    "CANONICAL_CORE_NAME",
    "CANONICAL_GENERATION_VARIANT",
    "CANONICAL_MODEL_CONTRACT_VERSION",
    "CANONICAL_NUMERICS_VERSION",
    "normalize_core_name",
    "normalize_generation_variant",
    "resolve_env_var",
    "validate_restart_contract",
    "validate_stored_core_identity",
]
