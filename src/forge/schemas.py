"""Canonical 2D benchmark schema identifiers and artifact validators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REACTION_CASES: tuple[str, ...] = ("gray_scott", "cahn_hilliard")
TURBULENCE_CASES: tuple[str, ...] = ("kolmogorov_ns", "mhd2d")
ALL_2D_CASES: tuple[str, ...] = REACTION_CASES + TURBULENCE_CASES

CANONICAL_CAMPAIGN_VERSIONS: dict[str, str] = {
    "reaction": "reaction-2d-v1",
    "turbulence": "turbulence-2d-v1",
}

CANONICAL_ACTUAL_G_SCHEMAS: dict[str, str] = {
    "reaction": "forge-reaction-actual-g-v1",
    "turbulence": "forge-turbulence-actual-g-v1",
}

CANONICAL_PAIRS_DATASET_SCHEMAS: dict[str, str] = {
    "reaction": "forge-reaction-pairs-v1",
    "turbulence": "forge-turbulence-pairs-v1",
}

CANONICAL_WINDOW_MANIFEST_SCHEMA: str = "forge-parent-windows-v1"
CANONICAL_RESIDUAL_STATISTICS_FILENAME: str = "residual-statistics.json"

CANONICAL_CASE_PROTOCOLS: dict[str, str] = {
    "cahn_hilliard": "forge-ch-v1",
    "gray_scott": "forge-gs-v1",
    "kolmogorov_ns": "forge-ns-v1",
    "mhd2d": "forge-mhd-v1",
}

CANONICAL_CHECKPOINT_SEMANTICS: str = "coarse-plus-0.1-fno-residual"


def case_family(case_name: str) -> str:
    if case_name in REACTION_CASES:
        return "reaction"
    if case_name in TURBULENCE_CASES:
        return "turbulence"
    raise ValueError(f"Unsupported 2D benchmark case: {case_name!r}")


def is_turbulence(case_name: str) -> bool:
    return case_family(case_name) == "turbulence"


def validate_campaign_version(
    version: str | None, cases: set[str] | list[str] | tuple[str, ...]
) -> None:
    case_set = set(cases)
    if not case_set or not case_set.issubset(ALL_2D_CASES):
        raise ValueError("Unsupported 2D benchmark case in campaign config")
    if version == CANONICAL_CAMPAIGN_VERSIONS["reaction"]:
        if not case_set.issubset(REACTION_CASES):
            raise ValueError(
                f"Campaign version {version!r} is restricted to reaction-diffusion cases {REACTION_CASES}"
            )
        return
    if version == CANONICAL_CAMPAIGN_VERSIONS["turbulence"]:
        if not case_set.issubset(TURBULENCE_CASES):
            raise ValueError(
                f"Campaign version {version!r} is restricted to spectral turbulence cases {TURBULENCE_CASES}"
            )
        return
    raise ValueError(f"Unknown or unsupported 2D campaign version: {version!r}")


def canonical_collection_protocol(case_name: str) -> str:
    return CANONICAL_CAMPAIGN_VERSIONS[case_family(case_name)]


def validate_collection_protocol(protocol: str | None, case_name: str) -> None:
    expected = CANONICAL_CAMPAIGN_VERSIONS[case_family(case_name)]
    if protocol != expected:
        raise ValueError(
            f"Unsupported or cross-family collection protocol {protocol!r} for {case_name!r}; expected {expected!r}"
        )


def canonical_actual_g_schema(case_name: str) -> str:
    return CANONICAL_ACTUAL_G_SCHEMAS[case_family(case_name)]


def validate_actual_g_schema(schema: str | None, case_name: str) -> None:
    expected = CANONICAL_ACTUAL_G_SCHEMAS[case_family(case_name)]
    if schema != expected:
        raise ValueError(
            f"Invalid or cross-family actual-G replay schema {schema!r} for {case_name!r}; expected {expected!r}"
        )


def canonical_pairs_dataset_schema(case_name: str) -> str:
    return CANONICAL_PAIRS_DATASET_SCHEMAS[case_family(case_name)]


def validate_pairs_dataset_schema(schema: str | None, case_name: str) -> None:
    expected = CANONICAL_PAIRS_DATASET_SCHEMAS[case_family(case_name)]
    if schema != expected:
        raise ValueError(
            f"Invalid or cross-family paired dataset schema {schema!r} for {case_name!r}; expected {expected!r}"
        )


def validate_window_manifest_schema(schema: str | None, case_name: str) -> None:
    case_family(case_name)
    if schema != CANONICAL_WINDOW_MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported window manifest schema {schema!r} for {case_name!r}; expected {CANONICAL_WINDOW_MANIFEST_SCHEMA!r}"
        )


def canonical_case_protocol(case_name: str) -> str:
    case_family(case_name)
    return CANONICAL_CASE_PROTOCOLS[case_name]


def is_valid_case_protocol(protocol: str | None, case_name: str) -> bool:
    if case_name not in CANONICAL_CASE_PROTOCOLS or not isinstance(protocol, str):
        return False
    return protocol == CANONICAL_CASE_PROTOCOLS[case_name]


def validate_case_protocol(protocol: str | None, case_name: str) -> None:
    case_family(case_name)
    if not is_valid_case_protocol(protocol, case_name):
        expected = CANONICAL_CASE_PROTOCOLS[case_name]
        raise ValueError(
            f"Invalid or cross-case protocol {protocol!r} for {case_name!r}; expected {expected!r}"
        )


def validate_checkpoint_semantics(semantics: str | None) -> None:
    if semantics != CANONICAL_CHECKPOINT_SEMANTICS:
        raise ValueError(f"Unsupported checkpoint semantics {semantics!r}")


def canonical_json_digest(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_statistics_case(payload: Any, case_name: str | None, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Residual statistics file {path} must contain a JSON object")
    if case_name is not None and "case_name" in payload and payload.get("case_name") != case_name:
        raise ValueError(
            f"Cross-case residual statistics in {path}: expected {case_name!r}, got {payload.get('case_name')!r}"
        )
    return payload


def load_residual_statistics(
    case_dir: str | Path, *, case_name: str | None = None
) -> dict[str, Any] | None:
    """Read residual statistics from a dataset directory if present."""
    stats_path = Path(case_dir) / CANONICAL_RESIDUAL_STATISTICS_FILENAME
    if not stats_path.is_file():
        return None
    return _validate_statistics_case(
        json.loads(stats_path.read_text(encoding="utf-8")),
        case_name,
        stats_path,
    )
