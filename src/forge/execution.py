"""Validation gate verification and non-self-referential source/config digests."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .core.contracts import Recipe

SOURCE_BUNDLE_FORMAT = "forge-source-bundle-v1"
_NON_SCIENTIFIC_KEYS = frozenset(
    {
        "output",
        "output_root",
        "pairs",
        "pairs_root",
        "original",
        "original_root",
        "reference",
        "reference_root",
        "started_unix",
        "finished_unix",
        "host",
        "pid",
        "source_sha256",
        "checkout_source_sha256",
        "source_tree_sha256",
        "effective_config_sha256",
        "source_bundle_sha256",
    }
)


def validation_gate(
    common,
    physics,
    common_sha256,
    case_name="cahn_hilliard",
    initialization="paired_pretrain",
):
    def finite(value):
        return (
            isinstance(value.get("mean_mse"), (int, float))
            and math.isfinite(value["mean_mse"])
            and not value.get("failures")
            and value.get("uncapped_nonfinite_outputs", 0) == 0
        )

    passed = (
        finite(common)
        and finite(physics)
        and (
            math.isclose(common["mean_mse"], physics["mean_mse"], rel_tol=1e-7, abs_tol=1e-12)
            if initialization == "physics_start"
            else common["mean_mse"] < physics["mean_mse"]
        )
    )
    if initialization == "physics_start":
        a, b = common.get("per_window", []), physics.get("per_window", [])
        passed = (
            passed
            and bool(a)
            and len(a) == len(b)
            and all(
                math.isclose(x, y, rel_tol=1e-7, abs_tol=1e-12) for x, y in zip(a, b, strict=True)
            )
        )
    return {
        "status": "passed" if passed else "not_passed",
        "case_name": case_name,
        "common_validation_mse": common.get("mean_mse"),
        "pure_physics_validation_mse": physics.get("mean_mse"),
        "initialization": initialization,
        "criterion": "zero-residual public rollout equals pure physics"
        if initialization == "physics_start"
        else "common improves pure physics",
        "selection": "validation split only",
        "common_sha256": common_sha256,
        "post_training_authorized_by_gate": passed,
    }


def compute_source_tree_sha256(pkg_dir: str | Path | None = None) -> str:
    """Compute canonical SHA-256 over normalized relative source paths and contents."""
    target = Path(pkg_dir).resolve() if pkg_dir is not None else Path(__file__).resolve().parent
    anchor = target.parent
    h = hashlib.sha256()
    for p in sorted(target.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(anchor).as_posix()
        f_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{rel}\0{f_sha}\n".encode("utf-8"))
    return h.hexdigest()


def compute_effective_config_sha256(
    recipe: Recipe | dict[str, Any], *, method: str | None = None
) -> str:
    """Compute non-self-referential SHA-256 of the effective scientific configuration."""
    raw = recipe.dictionary() if isinstance(recipe, Recipe) else dict(recipe)
    scientific = {k: v for k, v in raw.items() if k not in _NON_SCIENTIFIC_KEYS}
    payload: dict[str, Any] = {"recipe": scientific}
    if method is not None:
        payload["method"] = str(method)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_source_bundle_sha256(source_tree_sha256: str, effective_config_sha256: str) -> str:
    """Combine source_tree_sha256 and effective_config_sha256 without self-reference."""
    payload = {
        "bundle_format": SOURCE_BUNDLE_FORMAT,
        "source_tree_sha256": str(source_tree_sha256),
        "effective_config_sha256": str(effective_config_sha256),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def executable_digest(base: str | Path | None = None) -> str:
    if base is not None:
        base_path = Path(base)
        if (base_path / ".git").exists():
            snapshot = subprocess.check_output(
                ["git", "ls-tree", "HEAD", "src", "configs", "pyproject.toml"],
                cwd=base_path,
            )
            return hashlib.sha256(snapshot).hexdigest()
    return compute_source_tree_sha256()
