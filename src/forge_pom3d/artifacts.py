"""Atomic artifact persistence, path sanitization, and non-self-referential source/config digests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .config import ForgePOM3DConfig

DIGEST_SCHEMA_VERSION = "forge-pom3d-digest-v1"

EXCLUDED_EFFECTIVE_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "dataset_root",
        "output_root",
        "input_stats_path",
        "common_ckpt",
        "source_version",
        "source_sha256",
        "source_tree_sha256",
        "effective_config_sha256",
        "source_bundle_sha256",
        "common_hash",
    }
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def atomic_torch_save(state_dict: dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(state_dict, tmp_path)
    tmp_path.replace(target_path)


def all_numeric_leaves_finite(obj: Any) -> bool:
    """Return True iff every numeric leaf in nested dicts/lists/tuples is finite."""
    if obj is None or isinstance(obj, (str, bool)):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(float(obj))
    if isinstance(obj, dict):
        return all(all_numeric_leaves_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_numeric_leaves_finite(v) for v in obj)
    return True


def relativize_artifact_path(target_path: str | Path, base_dir: str | Path | None = None) -> str:
    """Return a portable relative path (never leaking host absolute filesystem roots)."""
    p = Path(target_path)
    if base_dir is not None:
        try:
            rel = p.resolve().relative_to(Path(base_dir).resolve())
            return rel.as_posix()
        except ValueError:
            pass
    return p.name


def compute_source_tree_sha256() -> str:
    """Compute deterministic SHA-256 over all `.py` files in `forge_pom3d/` (including `ocean/`)."""
    pkg_dir = Path(__file__).resolve().parent
    src_parent = pkg_dir.parent
    h = hashlib.sha256()
    py_files = [
        p
        for p in pkg_dir.rglob("*.py")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("._")
    ]
    for p in sorted(py_files, key=lambda item: item.relative_to(src_parent).as_posix()):
        rel = p.relative_to(src_parent).as_posix()
        file_sha = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{rel}\0{file_sha}\n".encode())
    return h.hexdigest()


def extract_effective_scientific_config(cfg: ForgePOM3DConfig) -> dict[str, Any]:
    """Extract canonical scientific parameters & algorithm switches, excluding paths and digests."""
    raw = cfg.to_dict()
    return {k: v for k, v in sorted(raw.items()) if k not in EXCLUDED_EFFECTIVE_CONFIG_FIELDS}


def compute_effective_config_sha256(cfg: ForgePOM3DConfig) -> str:
    """Compute deterministic SHA-256 of canonical scientific parameters (non-self-referential)."""
    scientific = extract_effective_scientific_config(cfg)
    canonical_bytes = json.dumps(
        scientific, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def compute_source_bundle_sha256(
    source_tree_sha256: str,
    effective_config_sha256: str,
    schema_version: str = DIGEST_SCHEMA_VERSION,
) -> str:
    """Combine `source_tree_sha256`, `effective_config_sha256`, and schema version tag."""
    payload = {
        "effective_config_sha256": str(effective_config_sha256),
        "schema_version": str(schema_version),
        "source_tree_sha256": str(source_tree_sha256),
    }
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def compute_source_digests(cfg: ForgePOM3DConfig) -> dict[str, str]:
    """Compute and attach the 3-part non-self-referential source and config digests."""
    tree_sha = compute_source_tree_sha256()
    cfg_sha = compute_effective_config_sha256(cfg)
    bundle_sha = compute_source_bundle_sha256(tree_sha, cfg_sha, DIGEST_SCHEMA_VERSION)
    version_tag = f"src-{bundle_sha[:12]}"
    cfg.source_tree_sha256 = tree_sha
    cfg.effective_config_sha256 = cfg_sha
    cfg.source_bundle_sha256 = bundle_sha
    cfg.source_sha256 = bundle_sha
    cfg.source_version = version_tag
    return {
        "digest_schema_version": DIGEST_SCHEMA_VERSION,
        "source_tree_sha256": tree_sha,
        "effective_config_sha256": cfg_sha,
        "source_bundle_sha256": bundle_sha,
        "source_sha256": bundle_sha,
        "source_version": version_tag,
    }


__all__ = [
    "DIGEST_SCHEMA_VERSION",
    "EXCLUDED_EFFECTIVE_CONFIG_FIELDS",
    "all_numeric_leaves_finite",
    "append_jsonl",
    "atomic_torch_save",
    "compute_effective_config_sha256",
    "compute_source_bundle_sha256",
    "compute_source_digests",
    "compute_source_tree_sha256",
    "extract_effective_scientific_config",
    "relativize_artifact_path",
    "save_atomic_json",
    "sha256_file",
]
