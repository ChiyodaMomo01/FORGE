"""Atomic JSON/file digests and compressed array readers shared across all layers."""

from __future__ import annotations

import gzip
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def resolve_archived_path(path):
    return Path(path)


def decode_compressed_array(raw):
    if getattr(raw.dtype, "names", None) == ("forge_bshuf_f32",):
        shuffled = raw["forge_bshuf_f32"]
        d32 = (
            np.ascontiguousarray(shuffled.reshape(4, -1).T)
            .view(np.uint32)
            .reshape(shuffled.shape[1:])
        )
        if d32.ndim >= 2:
            d32[..., :, 0] = np.cumsum(d32[..., :, 0], axis=-1, dtype=np.uint32)
        u32 = np.cumsum(d32, axis=-1, dtype=np.uint32)
        moved = u32.view(np.float32)
        return np.ascontiguousarray(np.moveaxis(moved, -1, 0))
    return raw


def load_array(path, mmap_mode=None, allow_pickle=False):
    resolved = resolve_archived_path(path)
    if resolved.exists():
        if resolved.suffix == ".gz":
            with gzip.open(resolved, "rb") as stream:
                return decode_compressed_array(np.load(stream, allow_pickle=allow_pickle))
        return np.load(resolved, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
    with gzip.open(resolved.with_suffix(resolved.suffix + ".gz"), "rb") as stream:
        return decode_compressed_array(np.load(stream, allow_pickle=allow_pickle))


def save(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def save_clean_json(path, value: Any) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(target)


_PATH_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "curves_path",
        "curves_paths",
        "residual_path",
        "residual_paths",
        "event_paths",
        "trace_paths",
        "sealed_manifest",
        "query_dir",
    }
)


def _relativize_path_str(raw: str, anchor: Path | None) -> str:
    if not raw or not raw.startswith("/"):
        return raw
    p = Path(raw)
    if anchor is not None:
        try:
            return p.resolve().relative_to(anchor.resolve()).as_posix()
        except ValueError:
            pass
    return p.name


def sanitize_persisted_paths(value: Any, anchor: str | Path | None = None) -> Any:
    """Recursively replace host absolute paths in transient score records with relative names."""
    anchor_p = Path(anchor) if anchor is not None else None

    def _walk(obj: Any, parent_key: str | None = None) -> Any:
        if isinstance(obj, dict):
            return {k: _walk(v, str(k)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v, parent_key) for v in obj]
        if isinstance(obj, tuple):
            return [_walk(v, parent_key) for v in obj]
        if isinstance(obj, str) and parent_key in _PATH_FIELD_KEYS:
            return _relativize_path_str(obj, anchor_p)
        return obj

    return _walk(value)


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()
