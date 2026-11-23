"""Frozen-array dataset, disjoint parents and train-only statistics.

No torch checkpoint or physics implementation is needed to open pair data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.io import file_digest, resolve_archived_path, save
from ..schemas import (
    CANONICAL_RESIDUAL_STATISTICS_FILENAME,
    canonical_pairs_dataset_schema,
    is_turbulence,
    load_residual_statistics,
    validate_actual_g_schema,
    validate_collection_protocol,
    validate_pairs_dataset_schema,
)
from .coarse import coarse_value

EXPECTED_SPLITS = {
    "cahn_hilliard": (("train", 48), ("validation", 16), ("test_id", 16)),
    "gray_scott": (("train", 48), ("validation", 16), ("test_id", 16)),
    "kolmogorov_ns": (("train", 24), ("validation", 8), ("test_id", 8)),
    "mhd2d": (("train", 24), ("validation", 8), ("test_id", 8)),
}


def resolve_pair_record_path(case_root: str | Path, entry: str | Path) -> Path:
    raw = Path(entry)
    if raw.is_absolute():
        return resolve_archived_path(raw)
    return resolve_archived_path(Path(case_root) / raw)


def _assert_array_all_finite(z: np.ndarray, label: str, chunk: int = 16) -> None:
    if z.ndim == 0:
        if not bool(np.isfinite(z)):
            raise ValueError(f"nonfinite element in {label}")
        return
    for start in range(0, z.shape[0], chunk):
        block = np.asarray(z[start : start + chunk])
        if not np.isfinite(block).all():
            raise ValueError(f"nonfinite element in {label} at slice {start}:{start + chunk}")


def audit(root, case_name):
    root = Path(root) / case_name
    collection = json.loads((root / "collection.json").read_text(encoding="utf-8"))
    if collection["status"] != "complete" or collection["failures"]:
        raise ValueError("complete verified collection required before learning")
    if case_name not in EXPECTED_SPLITS:
        raise ValueError(f"unsupported case for paired audit: {case_name}")
    if "protocol" in collection:
        validate_collection_protocol(collection.get("protocol"), case_name)
    seen = set()
    splits = {}
    turbulence = is_turbulence(case_name)
    official_split = collection.get("official_benchmark_split", True) is not False
    if official_split:
        split_expectations = EXPECTED_SPLITS[case_name]
    else:
        custom_counts = collection.get("custom_split_counts")
        if not isinstance(custom_counts, dict):
            raise ValueError("custom_split_counts dict required for non-official split")
        split_expectations = tuple(
            (sp, int(custom_counts[sp])) for sp in ("train", "validation", "test_id")
        )
        if any(cnt <= 0 or cnt % 4 != 0 for _, cnt in split_expectations):
            raise ValueError("custom split counts must be positive multiples of 4")
    row = None
    for split, expected in split_expectations:
        records = sorted((root / split).glob("*/complete.json"))
        if len(records) != expected:
            raise ValueError("parent count mismatch")
        identities = []
        pair_counts = []
        for path in records:
            row = json.loads(path.read_text(encoding="utf-8"))
            validate_actual_g_schema(row.get("schema"), case_name)
            if (
                row["case"] != case_name
                or row["split"] != split
                or not row["original_lr_bitwise_equal"]
            ):
                raise ValueError("invalid collection provenance")
            if row["seed"] in seen:
                raise ValueError("parent split leakage")
            seen.add(row["seed"])
            identities.append(row["seed"])
            pair_counts.append(row["pairs"])
            expected_pairs = (
                len(range(0, row["frames"], int(row.get("pair_stride", 1))))
                if turbulence
                else row["frames"]
            )
            if split != "test_id" and row["pairs"] != expected_pairs:
                raise ValueError("missing G pairs")
            if split == "test_id" and row["pairs"]:
                raise ValueError("ID pairs are not authorized for learning")
            if turbulence:
                expected_keys = (
                    {
                        "reference_hr_audit",
                        "reference_lr",
                        "input_hr_audit",
                        "output_hr_audit",
                        "input_lr",
                        "output_lr",
                    }
                    if row["pairs"]
                    else {"reference_hr_audit", "reference_lr"}
                )
                if set(row["arrays"].keys()) != expected_keys:
                    raise ValueError("missing or unexpected turbulence array keys")
            for key in row["arrays"]:
                z = np.load(path.parent / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                if list(z.shape) != row["arrays"][key]["shape"] or z.dtype != np.float64:
                    raise ValueError("array/header mismatch")
                _assert_array_all_finite(z, f"{split}/{path.parent.name}/{key}.npy")
        splits[split] = {
            "parents": identities,
            "records": [p.relative_to(root).as_posix() for p in records],
            "pair_counts": pair_counts,
        }
    assert row is not None
    result = {
        "schema": canonical_pairs_dataset_schema(case_name),
        "case_name": case_name,
        "official_benchmark_split": official_split,
        "splits": splits,
        "lr_grid": row["lr_grid"],
        "collection_sha256": file_digest(root / "collection.json"),
        "label_source": "actual consumed G return, never closure_target",
        "field_semantics": "HR reference restricted by frozen case-specific operator",
    }
    save(root / "dataset.json", result)
    return result


class PairDataset(Dataset):
    def __init__(self, root, case_name, split):
        if split not in ("train", "validation"):
            raise ValueError("pretraining uses training/validation splits only")
        self.root = Path(root) / case_name
        info = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        if info.get("case_name") != case_name:
            raise ValueError("cross-case data rejected")
        validate_pairs_dataset_schema(info.get("schema"), case_name)
        self.n = info["lr_grid"]
        self.records = [
            resolve_pair_record_path(self.root, x) for x in info["splits"][split]["records"]
        ]
        self.lengths = info["splits"][split]["pair_counts"]
        self.offsets = np.cumsum([0, *self.lengths])
        self.arrays = {}

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        p = int(np.searchsorted(self.offsets, index, side="right") - 1)
        if p not in self.arrays:
            folder = self.records[p].parent
            self.arrays[p] = [
                np.load(folder / (key + "_lr.npy"), mmap_mode="r") for key in ("input", "output")
            ]
        k = int(index - self.offsets[p])
        x, y = self.arrays[p]
        return torch.from_numpy(x[k].copy()), torch.from_numpy(y[k].copy())


def statistics(root, case_name, output=None):
    dataset = PairDataset(root, case_name, "train")
    if output is None:
        load_residual_statistics(dataset.root, case_name=case_name)
    sums = squares = output_squares = residual_sums = residual_squares = None
    count = 0
    for record in dataset.records:
        folder = record.parent
        x = np.load(folder / "input_lr.npy", mmap_mode="r")
        y = np.load(folder / "output_lr.npy", mmap_mode="r")
        for k in range(0, len(x), 8):
            a = np.asarray(x[k : k + 8])
            b = np.asarray(y[k : k + 8])
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("nonfinite frozen pair")
            residual = b - coarse_value(case_name, torch.from_numpy(a.copy())).numpy()
            u = a.sum((0, 2, 3))
            v = np.square(a).sum((0, 2, 3))
            w = np.square(b).sum((0, 2, 3))
            rs = residual.sum((0, 2, 3))
            rr = np.square(residual).sum((0, 2, 3))
            residual_sums = rs if residual_sums is None else residual_sums + rs
            residual_squares = rr if residual_squares is None else residual_squares + rr
            sums = u if sums is None else sums + u
            squares = v if squares is None else squares + v
            output_squares = w if output_squares is None else output_squares + w
            count += len(a) * a.shape[-1] * a.shape[-2]
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0.0))
    std = np.maximum(std, np.maximum(np.sqrt(squares / count) * 1e-6, 1e-30))
    rms = np.maximum(np.sqrt(output_squares / count), 1e-30)
    result = {
        "case_name": case_name,
        "source": "ALL train pairs only",
        "parents": [p.parent.name for p in dataset.records],
        "samples": len(dataset),
        "input_mean": mean.tolist(),
        "input_std": std.tolist(),
        "output_rms": rms.tolist(),
        "residual_mean": (residual_sums / count).tolist(),
        "residual_std": np.sqrt(
            np.maximum(residual_squares / count - (residual_sums / count) ** 2, 0.0)
        ).tolist(),
        "residual_rms": np.maximum(np.sqrt(residual_squares / count), 1e-30).tolist(),
        "residual_definition": "R(actual G_HR output)-public G_coarse(R(actual G_HR input))",
        "residual_centering": "zero-centered RMS scaling preserves zero-initialization at pure physics",
        "constant_input_channels": np.flatnonzero(
            std <= np.maximum(np.sqrt(squares / count) * 1e-6, 1e-30)
        ).tolist(),
        "dataset_sha256": file_digest(dataset.root / "dataset.json"),
    }
    if output is not False:
        save(
            Path(output)
            if output is not None
            else dataset.root / CANONICAL_RESIDUAL_STATISTICS_FILENAME,
            result,
        )
    return result
