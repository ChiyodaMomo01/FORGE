"""Trusted window bank from verified HR replay references."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..core.contracts import ACTIVE_SPLITS, trial_map
from ..core.io import file_digest, resolve_archived_path, save
from ..paired.data import resolve_pair_record_path
from ..schemas import (
    CANONICAL_WINDOW_MANIFEST_SCHEMA,
    validate_pairs_dataset_schema,
    validate_window_manifest_schema,
)


def prepare(original_root, pair_root, output, recipe):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pair_root = Path(pair_root) / recipe.case_name
    dataset_path = pair_root / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset.get("case_name") != recipe.case_name:
        raise ValueError("cross-case dataset rejected")
    validate_pairs_dataset_schema(dataset.get("schema"), recipe.case_name)
    info = {
        "schema": CANONICAL_WINDOW_MANIFEST_SCHEMA,
        "case_name": recipe.case_name,
        "recipe": recipe.dictionary(),
        "splits": {},
        "restart": "public cold restart from restricted HR field; no hidden HR history returned",
        "statistical_unit": "original parent; overlapping windows correlated",
        "source_dataset": f"{recipe.case_name}/dataset.json",
        "source_dataset_sha256": file_digest(dataset_path),
    }
    sums = squares = None
    count = 0
    for split in ACTIVE_SPLITS:
        records = [
            resolve_pair_record_path(pair_root, x) for x in dataset["splits"][split]["records"]
        ]
        ids = [p.parent.name for p in records]
        metadata = json.loads(
            (Path(original_root) / recipe.case_name / split / (ids[0] + ".json")).read_text(
                encoding="utf-8"
            )
        )
        files = []
        row = None
        for path in records:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["frames"] != recipe.frames or row["sample_dt"] != recipe.sample_dt:
                raise ValueError("frozen time contract mismatch")
            fields = path.parent / "reference_lr.npy"
            data = np.load(fields, mmap_mode="r")
            if data.shape != (
                recipe.frames + 1,
                recipe.field_channels,
                row["lr_grid"],
                row["lr_grid"],
            ):
                raise ValueError("reference shape mismatch")
            files.append(str(fields.resolve()))
            if split == "train":
                a = np.asarray(data[::8])
                v = a.sum((0, 2, 3))
                v2 = np.square(a).sum((0, 2, 3))
                sums = v if sums is None else sums + v
                squares = v2 if squares is None else squares + v2
                count += len(a) * a.shape[-1] * a.shape[-2]
        assert row is not None
        info["splits"][split] = {
            "parents": ids,
            "parent_count": len(ids),
            "grid": row["lr_grid"],
            "references": files,
            "window_count": len(ids) * len(recipe.starts),
            "trial_count": len(trial_map(len(ids), recipe)),
            "deployment_config": metadata["deployment_config"],
        }
    info["field_mean"] = (sums / count).tolist()
    info["field_std"] = np.sqrt(np.maximum(squares / count - (sums / count) ** 2, 1e-30)).tolist()
    info["field_scale_source"] = "train reference only, sample stride8, new float64 restricted HR"
    save(output / "window-manifest.json", info)
    return info


class WindowBank:
    def __init__(self, output, recipe):
        self.output = Path(output)
        self.recipe = recipe
        self.info = json.loads((self.output / "window-manifest.json").read_text(encoding="utf-8"))
        self.arrays = {}
        if self.info.get("case_name") != recipe.case_name:
            raise ValueError("unsupported schema or cross-case window bank")
        validate_window_manifest_schema(self.info.get("schema"), recipe.case_name)
        self.maps = {
            s: trial_map(z["parent_count"], recipe) for s, z in self.info["splits"].items()
        }

    def cohort(self, split, trials, device="cpu", positions=None):
        if split not in ACTIVE_SPLITS:
            raise ValueError(f"unsupported split {split!r}; expected one of {ACTIVE_SPLITS}")
        if split not in self.arrays:
            self.arrays[split] = [
                np.load(
                    resolve_archived_path(Path(x) if Path(x).is_absolute() else (self.output / x)),
                    mmap_mode="r",
                )
                for x in self.info["splits"][split]["references"]
            ]
        initial = []
        future = []
        rows = []
        position = 0
        for trial in trials:
            start, group = self.maps[split][int(trial)]
            for p in range(group * 4, group * 4 + 4):
                take = positions is None or position in positions
                position += 1
                if not take:
                    continue
                data = self.arrays[split][p]
                initial.append(data[start])
                future.append(data[start + 1 : start + self.recipe.horizon + 1])
                rows.append(
                    {
                        "parent": p,
                        "parent_id": self.info["splits"][split]["parents"][p],
                        "start": start,
                        "trial": int(trial),
                    }
                )
        return (
            torch.as_tensor(np.stack(initial), dtype=torch.float64, device=device),
            torch.as_tensor(np.stack(future), dtype=torch.float64, device=device),
            rows,
        )
