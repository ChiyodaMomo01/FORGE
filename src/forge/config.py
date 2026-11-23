"""Explicit versioned campaigns; independent parent splits precede windows."""

from __future__ import annotations

import json
from pathlib import Path

from .core.io import save as save_json
from .schemas import (
    ALL_2D_CASES,
    REACTION_CASES,
    TURBULENCE_CASES,
    is_turbulence,
    validate_campaign_version,
)

ALL_CASES = ALL_2D_CASES
CASES = ALL_CASES
COMPONENT_LABEL_CHANNELS = {
    "gray_scott": 1,
    "cahn_hilliard": 1,
    "kolmogorov_ns": 2,
    "mhd2d": 3,
}


def load_campaign(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("cases"), dict):
        raise ValueError("campaign config must define a cases object")
    validate_campaign_version(config.get("version"), set(config["cases"]))
    if set(config.get("splits", {})) != {"train", "validation", "test_id"}:
        raise ValueError("train, validation, and test_id splits required")
    if any(type(n) is not int or n < 1 for n in config["splits"].values()):
        raise ValueError("positive parent counts required")
    for name, case in config["cases"].items():
        if (
            case["hr_grid"] % config["lr_grid"]
            or case["frames"] < 1
            or case["sample_dt"] <= 0
            or case["burn_in"] < 0
        ):
            raise ValueError(f"invalid space/time configuration for {name}")
    return config


def parent_tasks(config):
    tasks = []
    for name, case in config["cases"].items():
        offset = 0
        for split, count in config["splits"].items():
            for index in range(count):
                seed = case["seed_base"] + offset + index
                reference = dict(case["reference_config"])
                deployment = dict(case["deployment_config"])
                tasks.append(
                    {
                        "case": name,
                        "split": split,
                        "seed": seed,
                        "parent": f"{seed:07d}",
                        "version": config["version"],
                        "lr_grid": config["lr_grid"],
                        "dtype": config["dtype"],
                        "reference_origin": case.get(
                            "reference_origin", config["reference_origin"]
                        ),
                        **(
                            {"reference_revision": case["reference_revision"]}
                            if "reference_revision" in case
                            else {}
                        ),
                        **{k: case[k] for k in ("hr_grid", "frames", "sample_dt", "burn_in")},
                        "reference_config": {**reference, "ood": False},
                        "deployment_config": {**deployment, "ood": False},
                    }
                )
            offset += count
    return tasks


__all__ = [
    "ALL_CASES",
    "CASES",
    "COMPONENT_LABEL_CHANNELS",
    "REACTION_CASES",
    "TURBULENCE_CASES",
    "is_turbulence",
    "load_campaign",
    "parent_tasks",
    "save_json",
]
